#!/usr/bin/env python3
"""Overfit N0 on one fixed batch (test plan, step 3).

Takes one deterministic batch (the validation tiles of the day --date with
the most targets), trains the
N0 model of config/xp/N0.yaml on it for --steps steps (Adam, constant lr, no
warm-up, bf16 autocast on GPU, EMA = student), and reports:
  * the FM loss (running mean) per level: it should drop well below its
    starting value;
  * peak GPU memory and time per step (to size batch / Swin depth);
  * ensemble scores of the trained model on the same batch: they should be
    much better than at initialisation.

Usage:
    python scripts/overfit.py --date 2022-02-15 --levels C --no-pyramid
    python scripts/overfit.py --date 2022-02-15 --levels C M H
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import hydra
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from croscim.data import CascadeDataModule

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--date", required=True)
    ap.add_argument("--levels", nargs="+", default=["C"])
    ap.add_argument("--no-pyramid", action="store_true")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--n-members", type=int, default=4)
    ap.add_argument("--sampler-steps", type=int, default=None, help="Heun steps of the scores (default: config)")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = CascadeDataModule(args.root, os.path.join(REPO, "assets", "ne_10m_land.zip"), *[[args.date, args.date]] * 3)
    dm.setup()
    # the --batch-size validation tiles of the day with the most targets
    ds, scored = dm.ds["val"], []
    for i in range(len(ds)):
        raw = {k: v.to(dev) for k, v in torch.utils.data.default_collate([ds[i]]).items()}
        b = dm.on_after_batch_transfer(raw, 0)
        scored.append((sum(int(b[k].sum()) for k in ("tgt_C_m", "tgt_S_m", "tgt_dM_m")) + int(b["tgt_dH_m"].sum()) // 256, i))
    items = [ds[i] for _, i in sorted(scored, reverse=True)[:args.batch_size]]
    raw = {k: v.to(dev) for k, v in torch.utils.data.default_collate(items).items()}
    batch = dm.on_after_batch_transfer(raw, 0)

    cfg = OmegaConf.load(os.path.join(REPO, "config", "xp", "N0.yaml"))
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    mcfg.update(levels=args.levels, pyramid=not args.no_pyramid, ema_decay=0.0, n_members=args.n_members)
    if args.sampler_steps:
        mcfg["n_steps"] = args.sampler_steps
    model = hydra.utils.instantiate(mcfg).to(dev)
    model.train()
    opt = torch.optim.Adam([p for p in model.nets.parameters() if p.requires_grad], lr=args.lr)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)

    with torch.no_grad(), amp:
        model.ema.update_parameters(model.nets)
        before = {k: float(v) for k, v in model.scores(batch).items()}
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hist, t0 = [], time.perf_counter()
    for step in range(1, args.steps + 1):
        with amp:
            out = model.losses(batch, drop=False)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.nets.parameters(), 0.5)
        opt.step()
        model.ema.update_parameters(model.nets)  # ema_decay = 0: EMA = student
        hist.append({k: float(v) for k, v in out.items()})
        if step % 25 == 0 or step == 1:
            last = hist[-25:]
            print(f"step {step}: " + ", ".join(f"{k} {sum(h[k] for h in last) / len(last):.3f}" for k in hist[-1]),
                  flush=True)
    dt = (time.perf_counter() - t0) / args.steps
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
    with torch.no_grad(), amp:
        after = {k: float(v) for k, v in model.scores(batch).items()}

    first = {k: sum(h[k] for h in hist[:25]) / len(hist[:25]) for k in hist[0]}
    final = {k: sum(h[k] for h in hist[-25:]) / len(hist[-25:]) for k in hist[-1]}
    report = {"levels": args.levels, "pyramid": not args.no_pyramid, "batch_size": len(items),
              "time_per_step_s": dt, "peak_gpu_mem_GiB": mem,
              "loss_first25": first, "loss_last25": final, "scores_before": before, "scores_after": after}
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
