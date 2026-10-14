#!/usr/bin/env python3
"""End-to-end smoke test of the whole pipeline on a short period (one job).

Chains, with reduced settings (2 short epochs per phase, few sampler steps),
the real configs, Lightning loops and checkpoints:

    phase 3  N0, G_C alone, no pyramid
    phase 4  N0, G_M + G_H, initialised from phase 3
    phase 5  N0, full cascade with pyramid, initialised from phase 4
    phase 6  N1 (horizon 2) from the phase-5 checkpoint
    phase 7  rollouts A (w = 1, 0), mode A vs B, I^up / I^down, response,
             controllability, on one validation sequence

and writes --out/report.json (time, peak GPU memory, val / test metrics,
checkpoint per phase), updated after every phase. A test of the plumbing,
not of the scientific performance.

Usage (one month converted, e.g. 2022-02):
    sbatch scripts/submit_odyssey.sbatch scripts/smoke_pipeline.py \\
        --out /Odyssey/private/$USER/results/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO)
from croscim import rollout as R  # noqa: E402

TINY = ["base_dim=8", "encoder_depths=[1,1,1,1]", "decoder_depths=[1,1,1]", "num_heads=[1,1,1,1]",
        "time_dim=16", "modulation_rank=4"]


def load_cfg(xp: str, overrides: list[str]):
    with initialize_config_dir(config_dir=os.path.join(REPO, "config"), version_base="1.3"):
        return compose("main", overrides=[f"xp={xp}", *overrides])


def floats(metrics: dict) -> dict:
    return {k: round(float(v), 5) for k, v in metrics.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", nargs=2, default=["2022-02-15", "2022-02-23"])
    ap.add_argument("--val", nargs=2, default=["2022-02-24", "2022-02-26"])
    ap.add_argument("--train-batches", type=int, default=30)
    ap.add_argument("--tiny", action="store_true", help="tiny networks on CPU (local test of this script)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cuda = torch.cuda.is_available() and not args.tiny
    report, path = {}, os.path.join(args.out, "report.json")

    def save():
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    common = [f"datamodule.root={args.root}", f"datamodule.natural_earth_zip={REPO}/assets/ne_10m_land.zip",
              f"datamodule.train_dates=[{args.train[0]},{args.train[1]}]",
              f"datamodule.val_dates=[{args.val[0]},{args.val[1]}]",
              f"datamodule.test_dates=[{args.val[0]},{args.val[1]}]",
              "model.n_steps=4", "model.n_members=2", "model.score_every=1", "model.n_score_batches=1"]
    n0_only = ["model.cond_steps=2"]  # N1 takes its architecture from the N0 checkpoint
    if args.tiny:
        n0_only += [f"swin_config.{t}" for t in TINY] + [f"swin_config_fine.{t}" for t in TINY]
        common += ["datamodule.dl_kw.num_workers=0", "datamodule.dl_kw.pin_memory=false"]

    def trainer(cfg, name):
        kw = dict(max_epochs=2, limit_train_batches=args.train_batches, limit_val_batches=2, limit_test_batches=2,
                  logger=pl.loggers.CSVLogger(args.out, name=name), enable_progress_bar=False,
                  num_sanity_val_steps=0)
        if not cuda:
            kw.update(accelerator="cpu", devices=1, precision="32")
        return hydra.utils.instantiate(cfg.trainer, **kw)

    def run(name, cfg, model, dm, init=None):
        if init is not None:
            sd = torch.load(init, map_location="cpu", weights_only=False)["state_dict"]
            missing, _ = model.load_state_dict(sd, strict=False)
            print(f"{name}: initialised from {init} ({len(missing)} tensors left at random init)", flush=True)
        tr = trainer(cfg, name)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        tr.fit(model, datamodule=dm)
        t_fit = time.perf_counter() - t0
        val = floats(tr.callback_metrics)
        tr.test(model, datamodule=dm)
        report[name] = {"fit_s": round(t_fit, 1), "peak_gpu_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                        if cuda else None, "val": val, "test": floats(tr.callback_metrics),
                        "ckpt": tr.checkpoint_callback.last_model_path}
        save()
        print(f"{name}: done in {t_fit:.0f} s", flush=True)
        return report[name]["ckpt"]

    # ---- N0: phases 3, 4, 5 ----
    phases = [("phase3_C", ["model.levels=[C]", "model.pyramid=false"]),
              ("phase4_MH", ["model.levels=[M,H]", "model.pyramid=false"]),
              ("phase5_full", ["model.levels=[C,M,H]", "model.pyramid=true"])]
    dm0, ckpt = None, None
    for name, extra in phases:
        cfg = load_cfg("N0", common + n0_only + extra)
        dm0 = dm0 or hydra.utils.instantiate(cfg.datamodule)
        ckpt = run(name, cfg, hydra.utils.instantiate(cfg.model), dm0, init=ckpt)

    # ---- N1: phase 6 ----
    cfg = load_cfg("N1", common + [f"model.n0_ckpt={ckpt}", "model.k_max=2", "model.curriculum_epochs=1",
                                   "model.push_steps=2", "model.n_score_batches=1"])
    dm1 = hydra.utils.instantiate(cfg.datamodule)
    n1 = hydra.utils.instantiate(cfg.model)
    run("phase6_N1", cfg, n1, dm1)

    # ---- phase 7: scenarios and diagnostics on one validation sequence ----
    dev = torch.device("cuda" if cuda else "cpu")
    n1 = n1.to(dev).eval()
    batch = next(iter(dm1.val_dataloader()))
    days = dm1.on_after_batch_transfer({k: v.to(dev) for k, v in batch.items()}, 0)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else torch.autocast("cpu", enabled=False)
    out, t0 = {}, time.perf_counter()
    with torch.no_grad(), amp:
        za, pa = R.rollout(n1, days, 2, w=1.0, seed=0)
        zf, _ = R.rollout(n1, days, 2, w=0.0, seed=0)
        out["rollout_A_assim_vs_free_rmse_X"] = float(((za - zf)[:, 1:, :3] ** 2).mean().sqrt())
        out["mode_A_vs_B"] = [{k: (float(v) if torch.is_tensor(v) else v) for k, v in d.items()}
                              for d in R.compare_modes(n1, days, 1)]
        out["loss_gaps"] = floats(R.loss_gaps(n1, days))
        resp = R.response(n1, days, torch.tensor(0.05, device=dev), ["edge_density"], 2, n_members=2)
        out["response_mean_abs_X_k2"] = float(resp[:, 2].abs().mean())
        out["controllability_sic_std"] = float(R.controllability(n1.n0, days[0], za[:, 0], pa[:, 0], "sic_std", 0.02))
    out["time_s"] = round(time.perf_counter() - t0, 1)
    report["phase7_diagnostics"] = out
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
