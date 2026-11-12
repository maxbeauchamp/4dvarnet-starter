#!/usr/bin/env python3
"""Day-to-day coherence of the N0 analyses used as dense N1 targets (N1m).

On the C tile with the most ASIP at each start date, the N0 (EMA) analysis
A_t of D + 1 consecutive days, drawn in three ways:
  indep   a new noise every day (the N1m targets);
  mean2   the mean of two such samples (the N1m targets, dense_members 2);
  paired  the same initial noise every day (A_t, A_t+1 only differ by what
          the observations change);
with 8 and 20 Heun steps. Printed: RMS and mean of A_t+1 - A_t over the
ocean (SIC, SIT, HS), the mean change of SIC over open water (SIC < 0.15 at
t: spurious ice) and of SIC over the pack (> 0.9), against the observed
day-to-day change of ASIP (pixels with ASIP on both days).

Usage:
    sbatch scripts/submit_odyssey.sbatch scripts/target_coherence.py --ckpt <N1 .ckpt> \\
        --start 2022-03-05 2022-11-05 2022-12-05 --days 5 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO)
sys.path.append(os.path.join(REPO, "scripts"))

from croscim import data as D  # noqa: E402
from croscim import grid as g  # noqa: E402
from croscim.ocean import ocean_fraction_h  # noqa: E402
from croscim.transition import load_n1  # noqa: E402
from plot_panels import best_tile  # noqa: E402

VARS = ("SIC", "SIT", "HS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="N1 checkpoint (its frozen N0 is used)")
    ap.add_argument("--n0-ckpt", default=None)
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", nargs="+", required=True)
    ap.add_argument("--days", type=int, default=5, help="D: consecutive days after each start")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, nargs="+", default=[8, 20])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n0 = load_n1(args.ckpt, args.n0_ckpt).n0.to(dev).eval()
    omega_h = ocean_fraction_h(os.path.join(REPO, "assets", "ne_10m_land.zip"))
    omega_t = torch.from_numpy(omega_h)
    omega_c = g.restrict(omega_t, torch.ones_like(omega_t), g.factor("H", "C"))[0].numpy()

    def sample(b, ctx, keep, n_steps, seed):
        gen = torch.Generator(dev).manual_seed(seed)
        return n0.sample_c(n0.ema.module, ctx, b, keep, n_steps, gen)[0].float()   # [12, 64, 64] physical

    acc = {}       # (mode, steps) -> list of per-pair stats
    asip = []
    for start in args.start:
        t0 = D.to_day(start)
        iy0, ix0 = best_tile(args.root, t0, omega_c)
        ds = D.CascadeDataset(args.root, [t0], omega_h, np.array([[iy0, ix0]]), train=False, horizon=args.days)
        raw = {k: v.to(dev) for k, v in torch.utils.data.default_collate([ds[0]]).items()}
        days = D.build_batch(raw)
        ocean = days[0]["omega_c"][0] > 0
        for k in range(1, args.days + 1):  # observed: ASIP on both days
            a, b = days[k - 1], days[k]
            both = (a["tgt_C_m"][0, 0] & (a["tgt_C_sigma"][0, 0] == 0) & b["tgt_C_m"][0, 0]
                    & (b["tgt_C_sigma"][0, 0] == 0) & ocean)
            if both.any():
                d = (b["tgt_C"][0, 0] - a["tgt_C"][0, 0])[both]
                asip.append((float((d ** 2).mean().sqrt()), float(d.mean()), int(both.sum())))
        with torch.no_grad():
            ctxs = [n0.encode(n0.ema.module, b) for b in days]
            keep = n0.keep_groups(1, dev, drop=False)
            for n_steps in args.steps:
                for s in range(args.seeds):
                    seqs = {
                        "indep": [sample(b, c, keep, n_steps, 1000 * k + 7 * s) for k, (b, c) in enumerate(zip(days, ctxs))],
                        "mean2": [0.5 * (sample(b, c, keep, n_steps, 1000 * k + 7 * s)
                                         + sample(b, c, keep, n_steps, 1000 * k + 7 * s + 3))
                                  for k, (b, c) in enumerate(zip(days, ctxs))],
                        "paired": [sample(b, c, keep, n_steps, 7 * s) for b, c in zip(days, ctxs)],
                    }
                    for mode, seq in seqs.items():
                        for k in range(1, len(seq)):
                            d = seq[k] - seq[k - 1]
                            sic0 = seq[k - 1][0]
                            row = {}
                            for c, n in enumerate(VARS):
                                row[f"rms_d{n}"] = float((d[c][ocean] ** 2).mean().sqrt())
                                row[f"mean_d{n}"] = float(d[c][ocean].mean())
                            ow, pack = ocean & (sic0 < 0.15), ocean & (sic0 > 0.9)
                            row["mean_dSIC_ow"] = float(d[0][ow].mean()) if ow.any() else np.nan
                            row["mean_dSIC_pack"] = float(d[0][pack].mean()) if pack.any() else np.nan
                            acc.setdefault(f"{mode}_{n_steps}", []).append(row)
        print(f"{start}: tile ({iy0}, {ix0}) done", flush=True)

    keys = ["rms_dSIC", "rms_dSIT", "rms_dHS", "mean_dSIC", "mean_dSIT", "mean_dHS", "mean_dSIC_ow", "mean_dSIC_pack"]
    res = {m: {k: float(np.nanmean([r[k] for r in rows])) for k in keys} for m, rows in acc.items()}
    if asip:
        res["ASIP_observed"] = {"rms_dSIC": float(np.mean([a[0] for a in asip])),
                                "mean_dSIC": float(np.mean([a[1] for a in asip])), "n_pairs": len(asip)}
    print("\nday-to-day change A_t+1 - A_t over the ocean (mean over starts, days, seeds)")
    print(f"{'':12s}" + "".join(f"{k:>15s}" for k in keys))
    for m, r in res.items():
        print(f"{m:12s}" + "".join(f"{r.get(k, np.nan):15.4f}" for k in keys))
    with open(os.path.join(args.out, "target_coherence.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
