#!/usr/bin/env python3
"""Stability of an N1 rollout, channel by channel: for each lead, the RMS
over the ocean of the normalised state (X, S_C) and of psi, and the spread
between members. A channel whose amplitude jumps after the first step (e.g.
psi, unbounded, or SIT) shows where the rollout leaves the distribution the
network was trained on. Run with --w 0 (forecast) and --w 1 (assimilation),
and --fp32 to rule out the bf16 autocast.

Usage:
    sbatch scripts/submit_odyssey.sbatch scripts/rollout_stats.py --ckpt <N1 .ckpt> \\
        --start 2022-12-24 --lead 7 --members 4 --out <dir>
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
from croscim import rollout as R  # noqa: E402
from croscim.cascade import N_Z, Z_NAMES  # noqa: E402
from croscim.ocean import ocean_fraction_h  # noqa: E402
from croscim.transition import load_n1  # noqa: E402
from plot_panels import best_tile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="N1 checkpoint")
    ap.add_argument("--n0-ckpt", default=None)
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", required=True, help="start day t")
    ap.add_argument("--lead", type=int, default=7)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--w", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--fp32", action="store_true", help="no bf16 autocast")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" and not args.fp32
           else torch.autocast("cpu", enabled=False))
    n1 = load_n1(args.ckpt, args.n0_ckpt).to(dev).eval()
    K, n_steps = args.lead, args.n_steps or n1.hparams.n_steps
    omega_h = ocean_fraction_h(os.path.join(REPO, "assets", "ne_10m_land.zip"))
    omega_t = torch.from_numpy(omega_h)
    omega_c = g.restrict(omega_t, torch.ones_like(omega_t), g.factor("H", "C"))[0].numpy()
    t0 = D.to_day(args.start)
    iy0, ix0 = best_tile(args.root, t0, omega_c)
    ds = D.CascadeDataset(args.root, [t0], omega_h, np.array([[iy0, ix0]]), train=False, horizon=K)
    raw = {k: v.to(dev) for k, v in torch.utils.data.default_collate([ds[0]]).items()}
    days = D.build_batch(raw)
    ocean = days[0]["omega_c"][0] > 0
    names = list(Z_NAMES) + [f"psi{i}" for i in range(n1.d_psi)]
    out = {"start": args.start, "tile": [int(iy0), int(ix0)], "fp32": args.fp32, "increment": n1.hparams.get("increment")}
    for w in args.w:
        with torch.no_grad(), amp:
            runs = [R.rollout(n1, days, K, w=w, seed=s, sampler_steps=n_steps) for s in range(args.members)]
        z = torch.stack([r[0][0] for r in runs]).float()                     # [N, K+1, 12, 64, 64]
        psi = torch.stack([r[1][0] for r in runs]).float()                   # [N, K+1, d_psi, 64, 64]
        zn = torch.cat([n1.nrm.z(z.flatten(0, 1)).view_as(z), psi], 2)        # normalised state + psi
        rms = (zn[..., ocean] ** 2).mean((0, 3)).sqrt()                       # [K+1, C]
        spread = zn.std(0)[..., ocean].mean(-1)                               # [K+1, C]
        res = {}
        print(f"\nw = {w}: RMS of the normalised state over the ocean (members pooled) / member spread")
        print("lead " + " ".join(f"{n:>12}" for n in names))
        for k in range(K + 1):
            print(f"{k:4d} " + " ".join(f"{float(rms[k, c]):6.2f}/{float(spread[k, c]):<5.2f}" for c in range(len(names))))
            res[f"lead{k}"] = {n: [round(float(rms[k, c]), 4), round(float(spread[k, c]), 4)] for c, n in enumerate(names)}
        out[f"w{w:g}"] = res
    with open(os.path.join(args.out, f"rollout_stats_{args.start}{'_fp32' if args.fp32 else ''}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
