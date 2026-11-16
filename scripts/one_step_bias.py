#!/usr/bin/env python3
"""One N1 step from the N0 analysis: where does the SIT / open-water SIC jump of
the first forecast day come from (N1n)?

On the C tile with the most ASIP at each start date t: A_t, A_t+1 = N0 (EMA)
analyses of t and t + 1 drawn with the same noise (the N1n paired targets).
From A_t, M samples of one N1 step (EMA nets, sampler of the checkpoint), in
three variants:
  bounded   the rollout step: censored sampler, then Pi_S (project_s);
  raw       the sampler without bounds: the law the network learnt;
  raw_clip  raw, then Pi_S (clip to the physical bounds after sampling).
Printed per class of A_t (SIC: ow < 0.15 < edge < 0.9 < pack; SIT among ice
(SIC > 0.15): thin < 0.3 < mid < 1.5 < thick m), over the ocean: the mean of
(ensemble mean - A_t+1) for SIC and SIT, the ensemble spread, the share of raw
values below 0, and the target change A_t+1 - A_t.

Reading: raw unbiased, raw_clip / bounded biased up -> rectification at the
bound (clipping a too wide law); raw already biased -> the learnt conditional
mean is wrong (loss / floor / scale).

Usage:
    sbatch scripts/submit_odyssey.sbatch scripts/one_step_bias.py --ckpt <N1 .ckpt> \\
        --start 2022-03-05 2022-10-05 2022-11-05 2022-12-05 --out <dir>
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

from croscim.cascade import N_Z, Z_NAMES  # noqa: E402
from croscim.rollout import project_s  # noqa: E402

I_SIC, I_SIT = Z_NAMES.index("SIC"), Z_NAMES.index("SIT")
VARIANTS = ("bounded", "raw", "raw_clip")


@torch.no_grad()
def one_step(n1, b: dict, z, psi, keep: dict, n_steps: int, w: float, generator, bounded: bool):
    """N1.step with the sampler bounds optional; physical z out (no Pi_S)."""
    nets = n1.ema.module
    base = n1.base(b, z, psi)
    cond, valid = n1.cond(nets, b, z, psi, keep, base)
    cond_null = n1.cond(nets, b, z, psi, {**keep, "h": torch.zeros_like(keep["h"])}, base)[0] if w != 1.0 else None
    lo, hi = n1.lo, n1.hi
    sc = n1.inc_sc()
    if n1.hparams.increment:
        lo, hi = (lo - base) / sc, (hi - base) / sc
    if not bounded:
        lo = hi = None
    zn = nets.gen.sample(cond, valid, n_steps, lo=lo, hi=hi, cond_null=cond_null, w=w, generator=generator)
    if n1.hparams.increment:
        zn = base + zn * sc
    return n1.nrm.z_inv(zn[:, :N_Z])


def classes(a0, ocean) -> dict:
    sic, sit = a0[0, I_SIC], a0[0, I_SIT]
    ice = ocean & (sic > 0.15)
    return {"all": ocean, "ow": ocean & (sic < 0.15), "edge": ice & (sic <= 0.9), "pack": ocean & (sic > 0.9),
            "thin": ice & (sit < 0.3), "mid": ice & (sit >= 0.3) & (sit < 1.5), "thick": ice & (sit >= 1.5)}


@torch.no_grad()
def pair_stats(n1, b0: dict, b1: dict, seed: int, members: int, w: float, ana_steps: int, n_steps: int) -> dict:
    """Per variant and class of A_t: bias of the ensemble mean vs A_t+1, spread,
    share of raw values < 0; plus the target change. One start, one noise."""
    dev = b0["omega_c"].device
    a0, psi = n1.analysis(b0, ana_steps, torch.Generator(dev).manual_seed(seed))
    a1, _ = n1.analysis(b1, ana_steps, torch.Generator(dev).manual_seed(seed))
    a0, a1 = project_s(a0.float()), project_s(a1.float())
    keep = n1.keep(1, dev, train=False)
    ens = {v: [] for v in VARIANTS}
    for s in range(members):
        g = torch.Generator(dev).manual_seed(10_007 * seed + s)
        ens["bounded"].append(project_s(one_step(n1, b1, a0, psi, keep, n_steps, w, g, True).float()))
        g = torch.Generator(dev).manual_seed(10_007 * seed + s)  # same noise as bounded
        raw = one_step(n1, b1, a0, psi, keep, n_steps, w, g, False).float()
        ens["raw"].append(raw)
        ens["raw_clip"].append(project_s(raw))
    ocean = b0["omega_c"][0] > 0
    out = {}
    for name, m in classes(a0, ocean).items():
        if not m.any():
            continue
        row = {"n": int(m.sum()),
               "dSIC_tgt": float((a1 - a0)[0, I_SIC][m].mean()), "dSIT_tgt": float((a1 - a0)[0, I_SIT][m].mean())}
        for v, e in ens.items():
            e = torch.cat(e)                                             # [M, N_Z, H, W]
            for c, var in ((I_SIC, "SIC"), (I_SIT, "SIT")):
                row[f"{v}_bias_{var}"] = float((e[:, c].mean(0) - a1[0, c])[m].mean())
                row[f"{v}_spread_{var}"] = float(e[:, c].std(0)[m].mean()) if members > 1 else float("nan")
                if v == "raw":
                    row[f"raw_below0_{var}"] = float((e[:, c] < 0).float().mean(0)[m].mean())
        out[name] = row
    return out


def main():
    from croscim import data as D
    from croscim import grid as g
    from croscim.ocean import ocean_fraction_h
    from croscim.transition import load_n1
    from plot_panels import best_tile

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n0-ckpt", default=None)
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", nargs="+", required=True)
    ap.add_argument("--days", type=int, default=3, help="pairs (t+j, t+j+1), j < days, per start")
    ap.add_argument("--seeds", type=int, default=2, help="analysis noises per pair")
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--w", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--analysis-steps", type=int, default=20)
    ap.add_argument("--n-steps", type=int, default=None, help="N1 sampler steps (default: checkpoint)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n1 = load_n1(args.ckpt, args.n0_ckpt).to(dev).eval()
    n_steps = args.n_steps or n1.hparams.n_steps
    omega_h = ocean_fraction_h(os.path.join(REPO, "assets", "ne_10m_land.zip"))
    omega_t = torch.from_numpy(omega_h)
    omega_c = g.restrict(omega_t, torch.ones_like(omega_t), g.factor("H", "C"))[0].numpy()

    acc = {}   # (w, class) -> list of rows
    for start in args.start:
        t0 = D.to_day(start)
        iy0, ix0 = best_tile(args.root, t0, omega_c)
        ds = D.CascadeDataset(args.root, [t0], omega_h, np.array([[iy0, ix0]]), train=False, horizon=args.days)
        raw = {k: v.to(dev) for k, v in torch.utils.data.default_collate([ds[0]]).items()}
        days = D.build_batch(raw)
        for w in args.w:
            for j in range(args.days):
                for s in range(args.seeds):
                    res = pair_stats(n1, days[j], days[j + 1], 1000 * j + 7 * s + 1, args.members, w,
                                     args.analysis_steps, n_steps)
                    for cls, row in res.items():
                        acc.setdefault(f"w{w:g}", {}).setdefault(cls, []).append(row)
        print(f"{start}: tile ({iy0}, {ix0}) done", flush=True)

    res = {w: {cls: {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]} for cls, rows in d.items()}
           for w, d in acc.items()}
    for w, d in res.items():
        print(f"\n== {w}: one step from A_t, vs A_t+1 (mean over starts, days, seeds); bias of the ensemble mean")
        print(f"{'':6s}{'n':>7s}{'dSIC_tgt':>10s}" + "".join(f"{v + '_SIC':>14s}" for v in VARIANTS)
              + f"{'sprd_SIC':>10s}{'<0_SIC':>8s}{'dSIT_tgt':>10s}" + "".join(f"{v + '_SIT':>14s}" for v in VARIANTS)
              + f"{'sprd_SIT':>10s}{'<0_SIT':>8s}")
        for cls, r in d.items():
            print(f"{cls:6s}{r['n']:7.0f}{r['dSIC_tgt']:+10.4f}" + "".join(f"{r[f'{v}_bias_SIC']:+14.4f}" for v in VARIANTS)
                  + f"{r['bounded_spread_SIC']:10.4f}{r['raw_below0_SIC']:8.3f}{r['dSIT_tgt']:+10.4f}"
                  + "".join(f"{r[f'{v}_bias_SIT']:+14.4f}" for v in VARIANTS)
                  + f"{r['bounded_spread_SIT']:10.4f}{r['raw_below0_SIT']:8.3f}")
    with open(os.path.join(args.out, "one_step_bias.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
