#!/usr/bin/env python3
"""Forecast panels and lead-time metrics of a trained N1 (mode A).

For N start dates t evenly spread over [start, end] (t + K must be in the
stores), on the C tile with the most ASIP observations at t: an ensemble of
--members rollouts of --lead days from the N0 analysis at t (each member
starts from its own analysis sample), without observations after t
(--w 0: forecast) or with them (--w 1: assimilation). References: the
persistence of the analysis at t, and the N0 analysis of each day t+k (which
uses the observations of that day).

One figure per start date: one row per displayed lead (SIC mean / spread,
SIT mean, h_pack mean, CIMR SIC of the day (not used by the forecast), error
of the mean), and a last row with the fine levels generated on demand at the
last lead on a 256 km zoom (G_M^0, G_H^0 with h = null, mode A) next to the
ASIP SIC of that day. Plus metrics.json (RMSE / fair CRPS / spread on Y^tgt
per lead, persistence and N0-analysis RMSE) and lead_curves.png.

Usage:
    sbatch scripts/submit_odyssey.sbatch scripts/plot_forecast.py --ckpt <N1 last.ckpt> \\
        --start 2022-02-15 --end 2022-02-24 --lead 2 --n 3 --members 8 \\
        --out /Odyssey/private/$USER/results/forecast
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from croscim import data as D  # noqa: E402
from croscim import grid as g  # noqa: E402
from croscim import rollout as R  # noqa: E402
from croscim.cascade import CROP_C, X_NAMES, Z_NAMES, crop  # noqa: E402
from croscim.ocean import ocean_fraction_h  # noqa: E402
from croscim.transition import load_n1  # noqa: E402
from plot_panels import best_crops, best_tile, extent, masked, panel  # noqa: E402

FM, FH = g.factor("M", "C"), g.factor("H", "M")
I_PACK = Z_NAMES.index("h_pack")


def day_obs(b):
    """CIMR SIC of the day at C, all of it (Y^tgt where held out, Y^in elsewhere)."""
    tgt, m_tgt = b["tgt_C"][0, 0], b["tgt_C_m"][0, 0]
    inp, m_in = b["in_C_cimr"][0, -1, 0], b["in_C_cimr_w"][0, -1, 0] > 0
    return torch.where(m_tgt, tgt, inp).float().cpu(), (m_tgt | m_in).cpu()


@torch.no_grad()
def downscale_zoom(n0, b, z_c, psi, oy, ox, hy, hx, n_steps, seed):
    """Mode A fine levels on demand on the zooms: G_M^0 then G_H^0, h = null."""
    dev = z_c.device
    nets = n0.ema.module
    keep = n0.keep_groups(1, dev, drop=False)
    keep["h"] = torch.zeros_like(keep["h"])
    ctx = n0.encode(nets, b)
    t = lambda v: torch.tensor([v], device=dev)  # noqa: E731
    gen = torch.Generator(dev).manual_seed(seed)
    d_m = n0.sample_m(nets, ctx, b, z_c, psi, t(oy), t(ox), keep, n_steps, generator=gen)
    x_m = n0.x_m(z_c, d_m, b, t(oy), t(ox))
    x_m_h = crop(n0.coarse_h(x_m, b, t(oy), t(ox)), t(hy - oy), t(hx - ox), CROP_C["H"], FM * FH)  # P' X_M at H
    d_h = n0.sample_h(nets, ctx, b, z_c, x_m_h, psi, t(hy), t(hx), keep, n_steps, generator=gen)
    return (x_m_h[:, :1] + d_h)[0, 0].float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="N1 checkpoint (Lightning .ckpt)")
    ap.add_argument("--n0-ckpt", default=None, help="N0 checkpoint, if it moved since N1 was trained")
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--lead", type=int, default=7, help="forecast length K (days)")
    ap.add_argument("--n", type=int, default=3, help="number of start dates")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--w", type=float, default=0.0, help="weight of the observations after t (0: forecast)")
    ap.add_argument("--n-steps", type=int, default=None, help="Heun steps (default: the model's)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)

    n1 = load_n1(args.ckpt, args.n0_ckpt).to(dev).eval()
    n0, K = n1.n0, args.lead
    n_steps = args.n_steps or n1.hparams.n_steps
    omega_h = ocean_fraction_h(os.path.join(REPO, "assets", "ne_10m_land.zip"))
    omega_t = torch.from_numpy(omega_h)
    omega_c = g.restrict(omega_t, torch.ones_like(omega_t), g.factor("H", "C"))[0].numpy()
    dm = D.CascadeDataModule(args.root, None, *[[args.start, args.end]] * 3, horizon=K)
    starts = dm._days("val")
    if not starts:
        raise SystemExit(f"no start date t in [{args.start}, {args.end}] with t..t+{K} in the stores")
    starts = [starts[int(i)] for i in np.linspace(0, len(starts) - 1, min(args.n, len(starts)))]
    leads = sorted({int(round(v)) for v in np.linspace(0, K, min(4, K + 1))})

    metrics = {}
    for t0_day in starts:
        tic = time.perf_counter()
        date = str(D.EPOCH + dt.timedelta(t0_day))
        iy0, ix0 = best_tile(args.root, t0_day, omega_c)
        ds = D.CascadeDataset(args.root, [t0_day], omega_h, np.array([[iy0, ix0]]), train=False, horizon=K)
        raw = {k: v.to(dev) for k, v in torch.utils.data.default_collate([ds[0]]).items()}
        days = D.build_batch(raw)
        with torch.no_grad(), amp:
            runs = [R.rollout(n1, days, K, w=args.w, seed=s, sampler_steps=n_steps) for s in range(args.members)]
            traj = torch.stack([z[0] for z, _ in runs]).float()          # [N, K+1, 12, 64, 64]
            psis = torch.stack([p[0] for _, p in runs]).float()
            ana = torch.stack([torch.stack([n1.analysis(days[k], n_steps, torch.Generator(dev).manual_seed(100 + s))[0][0]
                                            for k in range(K + 1)]) for s in range(args.members)]).float()
            oy, ox, hy, hx = best_crops(days[K])
            fine = torch.stack([downscale_zoom(n0, days[K], traj[s:s + 1, K].to(dev), psis[s:s + 1, K].to(dev),
                                               oy, ox, hy, hx, n_steps, s) for s in range(min(3, args.members))])
        traj, ana = traj.cpu(), ana.cpu()

        # ---- metrics per lead on Y^tgt of day t+k ----
        m_date = {"tile": [iy0, ix0]}
        for k in range(K + 1):
            b = days[k]
            tgt = torch.cat([b["tgt_C"], b["tgt_S"]], 1).float().cpu()
            msk = torch.cat([b["tgt_C_m"], b["tgt_S_m"]], 1).cpu()
            fc = n0._accumulate(f"lead{k}", traj[None, :, k], tgt, msk, Z_NAMES)
            an = n0._accumulate(f"ana{k}", ana[None, :, k], tgt, msk, Z_NAMES)
            row = {f"fc_{n}": round(float(v), 5) for n, v in fc.items()}
            row.update({f"ana_{n}": round(float(v), 5) for n, v in an.items() if n.startswith("rmse")})
            for c, name in enumerate(X_NAMES):
                if msk[0, c].any():
                    pers = traj[:, 0, c].mean(0)
                    row[f"pers_rmse_{name}"] = round(float(((pers - tgt[0, c])[msk[0, c]] ** 2).mean().sqrt()), 5)
            m_date[f"lead{k}"] = row
        metrics[date] = m_date

        # ---- figure ----
        e_c = extent(iy0, ix0, D.N_C)
        e_h = extent(iy0, ix0, CROP_C["H"], hy, hx)
        land_c = days[0]["omega_c"][0].cpu().numpy() > 0
        land_h = crop(days[K]["omega_h"][:1].cpu(), torch.tensor([hy]), torch.tensor([hx]), CROP_C["H"], FM * FH)[0].numpy() > 0
        sic, spread = dict(cmap="Blues_r", vmin=0, vmax=1), dict(cmap="magma")
        rows = []
        for k in leads:
            mean, std = traj[:, k].mean(0).numpy(), traj[:, k].std(0).numpy()
            obs, m_obs = day_obs(days[k])
            box = e_h if k == K else None
            rows.append((f"t+{k} d", [
                ("SIC ens. mean", masked(mean[0], land_c), e_c, sic, box, ~land_c),
                ("SIC ens. spread", masked(std[0], land_c), e_c, spread, None, ~land_c),
                ("SIT ens. mean (m)", masked(mean[1], land_c), e_c, dict(cmap="viridis"), None, ~land_c),
                ("pack h_pack ens. mean", masked(mean[I_PACK], land_c), e_c, dict(cmap="cividis", vmin=0, vmax=1), None, ~land_c),
                ("CIMR SIC of the day (not used)" if args.w == 0 or k == 0 else "CIMR SIC of the day",
                 masked(obs.numpy(), m_obs.numpy()), e_c, sic, None, ~land_c),
                ("SIC error (mean - CIMR)", masked(mean[0] - obs.numpy(), m_obs.numpy() & land_c), e_c,
                 dict(cmap="RdBu_r", vmin=-0.5, vmax=0.5), None, ~land_c)]))
        hb = days[K]
        o = lambda v: torch.tensor([v])  # noqa: E731
        asip = crop(hb["in_H"][:, -1:].cpu(), o(hy), o(hx), CROP_C["H"], FM * FH)[0, 0]
        asip_w = crop(hb["in_H_w"][:, -1:].cpu(), o(hy), o(hx), CROP_C["H"], FM * FH)[0, 0] > 0
        tg = crop(hb["tgt_H"].cpu(), o(hy), o(hx), CROP_C["H"], FM * FH)[0, 0]
        tg_m = crop(hb["tgt_H_m"].cpu(), o(hy), o(hx), CROP_C["H"], FM * FH)[0, 0]
        asip_all = torch.where(tg_m, tg, asip).float().numpy()
        rows.append((f"H 1 km, t+{K} d (mode A)", [
            *[(f"SIC member {s + 1}", masked(fine[s].numpy(), land_h), e_h, sic, None, ~land_h) for s in range(len(fine))],
            ("SIC mean", masked(fine.mean(0).numpy(), land_h), e_h, sic, None, ~land_h),
            ("SIC spread", masked(fine.std(0).numpy(), land_h), e_h, spread, None, ~land_h),
            ("ASIP SIC of the day (not used)", masked(asip_all, (asip_w | tg_m).numpy()), e_h, sic, None, ~land_h)]))
        panel(os.path.join(args.out, f"forecast_{date}.png"),
              f"start {date}  tile ({iy0}, {ix0})  {args.members} members, w = {args.w}, {n_steps} Heun steps", rows)
        metrics[date]["time_s"] = round(time.perf_counter() - tic, 1)
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"{date}: forecast panel and metrics in {metrics[date]['time_s']} s", flush=True)

    curves(metrics, K, os.path.join(args.out, "lead_curves.png"))


def curves(metrics, K, path):
    """RMSE / spread vs lead for SIC, SIT, snow: forecast, persistence, N0 analysis (mean over start dates)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean(key, k):
        v = [d[f"lead{k}"].get(key, np.nan) for d in metrics.values()]
        return np.nanmean(v) if np.isfinite(v).any() else np.nan

    fig, axes = plt.subplots(1, len(X_NAMES), figsize=(4 * len(X_NAMES), 3.2))
    lead = np.arange(K + 1)
    for ax, name in zip(axes, X_NAMES):
        for key, label, style in ((f"fc_rmse_{name}", "forecast RMSE", "-o"), (f"fc_spread_{name}", "forecast spread", "--o"),
                                  (f"pers_rmse_{name}", "persistence RMSE", "-s"), (f"ana_rmse_{name}", "N0 analysis RMSE", ":^")):
            ax.plot(lead, [mean(key, k) for k in lead], style, label=label, ms=3)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("lead (days)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"scores on Y^tgt, mean over {len(metrics)} start dates", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
