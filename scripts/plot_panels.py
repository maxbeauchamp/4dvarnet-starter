#!/usr/bin/env python3
"""Inference panels and metrics of a trained N0 (cascade C -> M -> H).

For N dates evenly spread over [start, end], on the C tile (2048 km) with the
most ASIP observations that day, a zoom of 16 x 16 C cells (512 km) for M
and of 8 x 8 C cells (256 km) for H inside it (also where ASIP observes
most): an ensemble of --members members of the EMA cascade, then one figure
per date (4 rows, zoom boxes drawn on the row above):

    C 32 km   ensemble mean / spread of SIC, mean SIT, mean snow, X^_C target, CIMR Y^in
    S_C, psi  ensemble means of the sub-grid parameters, two psi_C components
    M 4 km    two members, mean / spread of SIC, ASIP target at M, mean dSIT_M
    H 1 km    three members, mean / spread of SIC, ASIP target and input

and metrics.json: per date and level, RMSE of the ensemble mean, fair CRPS
and spread on the Y^tgt targets (+ a summary figure).

Usage:
    sbatch scripts/submit_odyssey.sbatch scripts/plot_panels.py --ckpt <N0 last.ckpt> \\
        --start 2022-02-15 --end 2022-02-26 --n 4 --members 8 --out /Odyssey/private/$USER/results/panels
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
from croscim import data as D  # noqa: E402
from croscim import grid as g  # noqa: E402
from croscim.cascade import CROP_C, X_NAMES, Z_NAMES, crop  # noqa: E402
from croscim.ocean import ocean_fraction_h  # noqa: E402
from croscim.transition import load_n0  # noqa: E402

FM, FH = g.factor("M", "C"), g.factor("H", "M")


def best_tile(root, day, omega_c):
    """Tile of the validation lattice with the most ASIP observations (4 km weights) on `day`."""
    import zarr
    store = os.path.join(root, D.STORES["asip"])
    days = zarr.open_array(os.path.join(store, "days"), mode="r")[:]
    w = zarr.open_array(os.path.join(store, "m_w"), mode="r")[int(np.flatnonzero(days == day)[0])] > 0
    cov_c = w.reshape(200, FM, 200, FM).mean((1, 3))
    origins = D.valid_tile_origins(omega_c, D.N_C)
    score = [cov_c[i:i + D.N_C, j:j + D.N_C].sum() for i, j in origins]
    return tuple(int(v) for v in origins[int(np.argmax(score))])


def best_crops(b):
    """M crop (16 C cells) then H crop (8 C cells) inside it with the most ASIP
    targets over the ocean at day t (inputs count for 10 %: there must be
    targets at M / H for the metrics)."""
    ocean = b["omega_h"][0] > 0
    obs = (b["tgt_H_m"][0, 0] & ocean).float() + 0.1 * ((b["in_H_w"][0, -1] > 0) & ocean).float()
    cov = g.block_sum(obs, FM * FH)
    nm, nh = CROP_C["M"], CROP_C["H"]
    cand = [(y, x) for y in range(0, D.N_C - nm + 1, 4) for x in range(0, D.N_C - nm + 1, 4)]
    oy, ox = max(cand, key=lambda c: float(cov[c[0]:c[0] + nm, c[1]:c[1] + nm].sum()))
    sub = [(oy + dy, ox + dx) for dy in range(0, nm - nh + 1, 2) for dx in range(0, nm - nh + 1, 2)]
    hy, hx = max(sub, key=lambda c: float(cov[c[0]:c[0] + nh, c[1]:c[1] + nh].sum()))
    return oy, ox, hy, hx


@torch.no_grad()
def ensemble(model, b, n_members, n_steps, oy, ox, hy, hx):
    nets, dev = model.ema.module, b["omega_c"].device
    keep = model.keep_groups(1, dev, drop=False)
    ctx = model.encode(nets, b)
    t = lambda v: torch.tensor([v], device=dev)  # noqa: E731
    zc, xm, xh, dm = [], [], [], []
    for m in range(n_members):
        gen = torch.Generator(dev).manual_seed(m)
        z_c = model.sample_c(nets, ctx, b, keep, n_steps, generator=gen)
        d_m = model.sample_m(nets, ctx, b, z_c, ctx["psi"], t(oy), t(ox), keep, n_steps, generator=gen)
        x_m = model.x_m(crop(z_c, t(oy), t(ox), CROP_C["M"], 1), d_m)
        x_m_h = crop(x_m, t(hy - oy), t(hx - ox), CROP_C["H"], FM)
        d_h = model.sample_h(nets, ctx, b, z_c, x_m_h, ctx["psi"], t(hy), t(hx), keep, n_steps, generator=gen)
        zc.append(z_c[0]), xm.append(x_m[0]), dm.append(d_m[0]), xh.append((g.prolong(x_m_h[:, :1], FH) + d_h)[0])
    stack = lambda v: torch.stack(v).float().cpu()  # noqa: E731
    return stack(zc), stack(xm), stack(dm), stack(xh), ctx["psi"][0].float().cpu()


def extent(iy0, ix0, n_c, oy=0, ox=0):
    """imshow extent (km) of n_c x n_c C cells starting at C cell (iy0 + oy, ix0 + ox)."""
    y_e, x_e = g.tile_edges("C", iy0 + oy, ix0 + ox, n_c)
    return [x_e[0] / 1e3, x_e[-1] / 1e3, y_e[-1] / 1e3, y_e[0] / 1e3]


def panel(path, title, rows):
    """rows: list of (row label, list of (title, field, extent, kwargs, box or None, land mask))."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ncol = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.3 * ncol, 3.1 * len(rows)), squeeze=False)
    for i, (label, cells) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(cells):
                ax.axis("off")
                continue
            name, field, ext, kw, box, land = cells[j]
            ax.set_facecolor("white")  # no data
            ax.imshow(np.where(land, 1.0, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=2.5,
                      interpolation="nearest")  # land in grey
            im = ax.imshow(field, extent=ext, interpolation="nearest", **kw)
            ax.set_title(name, fontsize=8)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
            if box is not None:
                ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec="red", lw=1))
        axes[i, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def masked(x, m):
    return np.where(m, x, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="N0 checkpoint (Lightning .ckpt)")
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--n", type=int, default=4, help="number of dates")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=None, help="Heun steps (default: the model's)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)

    model = load_n0(args.ckpt).to(dev).eval()
    n_steps = args.n_steps or model.hparams.n_steps
    omega_h = ocean_fraction_h(os.path.join(REPO, "assets", "ne_10m_land.zip"))
    omega_t = torch.from_numpy(omega_h)
    omega_c = g.restrict(omega_t, torch.ones_like(omega_t), g.factor("H", "C"))[0].numpy()
    dm = D.CascadeDataModule(args.root, None, [args.start, args.end], [args.start, args.end], [args.start, args.end])
    days = dm._days("val")
    days = [days[int(i)] for i in np.linspace(0, len(days) - 1, min(args.n, len(days)))]

    metrics = {}
    for day in days:
        t0 = time.perf_counter()
        date = str(D.EPOCH + dt.timedelta(day))
        iy0, ix0 = best_tile(args.root, day, omega_c)
        ds = D.CascadeDataset(args.root, [day], omega_h, np.array([[iy0, ix0]]), train=False)
        raw = {k: v.to(dev) for k, v in torch.utils.data.default_collate([ds[0]]).items()}
        b = D.build_batch(raw)[0]
        oy, ox, hy, hx = best_crops(b)
        with amp:
            zc, xm, dmm, xh, psi = ensemble(model, b, args.members, n_steps, oy, ox, hy, hx)

        # ---- metrics on Y^tgt ----
        cpu = {k: v[0].float().cpu() if v.is_floating_point() else v[0].cpu() for k, v in b.items()
               if torch.is_tensor(v) and v.ndim > 1}
        o = lambda v: torch.tensor([v])  # noqa: E731
        sc = {}
        for key, ens, tgt, m, names in (
                ("C", zc, torch.cat([cpu["tgt_C"], cpu["tgt_S"]]), torch.cat([cpu["tgt_C_m"], cpu["tgt_S_m"]]), Z_NAMES),
                ("M", xm, crop(cpu["tgt_M"][None], o(oy), o(ox), CROP_C["M"], FM)[0],
                 crop(cpu["tgt_M_m"][None], o(oy), o(ox), CROP_C["M"], FM)[0], ("SIC", "SIT")),
                ("H", xh, crop(cpu["tgt_H"][None], o(hy), o(hx), CROP_C["H"], FM * FH)[0],
                 crop(cpu["tgt_H_m"][None], o(hy), o(hx), CROP_C["H"], FM * FH)[0], ("SIC",))):
            sc.update({f"{key}_{k}": round(float(v), 5) for k, v in
                       model._accumulate(key, ens[None], tgt[None], m[None], names).items()})
        metrics[date] = {"tile": [iy0, ix0], "m_crop": [oy, ox], "h_crop": [hy, hx], **sc}

        # ---- figure ----
        e_c = extent(iy0, ix0, D.N_C)
        e_m = extent(iy0, ix0, CROP_C["M"], oy, ox)
        e_h = extent(iy0, ix0, CROP_C["H"], hy, hx)
        land_c = cpu["omega_c"].numpy() > 0
        land_m = crop(cpu["omega_m"][None], o(oy), o(ox), CROP_C["M"], FM)[0].numpy() > 0
        land_h = crop(cpu["omega_h"][None], o(hy), o(hx), CROP_C["H"], FM * FH)[0].numpy() > 0
        sic = dict(cmap="Blues_r", vmin=0, vmax=1)
        spread = dict(cmap="magma")
        mean_c, std_c = zc.mean(0).numpy(), zc.std(0).numpy()
        zi = {n: i for i, n in enumerate(Z_NAMES)}
        aniso = np.hypot(mean_c[zi["aniso_c"]], mean_c[zi["aniso_s"]])
        rows = [
            ("C 32 km", [("SIC ens. mean", masked(mean_c[0], land_c), e_c, sic, e_m, ~land_c),
                         ("SIC ens. spread", masked(std_c[0], land_c), e_c, spread, e_m, ~land_c),
                         ("SIT ens. mean (m)", masked(mean_c[1], land_c), e_c, dict(cmap="viridis"), None, ~land_c),
                         ("snow ens. mean (m)", masked(mean_c[2], land_c), e_c, dict(cmap="viridis"), None, ~land_c),
                         ("X^_C SIC (Y^tgt)", masked(cpu["tgt_C"][0].numpy(), cpu["tgt_C_m"][0].numpy()), e_c, sic, None, ~land_c),
                         ("CIMR SIC (Y^in, day t)", masked(cpu["in_C_cimr"][-1, 0].numpy(),
                                                          cpu["in_C_cimr_w"][-1, 0].numpy() > 0), e_c, sic, None, ~land_c)]),
            ("S_C, psi_C", [("open water h_ow", masked(mean_c[zi["h_ow"]], land_c), e_c, dict(cmap="cividis", vmin=0, vmax=1), None, ~land_c),
                            ("pack h_pack", masked(mean_c[zi["h_pack"]], land_c), e_c, dict(cmap="cividis", vmin=0, vmax=1), None, ~land_c),
                            ("sic_std", masked(mean_c[zi["sic_std"]], land_c), e_c, dict(cmap="plasma"), None, ~land_c),
                            ("edge_density", masked(mean_c[zi["edge_density"]], land_c), e_c, dict(cmap="plasma"), None, ~land_c),
                            ("anisotropy |a|", masked(aniso, land_c), e_c, dict(cmap="plasma"), None, ~land_c),
                            ("sit_std (m)", masked(mean_c[zi["sit_std"]], land_c), e_c, dict(cmap="plasma"), None, ~land_c),
                            ("psi_C[0]", masked(psi[0].numpy(), land_c), e_c, dict(cmap="RdBu_r"), None, ~land_c),
                            ("psi_C[1]", masked(psi[1].numpy(), land_c), e_c, dict(cmap="RdBu_r"), None, ~land_c)]),
            ("M 4 km (512 km)", [("SIC member 1", masked(xm[0, 0].numpy(), land_m), e_m, sic, e_h, ~land_m),
                                 ("SIC member 2", masked(xm[1 % len(xm), 0].numpy(), land_m), e_m, sic, e_h, ~land_m),
                                 ("SIC ens. mean", masked(xm.mean(0)[0].numpy(), land_m), e_m, sic, e_h, ~land_m),
                                 ("SIC ens. spread", masked(xm.std(0)[0].numpy(), land_m), e_m, spread, e_h, ~land_m),
                                 ("ASIP SIC at M (Y^tgt)", masked(crop(cpu["tgt_M"][None], o(oy), o(ox), CROP_C["M"], FM)[0, 0].numpy(),
                                                                  crop(cpu["tgt_M_m"][None], o(oy), o(ox), CROP_C["M"], FM)[0, 0].numpy()), e_m, sic, None, ~land_m),
                                 ("dSIT_M ens. mean (m)", masked(dmm.mean(0)[1].numpy(), land_m), e_m, dict(cmap="RdBu_r"), None, ~land_m)]),
            ("H 1 km (256 km)", [(f"SIC member {k + 1}", masked(xh[k % len(xh), 0].numpy(), land_h), e_h, sic, None, ~land_h) for k in range(3)]
             + [("SIC ens. mean", masked(xh.mean(0)[0].numpy(), land_h), e_h, sic, None, ~land_h),
                ("SIC ens. spread", masked(xh.std(0)[0].numpy(), land_h), e_h, spread, None, ~land_h),
                ("ASIP SIC (Y^tgt)", masked(crop(cpu["tgt_H"][None], o(hy), o(hx), CROP_C["H"], FM * FH)[0, 0].numpy(),
                                           crop(cpu["tgt_H_m"][None], o(hy), o(hx), CROP_C["H"], FM * FH)[0, 0].numpy()), e_h, sic, None, ~land_h),
                ("ASIP SIC (Y^in, day t)", masked(crop(cpu["in_H"][None], o(hy), o(hx), CROP_C["H"], FM * FH)[0, -1].numpy(),
                                                 crop(cpu["in_H_w"][None], o(hy), o(hx), CROP_C["H"], FM * FH)[0, -1].numpy() > 0), e_h, sic, None, ~land_h)]),
        ]
        panel(os.path.join(args.out, f"panel_{date}.png"),
              f"{date}  tile (iy0, ix0) = ({iy0}, {ix0})  {args.members} members, {n_steps} Heun steps", rows)
        metrics[date]["time_s"] = round(time.perf_counter() - t0, 1)
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"{date}: panel and metrics in {metrics[date]['time_s']} s", flush=True)

    summary(metrics, os.path.join(args.out, "metrics_summary.png"))


def summary(metrics, path):
    """RMSE of the ensemble mean vs spread per level / variable, averaged over the dates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted({k.replace("_rmse_", "|") for d in metrics.values() for k in d if "_rmse_" in k})
    rmse = [np.nanmean([d.get(k.replace("|", "_rmse_"), np.nan) for d in metrics.values()]) for k in keys]
    spr = [np.nanmean([d.get(k.replace("|", "_spread_"), np.nan) for d in metrics.values()]) for k in keys]
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(keys)), 3.5))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, rmse, 0.4, label="RMSE ens. mean")
    ax.bar(x + 0.2, spr, 0.4, label="ens. spread")
    ax.set_xticks(x, [k.replace("|", " ") for k in keys], rotation=60, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title(f"scores on Y^tgt, mean over {len(metrics)} dates", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
