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
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO)
from croscim import data as D  # noqa: E402
from croscim import grid as g  # noqa: E402
from croscim import obs  # noqa: E402
from croscim.cascade import CROP_C, X_NAMES, Z_NAMES, crop, strict_mask, strict_mask_m  # noqa: E402
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


def build(raw, holdout_asip):
    """Batch of one sample; with holdout_asip, every ASIP pixel of day t is a
    target (the 5 km sources keep the usual block partition)."""
    if not holdout_asip:
        return D.build_batch(raw)[0]
    it = D.window({k: v[0] for k, v in raw.items()}, 0)
    part = D.partition(it)
    part["asip"] = it["asip_sic"][-1] != 255
    b = D._stack([D.build_sample(it, part)])
    b.update({k: raw[k] for k in ("day", "iy0", "ix0")})
    return b


def best_crops(b, by="asip"):
    """M crop (16 C cells) then H crop (8 C cells) inside it with the most ASIP
    targets over the ocean at day t (inputs count for 10 %: there must be
    targets at M / H for the metrics). by="sit": the M crop with the most
    CRISTAL SIT targets at M (SIT evaluation), the H crop as before."""
    ocean = b["omega_h"][0] > 0
    obs = (b["tgt_H_m"][0, 0] & ocean).float() + 0.1 * ((b["in_H_w"][0, -1] > 0) & ocean).float()
    cov = g.block_sum(obs, FM * FH)
    nm, nh = CROP_C["M"], CROP_C["H"]
    cand = [(y, x) for y in range(0, D.N_C - nm + 1, 4) for x in range(0, D.N_C - nm + 1, 4)]
    cov_m = cov if by == "asip" else g.block_sum(strict_mask_m(b)[0, 1].float(), FM)
    oy, ox = max(cand, key=lambda c: float(cov_m[c[0]:c[0] + nm, c[1]:c[1] + nm].sum()))
    sub = [(oy + dy, ox + dx) for dy in range(0, nm - nh + 1, 2) for dx in range(0, nm - nh + 1, 2)]
    hy, hx = max(sub, key=lambda c: float(cov[c[0]:c[0] + nh, c[1]:c[1] + nh].sum()))
    return oy, ox, hy, hx


@torch.no_grad()
def ensemble(model, b, n_members, n_steps, oy, ox, hy, hx):
    nets, dev = model.ema.module, b["omega_c"].device
    keep = model.keep_groups(1, dev, drop=False)
    ctx = model.encode(nets, b)
    t = lambda v: torch.tensor([v], device=dev)  # noqa: E731
    zc, xm, xh, dm, dh = [], [], [], [], []
    for m in range(n_members):
        gen = torch.Generator(dev).manual_seed(m)
        z_c = model.sample_c(nets, ctx, b, keep, n_steps, generator=gen)
        d_m = model.sample_m(nets, ctx, b, z_c, ctx["psi"], t(oy), t(ox), keep, n_steps, generator=gen)
        x_m = model.x_m(z_c, d_m, b, t(oy), t(ox))
        x_m_h = crop(model.coarse_h(x_m, b, t(oy), t(ox)), t(hy - oy), t(hx - ox), CROP_C["H"], FM * FH)  # P' X_M at H
        d_h = model.sample_h(nets, ctx, b, z_c, x_m_h, ctx["psi"], t(hy), t(hx), keep, n_steps, generator=gen)
        zc.append(z_c[0]), xm.append(x_m[0]), dm.append(d_m[0]), xh.append((x_m_h[:, :1] + d_h)[0]), dh.append(d_h[0])
    stack = lambda v: torch.stack(v).float().cpu()  # noqa: E731
    return stack(zc), stack(xm), stack(dm), stack(xh), stack(dh), ctx["psi"][0].float().cpu()


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
    ap.add_argument("--crop-by", choices=("asip", "sit"), default="asip",
                    help="M crop with the most ASIP (default) or CRISTAL SIT targets (SIT evaluation)")
    ap.add_argument("--holdout-asip", action="store_true",
                    help="every ASIP pass of day t is a target (none as input): evaluates M / H and the "
                         "ASIP-based S_C channels against a SAR scene the model has not seen")
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
        b = build(raw, args.holdout_asip)
        oy, ox, hy, hx = best_crops(b, args.crop_by)
        with amp:
            zc, xm, dmm, xh, dhh, psi = ensemble(model, b, args.members, n_steps, oy, ox, hy, hx)

        # ---- metrics on Y^tgt ----
        cpu = {k: v[0].float().cpu() if v.is_floating_point() else v[0].cpu() for k, v in b.items()
               if torch.is_tensor(v) and v.ndim > 1}
        o = lambda v: torch.tensor([v])  # noqa: E731
        sc = {}
        m_c = strict_mask(b)[0].cpu()  # without the weak CIMR SIC targets
        tgt_m = crop(cpu["tgt_M"][None], o(oy), o(ox), CROP_C["M"], FM)[0]
        msk_m = crop(strict_mask_m(b)[:1].cpu(), o(oy), o(ox), CROP_C["M"], FM)[0]  # without the open-water zeros
        tgt_h = crop(cpu["tgt_H"][None], o(hy), o(hx), CROP_C["H"], FM * FH)[0]
        msk_h = crop(cpu["tgt_H_m"][None], o(hy), o(hx), CROP_C["H"], FM * FH)[0]
        for key, ens, tgt, m, names in (
                ("C", zc, torch.cat([cpu["tgt_C"], cpu["tgt_S"]]), m_c, Z_NAMES),
                ("M", xm, tgt_m, msk_m, ("SIC", "SIT")), ("H", xh, tgt_h, msk_h, ("SIC",))):
            sc.update({f"{key}_{k}": round(float(v), 5) for k, v in
                       model._accumulate(key, ens[None], tgt[None], m[None], names).items()})
        # what G_M / G_H add: the same scores for the prolonged coarser field alone (P' X_C at M,
        # P' X_M at H, ensemble mean), and the spread of the generated part dX
        for key, coarse, d, tgt, m, names in (("M", xm - dmm, dmm, tgt_m, msk_m, ("SIC", "SIT")),
                                              ("H", xh - dhh, dhh, tgt_h, msk_h, ("SIC",))):
            for c, name in enumerate(names):
                if m[c].any():
                    sc[f"{key}_coarse_rmse_{name}"] = round(float(((coarse.mean(0)[c] - tgt[c])[m[c]] ** 2).mean().sqrt()), 5)
                    sc[f"{key}_dX_spread_{name}"] = round(float(d[:, c].std(0)[m[c]].mean()), 5)
        # references on the strict C targets (SIC: ASIP, SIT: CRISTAL): CIMR of day t (the prior,
        # all of it in Y^in) and of day t-1 (persistence), next to the model on the same cells
        mean_c = zc.mean(0)
        for c, name in enumerate(("SIC", "SIT")):
            for ref, d in (("cimr", -1), ("pers", -2)):
                m = m_c[c] & (cpu["in_C_cimr_w"][d, c] > 0)
                if m.any():
                    tgt, dif = cpu["tgt_C"][c][m], cpu["in_C_cimr"][d, c][m] - cpu["tgt_C"][c][m]
                    sc[f"C_{ref}_rmse_{name}"] = round(float((dif ** 2).mean().sqrt()), 5)
                    sc[f"C_{ref}_bias_{name}"] = round(float(dif.mean()), 5)
                    sc[f"C_model_on_{ref}_cells_rmse_{name}"] = round(float(((mean_c[c][m] - tgt) ** 2).mean().sqrt()), 5)
                    sc[f"C_{ref}_n_cells_{name}"] = int(m.sum())
        # SIT and snow where the model has open water (ensemble-mean SIC_C < 0.15; cell means: should be ~0)
        ow_c = (cpu["omega_c"] > 0) & (mean_c[0] < 0.15)
        if ow_c.any():
            sc["C_SIT_mean_openwater"] = round(float(mean_c[1][ow_c].mean()), 4)
            sc["C_HS_mean_openwater"] = round(float(mean_c[2][ow_c].mean()), 4)
        # ice edge and small scales vs ASIP (SIC, M and H crops):
        #   IIEE: area where the model and ASIP disagree on SIC >= 0.15, over the ASIP-covered pixels
        #   (ensemble mean, mean over members, prolonged coarser field alone), and that covered area;
        #   hf_std: std of SIC minus its 3 x 3 local mean, on pixels whose 3 x 3 window is all covered
        #   (ASIP vs members): too large = pixel noise that ASIP does not have
        def hf(x):  # x [n, Y, X]
            return x - F.avg_pool2d(x[:, None], 3, 1, 1, count_include_pad=False)[:, 0]
        for key, ens, coarse, tgt, m, km2 in (("M", xm[:, 0], (xm - dmm)[:, 0], tgt_m[0], msk_m[0], 16.0),
                                               ("H", xh[:, 0], (xh - dhh)[:, 0], tgt_h[0], msk_h[0], 1.0)):
            if m.sum() < 100:
                continue
            ice = tgt >= 0.15
            iiee = lambda f: float(((f >= 0.15) != ice)[m].sum()) * km2  # noqa: E731
            sc[f"{key}_iiee_mean_km2"] = round(iiee(ens.mean(0)), 1)
            sc[f"{key}_iiee_member_km2"] = round(float(np.mean([iiee(e) for e in ens])), 1)
            sc[f"{key}_iiee_coarse_km2"] = round(iiee(coarse.mean(0)), 1)
            sc[f"{key}_covered_km2"] = round(float(m.sum()) * km2, 1)
            full = F.avg_pool2d(m.float()[None, None], 3, 1, 1)[0, 0] > 0.999
            if full.any():
                sc[f"{key}_hf_std_asip_SIC"] = round(float(hf(tgt[None])[0][full].std()), 5)
                sc[f"{key}_hf_std_member_SIC"] = round(float(np.mean([float(h[full].std()) for h in hf(ens)])), 5)
        # fidelity to what the model saw: SIC vs the day-t ASIP of Y^in (empty with --holdout-asip;
        # not an independent score), model and prolonged coarser field alone
        for key, ens, coarse, y_in, w_in in (
                ("M", xm[:, 0], (xm - dmm)[:, 0], crop(cpu["in_M_asip"][None, -1:], o(oy), o(ox), CROP_C["M"], FM)[0, 0],
                 crop(cpu["in_M_asip_w"][None, -1:], o(oy), o(ox), CROP_C["M"], FM)[0, 0]),
                ("H", xh[:, 0], (xh - dhh)[:, 0], crop(cpu["in_H"][None, -1:], o(hy), o(hx), CROP_C["H"], FM * FH)[0, 0],
                 crop(cpu["in_H_w"][None, -1:], o(hy), o(hx), CROP_C["H"], FM * FH)[0, 0])):
            m = w_in > 0
            if m.sum() >= 100:
                sc[f"{key}_rmse_vs_in_SIC"] = round(float(((ens.mean(0) - y_in)[m] ** 2).mean().sqrt()), 5)
                sc[f"{key}_coarse_rmse_vs_in_SIC"] = round(float(((coarse.mean(0) - y_in)[m] ** 2).mean().sqrt()), 5)
                sc[f"{key}_n_in"] = int(m.sum())
        # SIT at M: the correction dSIT vs the correction needed by the CRISTAL targets (Y - P' X_C),
        # and where there is no target to constrain it (open water: ensemble-mean SIC_M < 0.15)
        d_sit, sic_mean = dmm.mean(0)[1], xm.mean(0)[0]
        ocean_m = crop(cpu["omega_m"][None], o(oy), o(ox), CROP_C["M"], FM)[0] > 0
        m = msk_m[1]
        sc["M_n_SIT_targets"] = int(m.sum())
        if m.sum() >= 5:
            need = (tgt_m[1] - (xm - dmm).mean(0)[1])[m]
            got = d_sit[m]
            sc["M_dSIT_needed_mean"] = round(float(need.mean()), 4)
            sc["M_dSIT_needed_std"] = round(float(need.std()), 4)
            sc["M_dSIT_on_targets_mean"] = round(float(got.mean()), 4)
            sc["M_dSIT_on_targets_std"] = round(float(got.std()), 4)
            if need.std() > 0 and got.std() > 0:
                sc["M_dSIT_corr_needed"] = round(float(torch.corrcoef(torch.stack([need, got]))[0, 1]), 3)
        for name, sel in (("openwater", ocean_m & (sic_mean < 0.15)), ("ice", ocean_m & (sic_mean >= 0.15))):
            if sel.any():
                sc[f"M_dSIT_mean_{name}"] = round(float(d_sit[sel].mean()), 4)
                sc[f"M_dSIT_absmean_{name}"] = round(float(d_sit[sel].abs().mean()), 4)
                sc[f"M_SIT_mean_{name}"] = round(float(xm.mean(0)[1][sel].mean()), 4)
        # SIT realism at M (a generative ensemble is judged on its members' texture and distribution,
        # and on its mean not being worse than the coarse field):
        #   sub-cell variability: std of SIT inside each C cell (as sit_std of S_C), members vs CRISTAL
        #   on the CRISTAL pixels (cells with >= MIN_SIT_OBS of them), and members on all the ocean
        #   pixels vs the sit_std predicted by G_C;
        #   distribution of the members at the CRISTAL pixels vs CRISTAL (quantiles, Wasserstein-1);
        #   ensemble: CRPS vs the MAE of the coarse field, spread / RMSE, biases;
        #   cell means of the correction dSIT where CRISTAL has nothing (systematic shifts).
        sit_m, cr, m_cr = xm[:, 1], tgt_m[1], msk_m[1]
        coarse_sit = (xm - dmm).mean(0)[1]
        fm = g.factor("M", "C")
        tgt_std, n_obs = obs.sit_std_stat(cr, m_cr.float())
        cells = n_obs >= obs.MIN_SIT_OBS
        if cells.any():
            mem_std = torch.stack([obs.sit_std_stat(s_, m_cr.float())[0] for s_ in sit_m])
            sc["M_sitstd_cristal"] = round(float(tgt_std[cells].mean()), 4)
            sc["M_sitstd_members_on_cristal"] = round(float(mem_std[:, cells].mean()), 4)
            sc["M_sitstd_coarse_on_cristal"] = round(float(obs.sit_std_stat(coarse_sit, m_cr.float())[0][cells].mean()), 4)
            sc["M_sitstd_n_cells"] = int(cells.sum())
        ice_c = g.restrict(((sic_mean >= 0.15) & ocean_m).float(), ocean_m.float(), fm)[0] > 0.5
        if ice_c.any():
            all_std = torch.stack([obs.sit_std_stat(s_, ocean_m.float())[0] for s_ in sit_m])
            gc_std = crop(zc.mean(0)[None, Z_NAMES.index("sit_std")][None], o(oy), o(ox), CROP_C["M"], 1)[0, 0]
            sc["M_sitstd_members_ice"] = round(float(all_std[:, ice_c].mean()), 4)
            sc["M_sitstd_G_C_ice"] = round(float(gc_std[ice_c].mean()), 4)
        if m_cr.sum() >= 10:
            y = cr[m_cr]
            mem = sit_m[:, m_cr].flatten()
            q = torch.tensor([0.1, 0.5, 0.9])
            sc["M_SIT_q10_50_90_cristal"] = [round(float(v), 3) for v in torch.quantile(y, q)]
            sc["M_SIT_q10_50_90_members"] = [round(float(v), 3) for v in torch.quantile(mem, q)]
            sc["M_SIT_q10_50_90_coarse"] = [round(float(v), 3) for v in torch.quantile(coarse_sit[m_cr], q)]
            qq = torch.linspace(0.01, 0.99, 99)
            w1 = lambda a_: float((torch.quantile(a_, qq) - torch.quantile(y, qq)).abs().mean())  # noqa: E731
            sc["M_SIT_w1_members"], sc["M_SIT_w1_coarse"] = round(w1(mem), 4), round(w1(coarse_sit[m_cr]), 4)
            sc["M_SIT_bias_ensmean"] = round(float((sit_m.mean(0)[m_cr] - y).mean()), 4)
            sc["M_SIT_bias_coarse"] = round(float((coarse_sit[m_cr] - y).mean()), 4)
            sc["M_SIT_mae_coarse"] = round(float((coarse_sit[m_cr] - y).abs().mean()), 4)
            if sc.get("M_rmse_SIT"):
                sc["M_SIT_spread_over_rmse"] = round(sc["M_spread_SIT"] / sc["M_rmse_SIT"], 3)
        has_cr = g.block_sum(m_cr.float(), fm) > 0
        ocean_cell = g.restrict(ocean_m.float(), torch.ones_like(ocean_m.float()), fm)[0] > 0.5
        cell_mean_d = g.restrict(dmm.mean(0)[1], ocean_m.float(), fm)[0].abs()
        for name, sel in (("notarget", ocean_cell & ~has_cr), ("target", ocean_cell & has_cr)):
            if sel.any():
                sc[f"M_dSIT_cellmean_abs_{name}"] = round(float(cell_mean_d[sel].mean()), 4)
        metrics[date] = {"tile": [iy0, ix0], "m_crop": [oy, ox], "h_crop": [hy, hx],
                         "holdout_asip": args.holdout_asip, **sc}

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
                         ("X^_C SIC (Y^tgt, ASIP)", masked(cpu["tgt_C"][0].numpy(), m_c[0].numpy()), e_c, sic, None, ~land_c),
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
                                 ("dSIT_M ens. mean (m)", masked(dmm.mean(0)[1].numpy(), land_m), e_m, dict(cmap="RdBu_r"), None, ~land_m),
                                 ("SIT ens. mean (m)", masked(xm.mean(0)[1].numpy(), land_m), e_m, dict(cmap="viridis", vmin=0), None, ~land_m),
                                 ("SIT member 1 (m)", masked(xm[0, 1].numpy(), land_m), e_m, dict(cmap="viridis", vmin=0), None, ~land_m),
                                 ("CRISTAL SIT at M (Y^tgt)", masked(tgt_m[1].numpy(), msk_m[1].numpy()), e_m, dict(cmap="viridis", vmin=0), None, ~land_m)]),
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
    rank_histograms(model._ranks, args.out)


def rank_histograms(ranks: dict, out: str):
    """Rank of the targets among the members, accumulated over the dates
    (flat: calibrated; U: under-dispersed; dome: over-dispersed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.save({k: v.cpu() for k, v in ranks.items()}, os.path.join(out, "ranks.pt"))
    show = [("C", c, n) for c, n in enumerate(Z_NAMES) if n in ("SIC", "SIT", "HS", "h_ow", "sic_std", "edge_density", "aniso_c")]
    show += [("M", 0, "SIC"), ("M", 1, "SIT"), ("H", 0, "SIC")]
    show = [s for s in show if s[0] in ranks]
    fig, axes = plt.subplots(1, len(show), figsize=(2.2 * len(show), 2.4), squeeze=False)
    for ax, (key, c, name) in zip(axes[0], show):
        h = ranks[key][c].cpu().numpy()
        ax.bar(np.arange(len(h)), h / max(h.sum(), 1), color="tab:blue")
        ax.axhline(1 / len(h), color="k", lw=0.8, ls="--")
        ax.set_title(f"{key} {name}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("rank histograms of the Y^tgt targets (dashed: calibrated)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rank_histograms.png"), dpi=110)
    plt.close(fig)


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
