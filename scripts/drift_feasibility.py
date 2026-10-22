#!/usr/bin/env python3
"""
Feasibility of a sea-ice drift learned from the observations only (ASIP,
CIMR, CRISTAL + ERA5 forcing), before any change to the model: no training,
on the stored ASIP 4 km maps (M grid) of consecutive days t-1, t.

1. Free drift: SIC_t predicted by advecting SIC_{t-1} (semi-Lagrangian,
   bilinear) with u_ice = alpha R(-theta) u10 (ERA5, mean of the two days),
   over the time between the two acquisitions of each pixel; RMSE against
   persistence on the pixels observed both days, and in the edge zone
   (0.1 < SIC < 0.9 on either day). Two wind frames are tried: u10 / v10
   east / north (rotated to the grid axes) or already along the grid axes;
   the right one is the one with skill.
2. Block matching (feature tracking from ASIP alone): for each 64 km block
   with texture, the integer shift (4 km steps) that best maps SIC_{t-1} on
   SIC_t. How often a displacement is identifiable, and how it compares with
   the free drift (speed ratio, turning angle, complex vector correlation).
   Displacements below 4 km / day are not resolved.

Outputs in --out: metrics.json (per pair), summary.json, pair_<date>.png.

Usage:
    sbatch scripts/submit_odyssey.sbatch scripts/drift_feasibility.py \\
        --start 2022-01-15 --end 2022-10-31 --every 7 --out /Odyssey/private/m19beauc/results/drift_feasibility
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import zarr

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO)

from croscim import grid as g  # noqa: E402
from croscim.layout import EPOCH, STORES  # noqa: E402

PER_M = 8            # 500 m pixels per 4 km cell (one side)
DXM = g.DX_M["M"]    # m
EDGE = (0.1, 0.9)    # edge zone: SIC strictly inside on either day
FRAMES = ("geo", "grid")


def day_of(s: str) -> int:
    return (dt.date.fromisoformat(s) - EPOCH).days


def date_of(d: int) -> str:
    return (EPOCH + dt.timedelta(days=int(d))).isoformat()


def open_stores(root):
    a, e = os.path.join(root, STORES["asip"]), os.path.join(root, STORES["era5"])
    asip = {k: zarr.open_array(os.path.join(a, k), mode="r") for k in ("acq", "m_mean", "m_w", "days")}
    era5 = {k: zarr.open_array(os.path.join(e, k), mode="r") for k in ("u10", "v10", "xc", "yc", "days")}
    for s in (asip, era5):
        s["index"] = {int(d): i for i, d in enumerate(s["days"][:])}
    return asip, era5


def acq_m(acq, i: int, band: int = 2048) -> np.ndarray:
    """Mean acquisition time (minutes since 00:00) of the 500 m pixels of each
    M cell (NaN if none), read in bands of rows."""
    n = acq.shape[-1]
    nm = n // PER_M
    out = np.full((nm, nm), np.nan, np.float32)
    for r0 in range(0, n, band):
        a = acq[i, r0:r0 + band].astype(np.float32)
        ok = a != -32768
        a[~ok] = 0.0
        rb = a.shape[0] // PER_M
        s = a.reshape(rb, PER_M, nm, PER_M).sum((1, 3))
        c = ok.reshape(rb, PER_M, nm, PER_M).sum((1, 3))
        out[r0 // PER_M:r0 // PER_M + rb] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out


def wind_m(era5, d: int, xm: np.ndarray, ym: np.ndarray):
    """ERA5 (u10, v10) of day d on the M cells (nearest 5 km cell), NaN outside."""
    i = era5["index"].get(d)
    if i is None:
        return None
    xc, yc = era5["xc"][:], era5["yc"][:]
    jj = np.rint((xm - xc[0]) / (xc[1] - xc[0])).astype(int)
    ii = np.rint((ym - yc[0]) / (yc[1] - yc[0])).astype(int)
    okj, oki = (jj >= 0) & (jj < len(xc)), (ii >= 0) & (ii < len(yc))
    out = []
    for name in ("u10", "v10"):
        f = era5[name][i]
        w = np.full((len(ym), len(xm)), np.nan, np.float32)
        w[np.ix_(oki, okj)] = f[np.ix_(ii[oki], jj[okj])]
        out.append(w)
    return out


def free_drift(u, v, delta, frame: str, alpha: float, theta_deg: float):
    """Ice velocity along the grid axes (m/s): alpha x wind turned theta to the
    right. frame "geo": (u, v) east / north, rotated by delta = lon - lon_0."""
    if frame == "geo":
        ux, uy = u * torch.cos(delta) - v * torch.sin(delta), u * torch.sin(delta) + v * torch.cos(delta)
    else:
        ux, uy = u, v
    th = -math.radians(theta_deg)
    return alpha * (ux * math.cos(th) - uy * math.sin(th)), alpha * (ux * math.sin(th) + uy * math.cos(th))


def advect(prev, valid, ux, uy, dt_s):
    """pred(x) = prev(x - u dt), bilinear; also whether the departure point is
    surrounded by valid pixels. Columns go along +x, rows along -y."""
    n = prev.shape[-1]
    ii, jj = torch.meshgrid(torch.arange(n, device=prev.device, dtype=prev.dtype),
                            torch.arange(n, device=prev.device, dtype=prev.dtype), indexing="ij")
    src_c, src_r = jj - ux * dt_s / DXM, ii + uy * dt_s / DXM
    grid = torch.stack([2 * src_c / (n - 1) - 1, 2 * src_r / (n - 1) - 1], -1)[None]
    p = F.grid_sample(torch.where(valid, prev, 0.0)[None, None], grid, mode="bilinear", align_corners=True)[0, 0]
    w = F.grid_sample(valid.to(prev)[None, None], grid, mode="bilinear", align_corners=True)[0, 0]
    return p, w > 0.999


def blocks(x, b: int):
    n = x.shape[-1] // b
    return x.reshape(n, b, n, b).sum((1, 3))


def block_match(s0, v0, s1, v1, max_shift: int, b: int):
    """MSE of SIC_t vs SIC_{t-1} shifted by (dy, dx) M cells, per block of b x b
    cells: mse, count [2S+1, 2S+1, nb, nb]."""
    n, S = s1.shape[-1], max_shift
    pad = torch.full((n + 2 * S, n + 2 * S), float("nan"), device=s1.device)
    pad[S:S + n, S:S + n] = torch.where(v0, s0, float("nan"))
    tgt = torch.where(v1, s1, float("nan"))
    nb = n // b
    mse = torch.full((2 * S + 1, 2 * S + 1, nb, nb), float("inf"), device=s1.device)
    cnt = torch.zeros_like(mse)
    for dy in range(-S, S + 1):
        for dx in range(-S, S + 1):
            sh = pad[S - dy:S - dy + n, S - dx:S - dx + n]  # prev moved by (+dy rows, +dx cols)
            e = (tgt - sh) ** 2
            ok = torch.isfinite(e)
            c = blocks(ok.float(), b)
            cnt[dy + S, dx + S] = c
            mse[dy + S, dx + S] = blocks(torch.where(ok, e, 0.0), b) / c.clamp_min(1)
    return mse, cnt


def block_drift(ux, uy, by, bx, b: int) -> np.ndarray:
    """Block-mean velocity (m/s along the grid axes) of blocks (by, bx), as complex km / day."""
    fx, fy = (blocks(c, b)[by, bx] / b ** 2 * 86.4 for c in (ux, uy))
    return (fx + 1j * fy).cpu().numpy() if len(by) else np.zeros(0, complex)


def compare(trk: np.ndarray, fd: np.ndarray) -> dict:
    """Tracked vs free-drift displacements (complex, km / day)."""
    if len(trk) < 5:
        return {"n": int(len(trk))}
    rho = (trk * np.conj(fd)).sum() / np.sqrt((np.abs(trk) ** 2).sum() * (np.abs(fd) ** 2).sum())
    ang = np.degrees(np.angle(trk * np.conj(fd)))
    ok = np.abs(fd) > 0
    return {"n": int(len(trk)), "vector_corr": round(float(np.abs(rho)), 3),
            "corr_angle_deg": round(float(np.degrees(np.angle(rho))), 1),
            "median_turn_deg": round(float(np.median(ang)), 1),
            "median_speed_ratio": round(float(np.median(np.abs(trk[ok]) / np.abs(fd[ok]))), 3) if ok.any() else None,
            "median_speed_tracked_km_day": round(float(np.median(np.abs(trk))), 2),
            "median_speed_free_km_day": round(float(np.median(np.abs(fd))), 2)}


def figure(path, title, s0, v0, s1, v1, err_p, err_f, arrows, step_km):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [-g.HALF_EXTENT_M / 1e3, g.HALF_EXTENT_M / 1e3, -g.HALF_EXTENT_M / 1e3, g.HALF_EXTENT_M / 1e3]
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6))
    sic = dict(cmap="Blues_r", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    err = dict(cmap="magma", vmin=0, vmax=0.4, extent=ext, interpolation="nearest")
    for a, (name, f, kw) in zip(ax, (("ASIP SIC t-1", np.where(v0, s0, np.nan), sic),
                                     ("ASIP SIC t", np.where(v1, s1, np.nan), sic),
                                     ("|persistence - SIC t|", err_p, err),
                                     ("|free drift - SIC t|", err_f, err),
                                     ("tracked (black) / free drift (red), km/day", np.where(v1, s1, np.nan), sic))):
        im = a.imshow(f, **kw)
        a.set_title(name, fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    x, y, trk, fd = arrows
    if len(x):
        ax[4].quiver(x, y, trk.real, trk.imag, color="k", angles="xy", scale_units="xy", scale=1 / step_km, width=0.003)
        ax[4].quiver(x, y, fd.real, fd.imag, color="r", angles="xy", scale_units="xy", scale=1 / step_km, width=0.003)
    # zoom on the observed area
    yy, xx = np.nonzero(v1)
    if len(xx):
        c = np.array(ext[:2])
        lo_x, hi_x = (xx.min(), xx.max())
        lo_y, hi_y = (yy.min(), yy.max())
        for a in ax:
            a.set_xlim(c[0] + lo_x * DXM / 1e3, c[0] + (hi_x + 1) * DXM / 1e3)
            a.set_ylim(ext[3] - (hi_y + 1) * DXM / 1e3, ext[3] - lo_y * DXM / 1e3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", required=True, help="first day t (pair t-1, t)")
    ap.add_argument("--end", required=True)
    ap.add_argument("--every", type=int, default=7, help="days between two pairs")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.02, 0.03], help="free drift: ice / wind speed")
    ap.add_argument("--theta", type=float, default=25.0, help="free drift: turning angle to the right (deg)")
    ap.add_argument("--max-shift", type=int, default=6, help="block matching: max shift (M cells of 4 km)")
    ap.add_argument("--block", type=int, default=16, help="block matching: block size (M cells)")
    ap.add_argument("--n-fig", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asip, era5 = open_stores(args.root)
    xm_t, ym_t = g.centers("M")
    xm, ym = xm_t.numpy(), ym_t.numpy()
    delta = torch.atan2(torch.as_tensor(xm)[None, :].expand(len(ym), -1),
                        -torch.as_tensor(ym)[:, None].expand(-1, len(xm))).float().to(dev)
    S, B = args.max_shift, args.block
    metrics, pool, n_fig = {}, {f: ([], []) for f in FRAMES}, 0
    for t in range(day_of(args.start), day_of(args.end) + 1, args.every):
        t0 = time.perf_counter()
        i0, i1 = asip["index"].get(t - 1), asip["index"].get(t)
        winds = [wind_m(era5, d, xm, ym) for d in (t - 1, t)]
        if i0 is None or i1 is None or any(w is None for w in winds):
            print(f"{date_of(t)}: missing ASIP or ERA5 day, skipped", flush=True)
            continue
        load = lambda k, i: torch.as_tensor(asip[k][i], device=dev).float()  # noqa: E731
        s0, s1 = load("m_mean", i0), load("m_mean", i1)
        v0, v1 = load("m_w", i0) > 0, load("m_w", i1) > 0
        both = v0 & v1
        if both.sum() < 1000:
            print(f"{date_of(t)}: {int(both.sum())} M cells observed both days, skipped", flush=True)
            continue
        a0, a1 = (torch.as_tensor(acq_m(asip["acq"], i), device=dev) for i in (i0, i1))
        dt_days = torch.where(torch.isfinite(a0) & torch.isfinite(a1), 1.0 + (a1 - a0) / 1440.0, torch.ones_like(a0))
        dt_s = dt_days * 86400.0
        u = torch.as_tensor(0.5 * (winds[0][0] + winds[1][0]), device=dev)
        v = torch.as_tensor(0.5 * (winds[0][1] + winds[1][1]), device=dev)
        wind_ok = torch.isfinite(u) & torch.isfinite(v)
        u, v = torch.nan_to_num(u), torch.nan_to_num(v)
        edge = both & (((s0 > EDGE[0]) & (s0 < EDGE[1])) | ((s1 > EDGE[0]) & (s1 < EDGE[1])))
        m = {"n_both": int(both.sum()), "n_edge": int(edge.sum()),
             "dt_days_median": round(float(dt_days[both].median()), 3)}
        # ---- 1. free drift vs persistence ----
        best = None
        for frame in FRAMES:
            for alpha in args.alphas:
                ux, uy = free_drift(u, v, delta, frame, alpha, args.theta)
                pred, ok = advect(s0, v0, ux, uy, dt_s)
                for zone, z in (("all", both), ("edge", edge)):
                    sel = z & ok & wind_ok
                    if sel.sum() < 100:
                        continue
                    r_p = float(((s0 - s1)[sel] ** 2).mean().sqrt())
                    r_f = float(((pred - s1)[sel] ** 2).mean().sqrt())
                    key = f"{zone}_{frame}_a{alpha:g}"
                    m[f"rmse_pers_{key}"], m[f"rmse_free_{key}"] = round(r_p, 5), round(r_f, 5)
                    m[f"skill_{key}"] = round(1 - (r_f / r_p) ** 2, 4)
                    if zone == "edge" and (best is None or m[f"skill_{key}"] > best[0]):
                        best = (m[f"skill_{key}"], frame, alpha, pred, ok)
        # ---- 2. block matching ----
        mse, cnt = block_match(s0, v0, s1, v1, S, B)
        full = 0.5 * B * B
        mse = torch.where(cnt >= full, mse, float("inf"))
        mse0 = mse[S, S]
        tgt_b = torch.where(v1, s1, float("nan")).reshape(len(ym) // B, B, len(xm) // B, B).permute(0, 2, 1, 3).flatten(2)
        texture = torch.nan_to_num(torch.from_numpy(np.nanstd(tgt_b.cpu().numpy(), -1)).to(dev)) >= 0.05
        flat = mse.flatten(0, 1)
        best_mse, k = flat.min(0)
        dy, dx = k // (2 * S + 1) - S, k % (2 * S + 1) - S
        cand = texture & torch.isfinite(mse0)
        improve = 1 - best_mse / mse0.clamp_min(1e-12)
        tracked = cand & (improve >= 0.3) & (dy.abs() < S) & (dx.abs() < S) & ((dy != 0) | (dx != 0))
        m["blocks_textured"], m["blocks_tracked"] = int(cand.sum()), int(tracked.sum())
        m["blocks_stationary"] = int((cand & (dy == 0) & (dx == 0)).sum())
        by, bx = torch.nonzero(tracked, as_tuple=True)
        dtb = blocks(torch.where(both, dt_days, 0.0), B) / blocks(both.float(), B).clamp_min(1)
        # tracked displacement as complex km / day along the grid axes (+x: columns, +y: -rows)
        trk = ((dx[by, bx] - 1j * dy[by, bx]).cpu().numpy() * DXM / 1e3 / dtb[by, bx].cpu().numpy()
               if len(by) else np.zeros(0, complex))
        for frame in FRAMES:
            fd = block_drift(*free_drift(u, v, delta, frame, 0.02, args.theta), by, bx, B)
            m[f"tracked_vs_free_{frame}"] = compare(trk, fd)
            pool[frame][0].append(trk), pool[frame][1].append(fd)
        if best is not None:
            m["best_edge"] = {"skill": best[0], "frame": best[1], "alpha": best[2]}
        metrics[date_of(t)] = m
        # ---- figure ----
        if n_fig < args.n_fig and best is not None:
            n_fig += 1
            _, frame, alpha, pred, ok = best
            cpu = lambda x: x.cpu().numpy()  # noqa: E731
            err_p = np.where(cpu(both), np.abs(cpu(s0 - s1)), np.nan)
            err_f = np.where(cpu(both & ok), np.abs(cpu(pred - s1)), np.nan)
            cx = (bx.cpu().numpy() + 0.5) * B * DXM / 1e3 - g.HALF_EXTENT_M / 1e3
            cy = g.HALF_EXTENT_M / 1e3 - (by.cpu().numpy() + 0.5) * B * DXM / 1e3
            fd = block_drift(*free_drift(u, v, delta, frame, 0.02, args.theta), by, bx, B)
            figure(os.path.join(args.out, f"pair_{date_of(t)}.png"),
                   f"{date_of(t - 1)} -> {date_of(t)}: free drift {frame}, alpha {alpha:g}, theta {args.theta:g} deg; "
                   f"edge-zone skill {best[0]:.3f}",
                   cpu(s0), cpu(v0), cpu(s1), cpu(v1), err_p, err_f, (cx, cy, trk, fd), 2.0)
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"{date_of(t)}: {m['n_both']} cells both days, edge {m['n_edge']}, "
              f"best edge skill {best[0] if best else float('nan'):.3f}, tracked blocks {m['blocks_tracked']} / "
              f"{m['blocks_textured']} ({time.perf_counter() - t0:.1f} s)", flush=True)

    # ---- summary over the pairs ----
    summ = {"n_pairs": len(metrics)}
    keys = sorted({k for m in metrics.values() for k in m if k.startswith("skill_")})
    for k in keys:
        vals = [m[k] for m in metrics.values() if k in m]
        summ[k] = {"mean": round(float(np.mean(vals)), 4), "median": round(float(np.median(vals)), 4),
                   "frac_positive": round(float(np.mean(np.array(vals) > 0)), 3)}
    for frame in FRAMES:
        trk = np.concatenate(pool[frame][0]) if pool[frame][0] else np.zeros(0, complex)
        fd = np.concatenate(pool[frame][1]) if pool[frame][1] else np.zeros(0, complex)
        summ[f"tracked_vs_free_{frame}"] = compare(trk, fd)
    tex = sum(m["blocks_textured"] for m in metrics.values())
    summ["frac_textured_blocks_tracked"] = round(sum(m["blocks_tracked"] for m in metrics.values()) / max(tex, 1), 3)
    summ["frac_textured_blocks_stationary"] = round(sum(m["blocks_stationary"] for m in metrics.values()) / max(tex, 1), 3)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
