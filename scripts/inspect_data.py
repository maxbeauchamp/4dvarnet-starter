#!/usr/bin/env python3
"""Inspect the cascade dataloader on real data (test plan, step 2).

Over --n-batches training batches (random tiles and days in [start, end]):
  * timings: worker read of one item, transfer, GPU `build_batch`;
  * target coverage: fraction of ocean cells with a target, per variable /
    channel and level, and size of the input / target partition;
  * normalisation: mean / std of the S_C channels and of the detail targets
    (to replace the provisional values of config/xp/N0.yaml);
  * reference score: persistence of CIMR (day t-1) against the C targets of
    day t (RMSE), the first baseline to beat in phase 3;
  * sensitivity of edge_density (jump threshold) and of the anisotropy
    (energy floor e0) to their settings, from the 1 km targets X^_H;
  * maps of the first --n-plots batches (targets, masks, partition).

Results go to --out (summary.json, norm_N0.yaml, PNG maps), rewritten every
10 batches so that a stopped job still leaves them; nothing is written in the
repository.

Usage:
    python scripts/inspect_data.py --start 2022-02-01 --end 2022-02-28 \
        --out /Odyssey/private/$USER/results/inspect
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from croscim import obs
from croscim import grid as g
from croscim.cascade import X_NAMES
from croscim.data import CascadeDataModule

EDGE_JUMPS = (0.1, 0.15, 0.3)
ANISO_E0S = (1e-4, 1e-3, 1e-2)
ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "ne_10m_land.zip")


def masked_moments(x: torch.Tensor, m: torch.Tensor, dims) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sums of x, x^2 and counts over the masked entries, kept per channel (dim 1)."""
    m = m.to(x.dtype)
    return (x * m).sum(dims), (x * x * m).sum(dims), m.sum(dims)


def plot(b: dict, i: int, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def show(ax, x, m, title, **kw):
        x = np.where(m, x, np.nan)
        im = ax.imshow(x, interpolation="nearest", **kw)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    c = {k: v[i].float().cpu().numpy() for k, v in b.items() if torch.is_tensor(v) and v.ndim > 1}
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    ax = axes.flatten()
    for k, name in enumerate(X_NAMES):
        show(ax[k], c["tgt_C"][k], c["tgt_C_m"][k], f"X^_C {name} (Y^tgt, day t)")
    show(ax[3], c["in_C_cimr"][-1, 0], c["in_C_cimr_w"][-1, 0] > 0, "CIMR SIC Y^in (day t)")
    show(ax[4], c["omega_c"], np.ones_like(c["omega_c"], bool), "ocean fraction C", vmin=0, vmax=1)
    for k, ch in enumerate((3, 4, 5, 6)):  # h_pack, sic_std, edge_density, aniso_c
        show(ax[5 + k], c["tgt_S"][ch], c["tgt_S_m"][ch], f"S^_C {obs.S_NAMES[ch]}")
    show(ax[9], c["tgt_S"][8], c["tgt_S_m"][8], "S^_C sit_std")
    show(ax[10], c["tgt_dM"][0], c["tgt_dM_m"][0], "dX^_M SIC", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    show(ax[11], c["tgt_dM"][1], c["tgt_dM_m"][1], "dX^_M SIT", cmap="RdBu_r")
    show(ax[12], c["tgt_dH"][0], c["tgt_dH_m"][0], "dX^_H SIC", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    part = np.where(c["tgt_H_m"][0], 2.0, np.where(c["in_H_w"][-1] > 0, 1.0, np.nan))
    show(ax[13], part, np.isfinite(part), "ASIP day t: 1 = Y^in, 2 = Y^tgt", vmin=0.5, vmax=2.5, cmap="coolwarm")
    show(ax[14], c["in_M_asip_w"].astype(bool).sum(0), np.ones(c["in_M_asip_w"].shape[1:], bool),
         "ASIP: observed days at M (W window)")
    fig.suptitle(f"day {int(b['day'][i])}  tile (iy0, ix0) = ({int(b['iy0'][i])}, {int(b['ix0'][i])})")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--n-plots", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)

    t0 = time.perf_counter()
    dm = CascadeDataModule(args.root, ASSET, *[[args.start, args.end]] * 3,
                           dl_kw={"batch_size": args.batch_size, "num_workers": args.num_workers, "shuffle": True})
    dm.setup()
    t_setup = time.perf_counter() - t0
    ds = dm.ds["train"]
    print(f"setup (ocean fraction + stores): {t_setup:.1f} s, {len(ds.days)} days, {len(ds.origins)} tile origins")

    t0 = time.perf_counter()
    ds[0]
    t_item = time.perf_counter() - t0

    acc, t_wait, t_build = {}, [], []

    def report(n_done):
        """Write summary.json / norm_N0.yaml from the batches seen so far."""
        def stats(key):
            n = acc[f"{key}_n"].clamp_min(1)
            mean = acc[f"{key}_s1"] / n
            return mean, (acc[f"{key}_s2"] / n - mean ** 2).clamp_min(0).sqrt()

        s_mean, s_std = stats("tgt_S")
        dm_mean, dm_std = stats("tgt_dM")
        dh_mean, dh_std = stats("tgt_dH")
        x_mean, x_std = stats("tgt_C")
        summary = {
            "days": len(ds.days), "n_batches": n_done, "batch_size": args.batch_size,
            "time_s": {"setup": t_setup, "item_read_single": t_item, "wait_median": float(np.median(t_wait[1:] or t_wait)),
                       "build_median": float(np.median(t_build[1:] or t_build))},
            "coverage": {key: dict(zip(names, torch.stack(acc[f"cov_{key}"]).mean(0).tolist()))
                         for key, names in (("tgt_C_m", X_NAMES), ("tgt_S_m", obs.S_NAMES),
                                            ("tgt_dM_m", ("dSIC_M", "dSIT_M")), ("tgt_dH_m", ("dSIC_H",)))},
            "asip_cells_day_t": {"in": float(np.mean(acc["asip_in_cells"])), "tgt": float(np.mean(acc["asip_tgt_cells"]))},
            "x_c_stats": {n: [float(m), float(s)] for n, m, s in zip(X_NAMES, x_mean, x_std)},
            "persistence_rmse_cimr_t-1": {n: (acc[f"pers_{n}_se"] / max(acc[f"pers_{n}_n"], 1)) ** 0.5 for n in ("SIC", "SIT")},
            # over the C cells where S^_C is defined: mean, std, fraction of cells > 0.01 (in absolute value)
            "s_settings": {k: [round(v[0] / max(v[3], 1), 5), round(max(v[1] / max(v[3], 1) - (v[0] / max(v[3], 1)) ** 2, 0) ** 0.5, 5),
                               round(v[2] / max(v[3], 1), 4)] for k, v in acc.get("s_settings", {}).items()},
        }
        norm = {n: [round(float(m), 4), round(float(max(s, 1e-3)), 4)] for n, m, s in zip(obs.S_NAMES, s_mean, s_std)}
        norm.update({"dSIC_M": [0.0, round(float(dm_std[0]), 4)], "dSIT_M": [0.0, round(float(dm_std[1]), 4)],
                     "dSIC_H": [0.0, round(float(dh_std[0]), 4)]})
        summary["norm_proposal"] = norm
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.out, "norm_N0.yaml"), "w") as f:
            f.write("# proposed entries of model.norm in config/xp/N0.yaml ([mean, std])\n")
            f.writelines(f"{k}: {v}\n" for k, v in norm.items())
        return summary

    loader = iter(dm.train_dataloader())
    for k in range(args.n_batches):
        t0 = time.perf_counter()
        try:
            raw = next(loader)
        except StopIteration:  # fewer items than --n-batches: new epoch (new random tiles)
            loader = iter(dm.train_dataloader())
            raw = next(loader)
        t_wait.append(time.perf_counter() - t0)
        raw = {key: v.to(dev, non_blocking=True) for key, v in raw.items()}
        sync()
        t0 = time.perf_counter()
        b = dm.on_after_batch_transfer(raw, 0)
        sync()
        t_build.append(time.perf_counter() - t0)

        ocean = {lvl: b[f"omega_{lvl.lower()}"] > 0 for lvl in "CMH"}
        for key, lvl in (("tgt_C_m", "C"), ("tgt_S_m", "C"), ("tgt_dM_m", "M"), ("tgt_dH_m", "H")):
            m = b[key] & ocean[lvl][:, None]
            acc.setdefault(f"cov_{key}", []).append((m.float().sum((0, 2, 3)) / ocean[lvl].sum().clamp_min(1)).cpu())
        for key, mkey in (("tgt_S", "tgt_S_m"), ("tgt_dM", "tgt_dM_m"), ("tgt_dH", "tgt_dH_m"), ("tgt_C", "tgt_C_m")):
            s1, s2, n = masked_moments(b[key].double(), b[mkey], (0, 2, 3))
            for name, v in (("s1", s1), ("s2", s2), ("n", n)):
                acc[f"{key}_{name}"] = acc.get(f"{key}_{name}", 0) + v.cpu()
        # sensitivity of edge_density / anisotropy to their settings, from X^_H
        w_h = b["omega_h"] * b["tgt_H_m"][:, 0]
        cells = b["tgt_S_m"][:, 0]
        variants = {f"edge_density@jump={j}": (obs.sic_stats(b["tgt_H"][:, 0], w_h, edge_jump=j)[:, 5],) for j in EDGE_JUMPS}
        variants.update({f"aniso@e0={e}": tuple(obs.sic_stats(b["tgt_H"][:, 0], w_h, e0=e)[:, 6:8].unbind(1))
                         for e in ANISO_E0S})
        sv = acc.setdefault("s_settings", {})
        for name, chans in variants.items():
            for c, x in enumerate(chans):
                x = x[cells].double()
                key = name if len(chans) == 1 else name.replace("aniso", ("aniso_c", "aniso_s")[c])
                old = sv.get(key, [0.0, 0.0, 0.0, 0])
                sv[key] = [old[0] + float(x.sum()), old[1] + float((x * x).sum()),
                           old[2] + float((x.abs() > 0.01).sum()), old[3] + int(x.numel())]
        # partition of ASIP at day t (H cells): Y^in vs Y^tgt
        acc.setdefault("asip_in_cells", []).append(float((b["in_H_w"][:, -1] > 0).sum()))
        acc.setdefault("asip_tgt_cells", []).append(float(b["tgt_H_m"].sum()))
        # persistence of CIMR (day t-1, all obs) against the C targets of day t
        for c, name in enumerate(("SIC", "SIT")):
            m = b["tgt_C_m"][:, c] & (b["in_C_cimr_w"][:, -2, c] > 0)
            err = (b["in_C_cimr"][:, -2, c] - b["tgt_C"][:, c])[m]
            acc[f"pers_{name}_se"] = acc.get(f"pers_{name}_se", 0.0) + float((err ** 2).sum())
            acc[f"pers_{name}_n"] = acc.get(f"pers_{name}_n", 0) + int(m.sum())
        if k < args.n_plots:
            plot(b, 0, os.path.join(args.out, f"batch{k:03d}.png"))
        print(f"batch {k + 1}/{args.n_batches}: wait {t_wait[-1]:.2f} s, build {t_build[-1]:.2f} s", flush=True)
        if (k + 1) % 10 == 0:
            report(k + 1)  # partial results, usable if the job is stopped

    summary = report(args.n_batches)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
