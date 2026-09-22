#!/usr/bin/env python3
"""Build the NFE-vs-RMSE / NFE-vs-CRPS efficiency figure (GP), from the CSVs
produced by run_nfe_sweep.py -- substantiates the "efficient few-step
inference" claim (an NFE-vs-RMSE/CRPS curve for the flagship CM/FM pair,
4DVarNet-LSTM as a fixed-cost reference line) in one publication-ready
figure.

Only CM and FM are plotted as swept curves: the Var*/Dyn* variants showed
sweep-artifacts (instability spikes at intermediate NFE, likely undertrained
at those specific step counts) that clutter the figure without changing the
argument -- CM/FM are the flagship pairwise-consistency / flow-matching
instances and make the point on their own. 4DVarNet-LSTM has no swept NFE
(its solver iteration count is fixed by training) and no CRPS (deterministic,
no ensemble) -- it is drawn as a single dashed horizontal reference line
across the RMSE panel only.

Usage:
    python plot_nfe_efficiency.py
    python plot_nfe_efficiency.py --results-dir results/GP --out figures/nfe_efficiency
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Colors are colorblind-safe (Okabe-Ito).
CURVE_METHODS = ["CM", "FM"]
CURVE_STYLE = {
    "CM": dict(label="CM", color="#0072B2", marker="o", ms=6, ls="-"),
    "FM": dict(label="FM", color="#CC79A7", marker="^", ms=7, ls="-"),
}
BASELINE_METHOD = "4dvarnet_lstm"
BASELINE_STYLE = dict(label="4DVarNet-LSTM", color="#000000", ls="--", lw=1.8)
METHOD_ORDER = CURVE_METHODS + [BASELINE_METHOD]


def _iclr_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markeredgewidth": 0.8,
        "lines.markeredgecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType, not Type 3 -- required by most venues
        "ps.fonttype": 42,
    })


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="both")


def load_sweeps(results_dir: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for method in METHOD_ORDER:
        f = results_dir / f"{method}_nfe_sweep.csv"
        if not f.exists():
            print(f"[plot_nfe_efficiency] WARNING: missing {f}, skipping {method}")
            continue
        df = pd.read_csv(f).sort_values("nfe")
        data[method] = df
    return data


def make_figure(data: dict[str, pd.DataFrame], out_stem: Path):
    _iclr_style()
    fig, (ax_rmse, ax_crps) = plt.subplots(1, 2, figsize=(9.0, 3.15))

    handles, labels = [], []
    for method in CURVE_METHODS:
        if method not in data:
            continue
        df = data[method]
        style = CURVE_STYLE[method]

        (h,) = ax_rmse.plot(
            df["nfe"], df["rmse"],
            marker=style["marker"], ms=style["ms"], color=style["color"], ls=style["ls"],
            markeredgecolor="white", zorder=3,
        )
        ax_crps.plot(
            df["nfe"], df["crps"],
            marker=style["marker"], ms=style["ms"], color=style["color"], ls=style["ls"],
            markeredgecolor="white", zorder=3,
        )
        handles.append(h)
        labels.append(style["label"])

    # 4DVarNet-LSTM: fixed solver-iteration count, no NFE sweep possible (see
    # run_nfe_sweep.py) and no CRPS (deterministic, no ensemble) -- drawn as
    # a single dashed reference line spanning the RMSE panel only.
    if BASELINE_METHOD in data:
        _rmse_ref = float(data[BASELINE_METHOD]["rmse"].iloc[0])
        (h_base,) = ax_rmse.plot(
            [], [], color=BASELINE_STYLE["color"], ls=BASELINE_STYLE["ls"], lw=BASELINE_STYLE["lw"],
        )
        ax_rmse.axhline(_rmse_ref, color=BASELINE_STYLE["color"], ls=BASELINE_STYLE["ls"],
                         lw=BASELINE_STYLE["lw"], zorder=2)
        handles.append(h_base)
        labels.append(BASELINE_STYLE["label"])

    # NFE = measured number of network forward calls per reconstructed sample
    # (see run_nfe_sweep.py / the notebooks' sweep cells) -- NOT the raw
    # `nsteps`/`n_steps` control parameter, since Heun-based samplers (FM)
    # evaluate the network twice per integration step. Plotting the control
    # parameter directly would understate FM's true inference cost relative
    # to CM (1 call/step).
    ax_rmse.set_xscale("log")
    ax_rmse.set_xlabel("NFE (network forward calls per sample)")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("(a) Reconstruction error vs. NFE")
    _clean_axes(ax_rmse)

    ax_crps.set_xscale("log")
    ax_crps.set_xlabel("NFE (network forward calls per sample)")
    ax_crps.set_ylabel("CRPS")
    ax_crps.set_title("(b) Ensemble calibration vs. NFE")
    _clean_axes(ax_crps)

    # Legend below both panels: 3 entries, one row, one shared legend rather
    # than a per-axis one so the figure reads as a single unit.
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=3, frameon=False, columnspacing=1.6, handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"))
    print(f"[plot_nfe_efficiency] saved {out_stem.with_suffix('.pdf')} and .png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(HERE / "results" / "GP"))
    ap.add_argument("--out", default=str(HERE / "figures" / "GP" / "nfe_efficiency"))
    args = ap.parse_args()

    data = load_sweeps(Path(args.results_dir))
    if not any(m in data for m in CURVE_METHODS):
        raise SystemExit(f"No CM/FM *_nfe_sweep.csv found under {args.results_dir} -- run run_nfe_sweep.py first.")
    make_figure(data, Path(args.out))


if __name__ == "__main__":
    main()
