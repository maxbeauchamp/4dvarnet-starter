#!/usr/bin/env python3
"""Build the NFE-vs-RMSE / NFE-vs-wall-clock efficiency figure (GP), from the
CSVs produced by run_nfe_sweep.py -- substantiates the "efficient few-step
inference" claim: an NFE-vs-RMSE curve, the K used per method, and wall-clock
comparisons, in one publication-ready figure.

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

# Display order + style: consistent across the whole method suite (matches
# the paper's own ordering: deterministic baseline, then CM family, then FM
# family). Colors are colorblind-safe (Okabe-Ito), one marker shape per
# family so CM-family / FM-family / baseline are visually grouped even
# before reading the legend.
METHOD_STYLE = {
    "4dvarnet_lstm": dict(label="4DVarNet-LSTM", color="#000000", marker="*", ms=14, ls="none", zorder=5),
    "CM":            dict(label="CM",            color="#0072B2", marker="o", ms=6, ls="-"),
    "VarCM":         dict(label="VarCM",         color="#56B4E9", marker="o", ms=6, ls="-"),
    "DynCM":         dict(label="DynCM",         color="#009E73", marker="s", ms=6, ls="-"),
    "VarDynCM":      dict(label="VarDynCM",      color="#D55E00", marker="s", ms=6, ls="-"),
    "FM":            dict(label="FM",            color="#CC79A7", marker="^", ms=7, ls="-"),
    "DynFM":         dict(label="DynFM",         color="#E69F00", marker="^", ms=7, ls="-"),
}
METHOD_ORDER = ["4dvarnet_lstm", "CM", "VarCM", "DynCM", "VarDynCM", "FM", "DynFM"]


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
        df = pd.read_csv(f).sort_values("N")
        data[method] = df
    return data


def make_figure(data: dict[str, pd.DataFrame], out_stem: Path, metric: str = "rmse"):
    _iclr_style()
    fig, (ax_metric, ax_time) = plt.subplots(1, 2, figsize=(9.0, 3.15))

    handles, labels = [], []
    for method in METHOD_ORDER:
        if method not in data:
            continue
        df = data[method]
        style = METHOD_STYLE[method]
        is_point = len(df) == 1   # 4DVarNet: single fixed-K reference point

        (h,) = ax_metric.plot(
            df["N"], df[metric],
            marker=style["marker"], ms=style["ms"], color=style["color"],
            ls="none" if is_point else style["ls"],
            markeredgecolor=style.get("markeredgecolor", "white"),
            zorder=style.get("zorder", 3),
        )
        ax_time.plot(
            df["N"], df["wall_clock_s_per_sample"] * 1000.0,
            marker=style["marker"], ms=style["ms"], color=style["color"],
            ls="none" if is_point else style["ls"],
            markeredgecolor=style.get("markeredgecolor", "white"),
            zorder=style.get("zorder", 3),
        )
        handles.append(h)
        labels.append(style["label"])

    ax_metric.set_xscale("log")
    ax_metric.set_xlabel("NFE (solver / sampling steps $N$)")
    ax_metric.set_ylabel("RMSE" if metric == "rmse" else "CRPS")
    ax_metric.set_title("(a) Reconstruction error vs. NFE")
    _clean_axes(ax_metric)

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("NFE (solver / sampling steps $N$)")
    ax_time.set_ylabel("Wall-clock [ms / sample]")
    ax_time.set_title("(b) Inference cost vs. NFE")
    _clean_axes(ax_time)

    # Legend below both panels, 3 entries per row (ncol=3), one shared legend
    # rather than a per-axis one so the figure reads as a single unit.
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=3, frameon=False, columnspacing=1.6, handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.24, 1, 1))

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"))
    print(f"[plot_nfe_efficiency] saved {out_stem.with_suffix('.pdf')} and .png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(HERE / "results" / "GP"))
    ap.add_argument("--out", default=str(HERE / "figures" / "GP" / "nfe_efficiency"))
    ap.add_argument("--metric", default="rmse", choices=["rmse", "crps"])
    args = ap.parse_args()

    data = load_sweeps(Path(args.results_dir))
    if not data:
        raise SystemExit(f"No *_nfe_sweep.csv found under {args.results_dir} -- run run_nfe_sweep.py first.")
    make_figure(data, Path(args.out), metric=args.metric)


if __name__ == "__main__":
    main()
