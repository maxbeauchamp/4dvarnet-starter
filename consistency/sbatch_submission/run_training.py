#!/usr/bin/env python3
"""Run (via papermill) the training of one method for a given xp.

Training phase: MAX_EPOCHS=2000 (enforced regardless of the call), N_SAMPLES
reduced to 1 (instead of 50) to avoid paying the cost of the full stochastic
sampling here — cf. user decision: the 50-member metrics are computed
separately by run_metrics.py, which reloads the checkpoint (RESET_TRAINING=
False, MAX_EPOCHS unchanged -> `trainer.fit` does nothing more on the
training side, only the sampling/metrics part is actually recomputed).

Usage:
    python run_training.py --xp GP --method CM
    python run_training.py --xp SIC --method VarCM --reset   # start from scratch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import papermill as pm
import yaml

HERE = Path(__file__).resolve().parent
NOTEBOOKS_ROOT = HERE.parent / "Notebooks"
RESULTS_ROOT = HERE / "results"
METHODS_YAML = HERE / "methods.yaml"


def load_methods():
    with open(METHODS_YAML) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xp", required=True, choices=["GP", "SIC", "SSH_GF"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--reset", action="store_true", help="RESET_TRAINING=True (start from scratch, overwrites the existing checkpoint)")
    ap.add_argument("--cuda-device", default="0")
    args = ap.parse_args()

    cfg = load_methods()
    if args.method not in cfg["methods"]:
        raise SystemExit(f"Unknown method: {args.method}. Choices: {list(cfg['methods'])}")
    method_cfg = cfg["methods"][args.method]
    xp_cfg = method_cfg["xp"][args.xp]

    notebook_in = NOTEBOOKS_ROOT / cfg["xp_notebook_dir"][args.xp] / xp_cfg["notebook"]
    if not notebook_in.exists():
        raise SystemExit(
            f"Notebook not found: {notebook_in}\n"
            f"(status declared in methods.yaml: {xp_cfg['status']} — "
            f"if 'generated', run generate_missing_notebooks.py first)"
        )

    out_dir = RESULTS_ROOT / args.xp
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_out = out_dir / f"{args.method}_train.ipynb"
    metrics_csv = out_dir / f"{args.method}_metrics_train.csv"

    params = {
        "MAX_EPOCHS": cfg["max_epochs"],
        "N_SAMPLES": 1,
        "RESET_TRAINING": bool(args.reset),
        "CUDA_VISIBLE_DEVICES": args.cuda_device,
        "METRICS_CSV": str(metrics_csv),
    }
    params.update(method_cfg.get("params") or {})

    print(f"[run_training] {args.method} / {args.xp} -> {notebook_out}")
    print(f"[run_training] params = {params}")
    # IMPORTANT: papermill's default working directory is the CALLER's cwd, not
    # the notebook's own directory. All relative paths inside the notebooks
    # (LOG_DIR, SPDE_PATH, sys.path.append('../..'), ...) are written assuming
    # the notebook is opened from its own folder in Jupyter — so we force cwd
    # here to match that, otherwise checkpoints/logs would land under
    # sbatch_submission/ instead of Notebooks/Notebooks_<XP>/, and a manually
    # reopened notebook would not find them.
    pm.execute_notebook(
        str(notebook_in),
        str(notebook_out),
        parameters=params,
        cwd=str(notebook_in.parent),
    )
    print(f"[run_training] done: {notebook_out}")


if __name__ == "__main__":
    main()
