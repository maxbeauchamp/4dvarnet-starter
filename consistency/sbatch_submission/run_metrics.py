#!/usr/bin/env python3
"""Run (via papermill) the metrics phase of one method for a given xp.

Reloads the checkpoint trained by run_training.py (RESET_TRAINING=False,
MAX_EPOCHS unchanged -> `trainer.fit` does nothing more on the training side)
and re-executes the notebook with N_SAMPLES=50 for the 6 generative methods
(CM, VarCM, DynCM, VarDynCM, FM, DynFM) or N_SAMPLES=1 for the 2 deterministic
methods (UNet, 4dvarnet_lstm) — cf. `generative` in methods.yaml.

Writes results/<xp>/<method>_metrics.csv, later consumed by make_latex_table.py.

Usage:
    python run_metrics.py --xp GP --method CM
    python run_metrics.py --xp GP --method CM --n-members 10   # one-off override
"""
from __future__ import annotations

import argparse
import sys
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
    ap.add_argument("--n-members", type=int, default=None, help="Override the number of members (default: methods.yaml n_samples_metrics)")
    ap.add_argument("--cuda-device", default="0")
    args = ap.parse_args()

    cfg = load_methods()
    if args.method not in cfg["methods"]:
        raise SystemExit(f"Unknown method: {args.method}. Choices: {list(cfg['methods'])}")
    method_cfg = cfg["methods"][args.method]
    xp_cfg = method_cfg["xp"][args.xp]

    notebook_in = NOTEBOOKS_ROOT / cfg["xp_notebook_dir"][args.xp] / xp_cfg["notebook"]
    if not notebook_in.exists():
        raise SystemExit(f"Notebook not found: {notebook_in}")

    out_dir = RESULTS_ROOT / args.xp
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_out = out_dir / f"{args.method}_metrics.ipynb"
    metrics_csv = out_dir / f"{args.method}_metrics.csv"

    n_members = args.n_members if args.n_members is not None else method_cfg["n_samples_metrics"]

    params = {
        "MAX_EPOCHS": cfg["max_epochs"],
        "N_SAMPLES": n_members,
        "RESET_TRAINING": False,
        "CUDA_VISIBLE_DEVICES": args.cuda_device,
        "METRICS_CSV": str(metrics_csv),
    }
    params.update(method_cfg.get("params") or {})

    print(f"\n{'='*88}\n[run_metrics] STARTING {args.method} / {args.xp}\n{'='*88}", flush=True)
    print(f"[run_metrics] {args.method} / {args.xp} -> {notebook_out} (n_members={n_members})", flush=True)
    print(f"[run_metrics] params = {params}", flush=True)
    # IMPORTANT: see the matching comment in run_training.py — force cwd to the
    # notebook's own directory so relative paths (and checkpoint lookup under
    # LOG_DIR) resolve exactly as they would for run_training.py or a manually
    # reopened notebook.
    # log_output=True: without it, papermill only captures each cell's stdout
    # into the OUTPUT .ipynb -- nothing is streamed live to this process's
    # stdout/stderr (i.e. nothing shows up in a redirected sbatch .out log)
    # until the whole notebook finishes. That made it impossible to tell, from
    # the .out file alone, whether a run was doing the expected fast metrics
    # pass or had silently fallen back to retraining from scratch (no matching
    # checkpoint found) for 2000 epochs. With log_output=True, every print()
    # inside the notebook (checkpoint-found/not-found messages, the
    # EpochHeartbeat "[progress] epoch X/Y" lines, ensemble-generation and
    # metrics-serialization prints) is now prefixed with the cell index and
    # streamed immediately.
    pm.execute_notebook(
        str(notebook_in),
        str(notebook_out),
        parameters=params,
        cwd=str(notebook_in.parent),
        log_output=True,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr,
    )
    print(f"[run_metrics] done: {metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
