#!/usr/bin/env python3
"""Run the NFE (number of function evaluations) efficiency sweep for one
method on one xp (GP or SSH_GF) -- substantiates the "efficient few-step
inference" claim with an NFE-vs-RMSE/CRPS curve, instead of relying only on
the qualitative convergence picture.

Reloads the checkpoint trained by run_training.py (SKIP_TRAINING=True, same
mechanism as run_metrics.py) and re-runs inference at several step counts
(NFE_GRID) on a small subset of the test set (NFE_N_TEST_BATCHES batches,
NFE_N_SAMPLES_SWEEP ensemble members per item) -- no retraining involved,
since nsteps/n_steps only control the schedule discretisation at inference
time for every generative method here (see the "NFE (few-step inference)
efficiency sweep" cell added to each instrumented notebook, right after its
existing full-test-set metrics cell -- see INSTRUMENTED_XP_METHODS below for
which (xp, method) combos actually have that cell).

For the deterministic 4dvarnet_lstm baseline, the number of solver
iterations is fixed by training (n_solver, unrolled at train time) and
cannot be swept -- its notebook cell instead reports a single reference
point (K, RMSE, wall-clock) on the same subset. On SSH_GF specifically, that
notebook has no SOLVER_TYPE switch (always the UNet gradient model, no LSTM
variant) -- see the caveat printed by that cell.

Writes results/<xp>/<method>_nfe_sweep.csv, later consumed by
plot_nfe_efficiency.py to build the final NFE-vs-RMSE / NFE-vs-CRPS figure.

Usage:
    python run_nfe_sweep.py --method CM
    python run_nfe_sweep.py --xp SSH_GF --method FM
    python run_nfe_sweep.py --xp SSH_GF --method 4dvarnet_lstm
    python run_nfe_sweep.py --method FM --n-test-batches 5 --n-samples-sweep 15
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

# NFE grids per method, chosen to bracket each method's production step
# count on both sides (see the "parameters" cell of each GP notebook for the
# same defaults -- kept here too so a sweep can be re-run with a different
# grid without touching the notebooks).
NFE_GRIDS = {
    "CM":            [1, 2, 3, 5, 8, 12, 15, 20, 30],
    "VarCM":         [1, 2, 3, 5, 8, 12, 15, 20, 30],
    "DynCM":         [2, 3, 4, 5, 6, 8, 10, 15, 20],
    "VarDynCM":      [2, 3, 4, 5, 6, 8, 10, 15, 20],
    "FM":            [1, 2, 4, 6, 10, 15, 20, 30, 50],
    "DynFM":         [5, 8, 12, 16, 21, 30, 42],
    "4dvarnet_lstm": [15],   # fixed by training (n_solver) -- single reference point
}


def load_methods():
    with open(METHODS_YAML) as f:
        return yaml.safe_load(f)


# Which (xp, method) combos actually have the "NFE (few-step inference)
# efficiency sweep" cell inserted into their notebook (see the GP/SSH_GF
# notebooks' sweep cells). Picking an un-instrumented combo would silently
# produce no CSV (the notebook has nothing gated on RUN_NFE_SWEEP to run),
# so this is checked explicitly instead of failing downstream in
# plot_nfe_efficiency.py with a cryptic "missing file" warning.
INSTRUMENTED_XP_METHODS = {
    "GP":     {"CM", "VarCM", "DynCM", "VarDynCM", "FM", "DynFM", "4dvarnet_lstm"},
    "SSH_GF": {"CM", "FM", "4dvarnet_lstm"},
    "SIC":    set(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xp", default="GP", choices=["GP", "SSH_GF", "SIC"])
    ap.add_argument("--method", required=True, choices=list(NFE_GRIDS))
    ap.add_argument("--n-test-batches", type=int, default=3,
                     help="Number of test_dataloader() batches used for the sweep (small subset -- this is a controlled ablation over NFE, not the headline metric)")
    ap.add_argument("--n-samples-sweep", type=int, default=20,
                     help="Ensemble members per test item at each NFE (needed for a stable CRPS estimate; smaller than the full N_SAMPLES=50 to keep the sweep affordable)")
    ap.add_argument("--grid", default=None,
                     help="Comma-separated override of the NFE grid, e.g. '1,2,4,8,16'")
    ap.add_argument("--cuda-device", default="0")
    args = ap.parse_args()

    if args.method not in INSTRUMENTED_XP_METHODS.get(args.xp, set()):
        raise SystemExit(
            f"No NFE sweep cell for method={args.method!r} on xp={args.xp!r}. "
            f"Instrumented methods for {args.xp}: {sorted(INSTRUMENTED_XP_METHODS.get(args.xp, set())) or 'none'}."
        )

    cfg = load_methods()
    method_cfg = cfg["methods"][args.method]
    xp_cfg = method_cfg["xp"][args.xp]

    notebook_in = NOTEBOOKS_ROOT / cfg["xp_notebook_dir"][args.xp] / xp_cfg["notebook"]
    if not notebook_in.exists():
        raise SystemExit(f"Notebook not found: {notebook_in}")

    out_dir = RESULTS_ROOT / args.xp
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_out = out_dir / f"{args.method}_nfe_sweep.ipynb"
    sweep_csv = out_dir / f"{args.method}_nfe_sweep.csv"

    grid = [int(x) for x in args.grid.split(",")] if args.grid else NFE_GRIDS[args.method]

    params = {
        "MAX_EPOCHS": cfg["max_epochs"],
        # The notebook's pre-existing full-test-set metrics cell -- AND the
        # illustrative "Publication Figures" cell before it, which hardcodes
        # ensemble[0]/ensemble[1] for a 2-member side-by-side comparison --
        # are NOT gated by RUN_NFE_SWEEP, so papermill still runs them.
        # N_SAMPLES=1 crashes the figures cell with IndexError on
        # ensemble[1] (only one member generated); 2 is the minimum that
        # keeps both pre-existing cells cheap without crashing. The sweep
        # itself uses its own NFE_N_SAMPLES_SWEEP, independently.
        "N_SAMPLES": 2,
        "RESET_TRAINING": False,
        "CUDA_VISIBLE_DEVICES": args.cuda_device,
        "METRICS_CSV": str(out_dir / f"{args.method}_metrics_nfe_placeholder.csv"),
        "SKIP_TRAINING": True,
        "RUN_NFE_SWEEP": True,
        "NFE_GRID": grid,
        "NFE_N_TEST_BATCHES": args.n_test_batches,
        "NFE_N_SAMPLES_SWEEP": args.n_samples_sweep,
        "NFE_SWEEP_CSV": str(sweep_csv),
    }
    params.update(method_cfg.get("params") or {})

    print(f"\n{'='*88}\n[run_nfe_sweep] STARTING {args.method} / {args.xp}\n{'='*88}", flush=True)
    print(f"[run_nfe_sweep] {args.method} -> {notebook_out}", flush=True)
    print(f"[run_nfe_sweep] grid={grid}  n_test_batches={args.n_test_batches}  n_samples_sweep={args.n_samples_sweep}", flush=True)
    pm.execute_notebook(
        str(notebook_in),
        str(notebook_out),
        parameters=params,
        cwd=str(notebook_in.parent),
        log_output=True,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr,
    )
    print(f"[run_nfe_sweep] done: {sweep_csv}", flush=True)


if __name__ == "__main__":
    main()
