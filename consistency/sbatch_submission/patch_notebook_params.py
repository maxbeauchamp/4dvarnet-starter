#!/usr/bin/env python3
"""Make a notebook `papermill`-parameterizable and serialize its metrics.

Injects, as the very first cell of the notebook (even before `!nvidia-smi` /
`%env`, since some notebooks reference CUDA_VISIBLE_DEVICES in their first
few cells):
    - a cell tagged "parameters" (papermill convention) defining
      MAX_EPOCHS, N_SAMPLES, RESET_TRAINING, CUDA_VISIBLE_DEVICES, METRICS_CSV

Then neutralizes, in the rest of the notebook, values that would otherwise
duplicate these parameters (they would silently overwrite the value injected
by papermill):
    - `max_epochs=<int>` (inside a Trainer(...) call)   -> `max_epochs=MAX_EPOCHS`
    - `%env CUDA_VISIBLE_DEVICES=<int>`                  -> equivalent os.environ[...]
    - `N_SAMPLES = <int>` (standalone line)              -> commented out
    - `RESET_TRAINING = True|False` (standalone line)    -> commented out

Also inserts a small `EpochHeartbeat` Lightning callback (right after the
parameters cell) and wires it into the notebook's `Trainer(...)` /
`trainer.fit(...)` calls, so that sbatch logs show one clear progress line
every N epochs (e.g. `[progress] epoch 140/2000 | train_loss=0.0231 | ...`)
instead of either nothing (default `log_output=False` in papermill) or a
noisy, carriage-return-based tqdm bar that doesn't render well in a plain log
file. The default tqdm progress bar is disabled (`enable_progress_bar=False`)
to avoid that noise once `log_output=True` is used by the runner scripts.

Finally appends a last cell that serializes `df_metrics` (variable name
already used consistently across all notebooks in the suite) to `METRICS_CSV`.

Idempotent: if a cell tagged "parameters" already exists (comment marker),
the notebook is not re-patched.

Usage:
    python patch_notebook_params.py --all
    python patch_notebook_params.py Notebooks/Notebooks_GP/Notebook_..._spde.ipynb
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

import cell_library as cl

HERE = Path(__file__).resolve().parent
NOTEBOOKS_ROOT = HERE.parent / "Notebooks"
METHODS_YAML = HERE / "methods.yaml"

PARAMS_MARKER = "# --- parameters (patch_notebook_params.py) ---"
HEARTBEAT_MARKER = "# --- epoch heartbeat (patch_notebook_params.py) ---"

_MAX_EPOCHS_RE = re.compile(r"max_epochs\s*=\s*\d+")
_ENV_CUDA_RE = re.compile(r"^%env\s+CUDA_VISIBLE_DEVICES\s*=\s*\d+\s*$", re.MULTILINE)
_N_SAMPLES_RE = re.compile(r"^(\s*)N_SAMPLES\s*=\s*\d+\s*(#.*)?$", re.MULTILINE)
_RESET_TRAINING_RE = re.compile(r"^(\s*)RESET_TRAINING\s*=\s*(True|False)\s*(#.*)?$", re.MULTILINE)
# Matches `Trainer(` / `pl.Trainer(`, not e.g. `MyTrainer(` (negative lookbehind
# on the char right before an optional `pl.` prefix).
_TRAINER_CTOR_RE = re.compile(r"(?<![A-Za-z0-9_.])(pl\.)?Trainer\(")
# Matches `trainer.fit(`, `config.trainer.fit(`, etc. — captures the exact
# expression so the heartbeat is attached to the same object before .fit().
_TRAINER_FIT_RE = re.compile(r"^([ \t]*)((?:[A-Za-z_][A-Za-z0-9_]*\.)*trainer)\.fit\(", re.MULTILINE)


def build_parameters_cell() -> "cl.nbformat.NotebookNode":
    cell = cl.new_code_cell(
        PARAMS_MARKER
        + "\n"
        + 'MAX_EPOCHS = 2000\n'
        + 'N_SAMPLES = 10           # run_metrics.py forces 50 for generative methods\n'
        + 'RESET_TRAINING = False\n'
        + 'CUDA_VISIBLE_DEVICES = "0"\n'
        + 'METRICS_CSV = "results/metrics.csv"\n'
    )
    cell.metadata["tags"] = ["parameters"]
    return cell


def build_heartbeat_cell() -> "cl.nbformat.NotebookNode":
    return cl.new_code_cell(
        HEARTBEAT_MARKER
        + "\n"
        + "import pytorch_lightning as _pl\n"
        + "\n"
        + "class EpochHeartbeat(_pl.Callback):\n"
        + '    """Prints one clear progress line every `every_n_epochs` epochs, so\n'
        + "    sbatch logs show training progress without the noise of a per-batch\n"
        + "    tqdm progress bar (which doesn't render well once redirected to a\n"
        + "    plain log file)."
        + '    """\n'
        + "\n"
        + "    def __init__(self, every_n_epochs: int = 1):\n"
        + "        self.every_n_epochs = every_n_epochs\n"
        + "\n"
        + "    def on_train_epoch_end(self, trainer, pl_module):\n"
        + "        epoch = trainer.current_epoch + 1\n"
        + "        if epoch % self.every_n_epochs != 0 and epoch != trainer.max_epochs:\n"
        + "            return\n"
        + "        parts = []\n"
        + "        for k, v in sorted(trainer.callback_metrics.items()):\n"
        + "            try:\n"
        + "                parts.append(f'{k}={float(v):.4f}')\n"
        + "            except (TypeError, ValueError):\n"
        + "                pass\n"
        + "        print(f'[progress] epoch {epoch}/{trainer.max_epochs} | ' + ' | '.join(parts), flush=True)\n"
    )


def build_serialization_cell() -> "cl.nbformat.NotebookNode":
    return cl.new_code_cell(
        "# --- metrics serialization (patch_notebook_params.py) ---\n"
        "import os\n"
        "os.makedirs(os.path.dirname(METRICS_CSV) or '.', exist_ok=True)\n"
        "_df_out = df_metrics.reset_index() if df_metrics.index.name == 'Method' else df_metrics\n"
        "_df_out.to_csv(METRICS_CSV, index=False)\n"
        "print(f'Metrics written to {METRICS_CSV}')\n"
    )


def already_patched(nb) -> bool:
    return any(c.cell_type == "code" and PARAMS_MARKER in c.source for c in nb.cells)


def ensure_heartbeat_cell(nb) -> None:
    """Insert build_heartbeat_cell() right after the parameters cell, unless
    already present (idempotent)."""
    if any(c.cell_type == "code" and HEARTBEAT_MARKER in c.source for c in nb.cells):
        return
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and PARAMS_MARKER in cell.source:
            nb.cells.insert(i + 1, build_heartbeat_cell())
            return
    raise ValueError("Parameters cell not found — run patch_notebook() first")


def apply_substitutions(nb) -> None:
    """Neutralize, in every cell EXCEPT the parameters/heartbeat cells
    themselves, the literals that would otherwise duplicate the papermill
    parameters, and wire the EpochHeartbeat callback into the Trainer(...) /
    trainer.fit(...) calls. Called both on first patch and by --reapply
    (idempotent: an already-substituted line no longer matches the regexes)."""
    for cell in nb.cells:
        if cell.cell_type != "code" or PARAMS_MARKER in cell.source or HEARTBEAT_MARKER in cell.source:
            continue
        src = cell.source
        src = _MAX_EPOCHS_RE.sub("max_epochs=MAX_EPOCHS", src)
        src = _ENV_CUDA_RE.sub(
            'import os; os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES', src
        )
        src = _N_SAMPLES_RE.sub(r"\1# N_SAMPLES set by the parameters cell above", src)
        src = _RESET_TRAINING_RE.sub(r"\1# RESET_TRAINING set by the parameters cell above", src)
        if "enable_progress_bar=False" not in src:
            src = _TRAINER_CTOR_RE.sub(r"\1Trainer(enable_progress_bar=False, ", src)

        def _attach_heartbeat(m: re.Match) -> str:
            indent, trainer_expr = m.group(1), m.group(2)
            attach_line = f"{indent}{trainer_expr}.callbacks.append(EpochHeartbeat(every_n_epochs=1))\n"
            # Printed right before trainer.fit(): with log_output=True (now set
            # in run_training.py/run_metrics.py), this is the first thing that
            # shows up in a redirected sbatch .out log for this cell, making it
            # immediately clear whether the run is about to do a full 2000-epoch
            # training pass (resume_ckpt=None -> starting from scratch, or a
            # stale/mismatched checkpoint) or a fast resume/no-op (checkpoint
            # already at MAX_EPOCHS) -- previously indistinguishable from the
            # outside until the whole notebook finished.
            diag_line = (
                f"{indent}print(f'[TRAINING] resume_ckpt={{resume_ckpt!r}} | "
                f"MAX_EPOCHS={{MAX_EPOCHS}}', flush=True)\n"
            )
            prefix = ""
            if attach_line not in src:
                prefix += attach_line
            if diag_line not in src:
                prefix += diag_line
            return prefix + m.group(0)

        src = _TRAINER_FIT_RE.sub(_attach_heartbeat, src)
        cell.source = src


def patch_notebook(path: Path, dry_run: bool = False) -> bool:
    nb = cl.load_notebook(path)
    if already_patched(nb):
        print(f"skip (already patched): {path}")
        return False

    # Inserted as the very first cell: some notebooks reference
    # CUDA_VISIBLE_DEVICES as early as their 2nd/3rd cell
    # (`%env CUDA_VISIBLE_DEVICES=...`), well before the big imports block —
    # the parameters cell must therefore precede everything, not just imports.
    nb.cells.insert(0, build_parameters_cell())
    ensure_heartbeat_cell(nb)

    apply_substitutions(nb)
    nb.cells.append(build_serialization_cell())

    try:
        display_path = path.relative_to(NOTEBOOKS_ROOT.parent)
    except ValueError:
        display_path = path
    print(f"{'[dry-run] ' if dry_run else ''}patch: {display_path}")
    if not dry_run:
        cl.save_notebook(nb, path)
    return True


def reapply_notebook(path: Path, dry_run: bool = False) -> bool:
    """Re-apply apply_substitutions() on an already-patched notebook (useful
    after fixing the substitution regexes, without duplicating the parameters
    cell)."""
    nb = cl.load_notebook(path)
    if not already_patched(nb):
        print(f"not patched yet, skipped (run without --reapply first): {path}")
        return False
    ensure_heartbeat_cell(nb)
    apply_substitutions(nb)
    print(f"{'[dry-run] ' if dry_run else ''}reapply: {path}")
    if not dry_run:
        cl.save_notebook(nb, path)
    return True


def iter_all_notebook_paths():
    with open(METHODS_YAML) as f:
        cfg = yaml.safe_load(f)
    seen = set()
    for method_cfg in cfg["methods"].values():
        for xp, xp_cfg in method_cfg["xp"].items():
            p = NOTEBOOKS_ROOT / cfg["xp_notebook_dir"][xp] / xp_cfg["notebook"]
            if p not in seen:
                seen.add(p)
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Specific notebooks to patch")
    ap.add_argument("--all", action="store_true", help="Patch the 24 notebooks referenced in methods.yaml")
    ap.add_argument("--reapply", action="store_true", help="Re-apply substitutions on already-patched notebooks (without re-inserting the parameters cell)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        paths = list(iter_all_notebook_paths())
    else:
        paths = [Path(p) for p in args.paths]
        if not paths:
            ap.error("Provide notebook paths or --all")

    for p in paths:
        if not p.exists():
            print(f"MISSING (not generated yet?): {p}")
            continue
        if args.reapply:
            reapply_notebook(p, dry_run=args.dry_run)
        else:
            patch_notebook(p, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
