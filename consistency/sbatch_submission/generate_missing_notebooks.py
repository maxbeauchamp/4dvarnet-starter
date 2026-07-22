#!/usr/bin/env python3
"""Generate the missing notebooks needed to complete the 8 methods x 3 xps.

Principle (cf. `status: generated` in methods.yaml): for a missing
(method, xp) pair, assemble
    setup + imports          <- existing "template" notebook of the target xp
    data loading (dataloader) <- same
    model + training + sampling + metrics  <- GP reference notebook for this
        method (already xp-agnostic code, cf. exploration: the
        UNet/Lit*/consistency_models_* classes are imported as-is, only a few
        numeric values differ depending on the xp)

This script produces a FIRST, executable version, not a guarantee of
scientific correctness: a "⚠️ ACTION REQUIRED" cell is inserted in every
generated notebook listing the points to check manually before a production
run (cf. approved plan — sigma_max, cond_channels wiring, NaN masking for
xps with data gaps).

Usage:
    python generate_missing_notebooks.py                # generate all 10 missing combos
    python generate_missing_notebooks.py --method VarCM --xp SIC   # a single combo
    python generate_missing_notebooks.py --dry-run       # list without writing
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import yaml

import cell_library as cl

HERE = Path(__file__).resolve().parent
NOTEBOOKS_ROOT = HERE.parent / "Notebooks"
METHODS_YAML = HERE / "methods.yaml"

# xps for which we know a cond_channels multiplier different from the GP
# reference notebook (cf. direct inspection: CM_sic uses
# `UNetConfig(channels=C, cond_channels=2*C)` because of the extra `osisaf`
# conditioning field, GP/SSH_GF use the default `UNetConfig(channels=C)`).
# Applied only to the literal `UNetConfig(channels=C)` pattern — the config
# classes specific to VarCM/DynCM/VarDynCM/DynFM are NOT rewritten
# automatically (cf. "ACTION REQUIRED" cell inserted in the generated notebook).
COND_CHANNELS_MULTIPLIER = {"GP": 1, "SIC": 2, "SSH_GF": 1}

_UNETCONFIG_PLAIN_RE = re.compile(r"UNetConfig\(channels=C\)")
_LOGDIR_ASSIGN_RE = re.compile(r'^(\s*LOG_DIR\s*=\s*).*$', re.MULTILINE)


def load_methods():
    with open(METHODS_YAML) as f:
        return yaml.safe_load(f)


def notebook_path(cfg: dict, xp: str, filename: str) -> Path:
    return NOTEBOOKS_ROOT / cfg["xp_notebook_dir"][xp] / filename


def build_action_required_cell(method: str, xp: str, cond_channels_patched: bool, has_nan: bool, log_dir: str) -> "cl.nbformat.NotebookNode":
    checklist = [
        f"- **LOG_DIR** automatically set to `{log_dir}` — check it doesn't collide with an existing run.",
        "- **sigma_max / SIGMA_NOISE**: value carried over as-is from the GP source notebook — "
        "needs to be re-tuned to this variable's physical scale "
        "(cf. SIC ≈ 1.0, SSH_GF ≈ 10, GP to verify) before any production run.",
    ]
    if cond_channels_patched:
        checklist.append(
            "- **cond_channels**: `UNetConfig(channels=C)` was automatically rewritten to "
            f"`UNetConfig(channels=C, cond_channels={COND_CHANNELS_MULTIPLIER[xp]}*C)` (factor inferred from the "
            f"existing {xp}/CM notebook). Verify this factor is correct for the config class used by **{method}** "
            "(may differ from CM's) and that the forward pass actually concatenates the right conditioning fields."
        )
    else:
        checklist.append(
            "- **cond_channels / conditioning wiring**: NOT automatically verified for this method "
            f"(a config class other than `UNetConfig(channels=C)` was detected). Compare against the already "
            f"validated CM notebook for xp {xp} and adapt if this variable has extra conditioning fields."
        )
    if has_nan:
        checklist.append(
            "- **Masked loss (NaN)**: this xp has NaN gaps in the target. The matching FM notebook "
            "(`Notebook_flowmatching_model_FM_sic.ipynb`, class `LitFlowMatchingSIC.training_step`) already "
            "implements masking via `masked_average` — **this generated notebook did NOT reproduce it "
            "automatically** for this method's Lightning class. Must be done before any real training run, "
            "otherwise the model will train on NaN noise."
        )
    text = (
        "## ⚠️ ACTION REQUIRED — automatically generated notebook\n\n"
        f"This notebook (`{method}` / `{xp}`) was assembled by "
        "`sbatch_submission/generate_missing_notebooks.py`, combining the data loading of xp "
        f"`{xp}` with the method code of the GP reference notebook. Points to validate before a "
        "production run:\n\n"
        + "\n".join(checklist)
    )
    return cl.new_markdown_cell(text)


def generate_one(cfg: dict, method: str, xp: str, dry_run: bool = False) -> Path:
    method_cfg = cfg["methods"][method]
    xp_cfg = method_cfg["xp"][xp]
    assert xp_cfg["status"] == "generated", f"{method}/{xp} is not marked 'generated' in methods.yaml"

    gp_cfg = method_cfg["xp"]["GP"]
    assert gp_cfg["status"] == "existing", f"No GP reference notebook for {method}"
    source_path = notebook_path(cfg, "GP", gp_cfg["notebook"])

    template_filename = cfg["dataloader_template"][xp]
    template_path = notebook_path(cfg, xp, template_filename)

    source_nb = cl.load_notebook(source_path)
    template_nb = cl.load_notebook(template_path)

    source_split = cl.split_notebook(source_nb)
    template_split = cl.split_notebook(template_nb)

    method_imports = cl.extract_method_specific_imports(source_nb)
    new_setup = cl.merge_method_imports_into_setup(template_split["setup"], method_imports)
    new_data = copy.deepcopy(template_split["data"])
    new_method = copy.deepcopy(source_split["method"])

    # Explicit overrides (SOLVER_TYPE, COND_MODE, ...) declared in methods.yaml,
    # inserted at the very start of the method block to guarantee they take
    # precedence over any assignment already present further down in the
    # copied cells.
    params = method_cfg.get("params") or {}
    if params:
        lines = [f'{k} = {v!r}' for k, v in params.items()]
        override_cell = cl.new_code_cell(
            "# --- overrides enforced by methods.yaml (generate_missing_notebooks.py) ---\n" + "\n".join(lines)
        )
        new_method.insert(0, override_cell)

    # LOG_DIR: force the registry value instead of the one inherited from GP.
    log_dir = xp_cfg["log_dir"]
    cond_patched = False
    for cell in new_method:
        if cell.cell_type != "code":
            continue
        if _LOGDIR_ASSIGN_RE.search(cell.source):
            cell.source = _LOGDIR_ASSIGN_RE.sub(rf'\1"{log_dir}"', cell.source)
        mult = COND_CHANNELS_MULTIPLIER[xp]
        if mult != 1 and _UNETCONFIG_PLAIN_RE.search(cell.source):
            cell.source = _UNETCONFIG_PLAIN_RE.sub(f"UNetConfig(channels=C, cond_channels={mult}*C)", cell.source)
            cond_patched = True

    has_nan = cfg["xp_has_nan_target"][xp]
    action_cell = build_action_required_cell(method, xp, cond_patched, has_nan, log_dir)

    new_cells = list(new_setup) + [action_cell] + list(new_data) + list(new_method)
    new_nb = cl.assemble_notebook(new_cells, metadata=copy.deepcopy(template_nb.metadata))

    out_path = notebook_path(cfg, xp, xp_cfg["notebook"])
    print(f"{'[dry-run] ' if dry_run else ''}{method:14s} {xp:8s} -> {out_path.relative_to(NOTEBOOKS_ROOT.parent)}"
          f"  ({len(new_setup)} setup + 1 action + {len(new_data)} data + {len(new_method)} method cells)")
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cl.save_notebook(new_nb, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", help="Only generate this method")
    ap.add_argument("--xp", help="Only generate this xp")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_methods()
    targets = []
    for method, method_cfg in cfg["methods"].items():
        if args.method and method != args.method:
            continue
        for xp, xp_cfg in method_cfg["xp"].items():
            if args.xp and xp != args.xp:
                continue
            if xp_cfg["status"] == "generated":
                targets.append((method, xp))

    if not targets:
        print("Nothing to generate with these filters.")
        return

    for method, xp in targets:
        generate_one(cfg, method, xp, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
