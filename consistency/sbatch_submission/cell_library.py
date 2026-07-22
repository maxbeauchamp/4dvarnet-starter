"""Notebook splitting/assembly utilities for the 8 methods x 3 xps.

Relies on a fact verified across all existing GP/SIC/SSH_GF notebooks: every
notebook follows the same skeleton (imports -> data loading -> model
definition -> training -> sampling/evaluation -> metrics), but with
heterogeneous markdown header wording (emoji or not, "DataModule" vs
"Implementation", etc.). Section boundaries are therefore located by cell
CONTENT rather than by position or exact header text:

- end of the "imports" block : first code cell containing `sys.path.append`
- start of the "model" block   : first markdown header containing "unet" (case-insensitive)

Everything before that boundary is considered "xp-specific" (setup + data
loading) ; everything after is considered "method-specific" (model
definition, training, sampling, metrics) and xp-agnostic (the only
differences are numeric values: channels, cond_channels, sigma_max,
LOG_DIR, ...).
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

NOTEBOOK_VERSION = 4


def load_notebook(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(str(path), as_version=NOTEBOOK_VERSION)


def save_notebook(nb: nbformat.NotebookNode, path: Path) -> None:
    nbformat.write(nb, str(path))


def find_import_cell_index(nb: nbformat.NotebookNode) -> int:
    """First code cell containing `sys.path.append` (the big imports block at the
    top of the notebook). Some notebooks re-use `sys.path.append` further down
    (e.g. the Metrics section, to import `spectral_utils`) — we only want the
    very first occurrence, which corresponds to the main imports block."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and "sys.path.append" in cell.source:
            return i
    raise ValueError("No imports cell found (sys.path.append)")


_HEADER_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def find_first_header_containing(nb: nbformat.NotebookNode, keyword: str, start: int = 0, min_level: int = 3) -> int:
    """Find the first markdown header of level >= `min_level` (### = H3, etc.)
    whose HEADER LINE (first non-empty line of the cell) contains `keyword`.

    Deliberately restricted to the header line (not the body text) because
    some markdown cells mention the word in their prose without being the
    section we're looking for (e.g. a "## Solver Choice" cell that explains
    in its text how to choose between a "unet" and "lstm" solver — this is
    not the "UNet Building Blocks" section).
    """
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    for i in range(start, len(nb.cells)):
        cell = nb.cells[i]
        if cell.cell_type != "markdown":
            continue
        first_line = next((l for l in cell.source.splitlines() if l.strip()), "")
        m = _HEADER_LINE_RE.match(first_line.strip())
        if not m or len(m.group(1)) < min_level:
            continue
        if pattern.search(m.group(2)):
            return i
    raise ValueError(f"No header (level >= {min_level}) containing {keyword!r} found")


def split_notebook(nb: nbformat.NotebookNode) -> dict:
    """Split a notebook into 3 cell blocks: setup, data, method.

    setup  : cells up to and including the big imports block (nvidia-smi, %env, imports)
    data   : cells between the imports and the first "UNet" header (data loading)
    method : cells from the first "UNet" header to the end (model, training,
             sampling, metrics)
    """
    import_idx = find_import_cell_index(nb)
    unet_idx = find_first_header_containing(nb, "unet", start=import_idx + 1)
    return {
        "setup": nb.cells[: import_idx + 1],
        "data": nb.cells[import_idx + 1 : unet_idx],
        "method": nb.cells[unet_idx:],
        "import_idx": import_idx,
        "unet_idx": unet_idx,
    }


_IMPORT_LINE_RE = re.compile(
    r"^(from (?:consistency_models\.\S+|flowmatching_models_FM) import \([^)]*\)"
    r"|from (?:consistency_models\.\S+|flowmatching_models_FM) import .+"
    r"|import flowmatching_models_FM.*)$",
    re.MULTILINE,
)


def extract_method_specific_imports(nb: nbformat.NotebookNode) -> list[str]:
    """Retrieve the import lines specific to the method (consistency_models_X /
    flowmatching_models_FM) from the big imports block of a GP source notebook."""
    import_idx = find_import_cell_index(nb)
    src = nb.cells[import_idx].source
    return [m.group(0) for m in _IMPORT_LINE_RE.finditer(src)]


def merge_method_imports_into_setup(setup_cells: list, method_import_lines: list[str]) -> list:
    """Append the method-specific imports (consistency_models_X, ...) to the end
    of the target `setup` block's big imports cell, without duplicating lines
    already present."""
    if not method_import_lines:
        return setup_cells
    setup_cells = list(setup_cells)
    # The imports block is the last code cell of setup (cf. find_import_cell_index).
    for i in range(len(setup_cells) - 1, -1, -1):
        if setup_cells[i].cell_type == "code" and "sys.path.append" in setup_cells[i].source:
            cell = setup_cells[i]
            existing = cell.source
            new_lines = [l for l in method_import_lines if l not in existing]
            if new_lines:
                cell.source = existing.rstrip("\n") + "\n\n# --- method-specific imports (added by generate_missing_notebooks.py) ---\n" + "\n".join(new_lines) + "\n"
            return setup_cells
    raise ValueError("Imports block not found in setup_cells")


def new_markdown_cell(text: str) -> nbformat.NotebookNode:
    from nbformat.v4 import new_markdown_cell as _nmc
    return _nmc(text)


def new_code_cell(text: str) -> nbformat.NotebookNode:
    from nbformat.v4 import new_code_cell as _ncc
    return _ncc(text)


def assemble_notebook(cells: list, metadata: dict) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = metadata
    return nb
