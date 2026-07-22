#!/usr/bin/env python3
"""Build the LaTeX summary table of the 8 methods for a given xp, from the
CSV files produced by run_metrics.py (results/<xp>/<method>_metrics.csv).

Each CSV follows the schema common to all notebooks in the suite (a `Method`
column, one row for the method itself + one 'OI baseline' row — cf.
`_metrics_row()` in the notebooks): we isolate the method's own row (any
label not containing "OI"), and keep the "OI baseline" row only once (it is
identical — same reference OI — across the 8 files).

Usage:
    python make_latex_table.py --xp GP
    python make_latex_table.py --xp GP --out results/GP/comparison_table.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from tabulate import tabulate

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
METHODS_YAML = HERE / "methods.yaml"

METHOD_ORDER = ["UNet", "4dvarnet_lstm", "CM", "VarCM", "DynCM", "VarDynCM", "FM", "DynFM"]


def load_methods():
    with open(METHODS_YAML) as f:
        return yaml.safe_load(f)


def split_method_and_oi_row(df: pd.DataFrame) -> tuple[dict | None, dict | None]:
    if "Method" not in df.columns:
        return None, None
    is_oi = df["Method"].astype(str).str.contains("OI", case=False, na=False)
    method_rows = df[~is_oi]
    oi_rows = df[is_oi]
    method_row = method_rows.iloc[0].to_dict() if len(method_rows) else None
    oi_row = oi_rows.iloc[0].to_dict() if len(oi_rows) else None
    return method_row, oi_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xp", required=True, choices=["GP", "SIC", "SSH_GF"])
    ap.add_argument("--out", default=None, help="Output .tex path (default: results/<xp>/comparison_table.tex)")
    args = ap.parse_args()

    cfg = load_methods()
    out_dir = RESULTS_ROOT / args.xp
    out_path = Path(args.out) if args.out else out_dir / "comparison_table.tex"

    rows = []
    oi_row = None
    value_columns: list[str] = []
    missing = []

    for method in METHOD_ORDER:
        csv_path = out_dir / f"{method}_metrics.csv"
        if not csv_path.exists():
            missing.append(method)
            continue
        df = pd.read_csv(csv_path, dtype=str)
        method_row, this_oi_row = split_method_and_oi_row(df)
        if method_row is None:
            missing.append(method)
            continue
        for col in df.columns:
            if col != "Method" and col not in value_columns:
                value_columns.append(col)
        method_row["Method"] = method
        method_row["N members"] = cfg["methods"][method]["n_samples_metrics"]
        rows.append(method_row)
        if oi_row is None and this_oi_row is not None:
            oi_row = this_oi_row
            oi_row["Method"] = "OI baseline"
            oi_row["N members"] = "—"

    if missing:
        print(f"⚠️  Methods without a metrics CSV (skipped): {missing}")
    if not rows:
        raise SystemExit(f"No metrics found in {out_dir} — run run_metrics.py first.")

    if oi_row is not None:
        rows.append(oi_row)

    columns = ["Method"] + value_columns + ["N members"]
    table_df = pd.DataFrame(rows)[columns].fillna("—")

    latex = tabulate(table_df.values.tolist(), headers=list(table_df.columns), tablefmt="latex_booktabs")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex + "\n")
    print(f"LaTeX table written: {out_path}")
    print()
    print(latex)


if __name__ == "__main__":
    main()
