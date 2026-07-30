#!/usr/bin/env python3
"""Fix the `time` coordinate of CIMR NetCDF files, which was written with a
single constant (wrong) timestamp on every file regardless of the actual
date encoded in the filename (e.g. `CIMR5km_2022-02-28_mod.nc` all carrying
`2019-09-21T23:23:44`).

The correct date is derived from the filename (`CIMR5km_<YYYY-MM-DD>...nc`)
and written at midnight (00:00:00) — there is no way to recover the true
time-of-day since the original bug wrote the exact same timestamp everywhere.

By default this is NON-DESTRUCTIVE: fixed copies are written to --out-dir,
originals are left untouched. Use --in-place only once you've verified the
output on a few files (originals are overwritten with no backup).

Usage:
    # Preview what would change, without writing anything
    python fix_cimr_time.py --cimr-dir /Odyssey/public/CROSCIM_dataset/data_noise --dry-run

    # Write corrected copies to a new directory (safe default)
    python fix_cimr_time.py --cimr-dir /Odyssey/public/CROSCIM_dataset/data_noise \
        --out-dir /Odyssey/public/CROSCIM_dataset/data_noise_fixed

    # Overwrite files in place (only after verifying the copies above)
    python fix_cimr_time.py --cimr-dir /Odyssey/public/CROSCIM_dataset/data_noise --in-place
"""
from __future__ import annotations

import argparse
import re
import shutil
from glob import glob
from pathlib import Path

import pandas as pd
import xarray as xr

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def correct_date_from_filename(path: Path) -> pd.Timestamp:
    m = DATE_RE.search(path.name)
    if not m:
        raise ValueError(f"No YYYY-MM-DD date found in filename: {path.name}")
    return pd.Timestamp(m.group(1))


def fix_file(path: Path, out_path: Path, dry_run: bool) -> str:
    correct_date = correct_date_from_filename(path)

    with xr.open_dataset(path) as ds:
        if "time" not in ds.coords and "time" not in ds.dims:
            return f"SKIP (no time coord): {path.name}"

        current = pd.Timestamp(ds["time"].values[0]) if ds["time"].ndim else pd.Timestamp(ds["time"].values)
        if current.normalize() == correct_date.normalize():
            return f"OK already correct: {path.name} (time={current})"

        if dry_run:
            return f"WOULD FIX: {path.name}  {current} -> {correct_date}"

        n_time = ds.sizes.get("time", 1)
        fixed = ds.assign_coords(time=("time", [correct_date] * n_time) if "time" in ds.dims else [correct_date])
        fixed["time"].encoding = {}  # drop stale units/dtype encoding from the source file

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        fixed.to_netcdf(tmp_path)
        fixed.close()

    # Only replace the destination after the write fully succeeded.
    tmp_path.replace(out_path)
    return f"FIXED: {path.name}  {current} -> {correct_date}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cimr-dir", required=True, help="Directory containing CIMR5km_*.nc files")
    ap.add_argument("--out-dir", default=None, help="Where to write corrected copies (ignored with --in-place)")
    ap.add_argument("--in-place", action="store_true", help="Overwrite original files instead of writing to --out-dir (no backup — verify with --dry-run / --out-dir first)")
    ap.add_argument("--dry-run", action="store_true", help="Only report what would change, write nothing")
    args = ap.parse_args()

    if not args.in_place and not args.dry_run and not args.out_dir:
        ap.error("Provide --out-dir, or pass --in-place / --dry-run explicitly")

    cimr_dir = Path(args.cimr_dir)
    files = sorted(Path(p) for p in glob(str(cimr_dir / "CIMR5km_*.nc")))
    print(f"Found {len(files)} CIMR files in {cimr_dir}")

    counts = {"FIXED": 0, "WOULD FIX": 0, "OK already correct": 0, "SKIP": 0, "ERROR": 0}
    for f in files:
        if args.in_place:
            out_path = f
        elif args.out_dir:
            out_path = Path(args.out_dir) / f.name
        else:
            out_path = f  # dry-run with neither flag: never actually written to
        try:
            result = fix_file(f, out_path, dry_run=args.dry_run)
        except Exception as e:
            result = f"ERROR: {f.name}: {e}"
        print(result)
        for key in counts:
            if result.startswith(key):
                counts[key] += 1
                break

    print("\nSummary:", counts)


if __name__ == "__main__":
    main()
