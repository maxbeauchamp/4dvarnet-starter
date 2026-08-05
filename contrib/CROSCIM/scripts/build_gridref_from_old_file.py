#!/usr/bin/env python3
"""One-off validation helper: build gridref_x50.nc/gridref_x10.nc from the
xc/yc/lon/lat of OLD NetCDF files (e.g. produced by the pre-refactor
pipeline), so the raw dataloader can be pointed at THIS exact historical
grid via the `gridref_dir` override, as a controlled test of whether a grid
offset vs the current gridref_x{50,10}.nc explains degraded predictions.

Pass the x50 file; the matching x10 file is auto-derived by replacing "x50"
with "x10" in the filename (same naming convention, only the suffix
changes). If that sibling file isn't found, gridref_x10.nc is copied
unchanged from the current gridref dir instead (test then only isolates
x50).

Usage:
    python build_gridref_from_old_file.py /path/to/old_file_x50.nc /path/to/out_dir
"""
import os
import shutil
import sys
from pathlib import Path

import xarray as xr

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from contrib.CROSCIM.dataloaders.load_data import gridref_path


def build_from_old_file(old_path, out_path, level):
    old = xr.open_dataset(old_path)
    for coord in ("xc", "yc", "lon", "lat"):
        if coord not in old.coords:
            raise ValueError(f"{old_path} has no '{coord}' coordinate — cannot build gridref_x{level} from it")

    grid = xr.Dataset(coords={
        "xc": old["xc"],
        "yc": old["yc"],
        "lon": old["lon"],
        "lat": old["lat"],
    })
    grid.to_netcdf(out_path)
    print(f"Wrote {out_path}  (xc={grid.sizes['xc']}, yc={grid.sizes['yc']})  from {old_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python build_gridref_from_old_file.py /path/to/old_file_x50.nc /path/to/out_dir")
        sys.exit(1)

    old_x50_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    build_from_old_file(old_x50_path, out_dir / "gridref_x50.nc", 50)

    # Auto-derive the x10 sibling by suffix substitution — only the "x50" ->
    # "x10" part of the filename changes, per the existing naming convention.
    old_x10_path = Path(str(old_x50_path).replace("x50", "x10"))
    out_x10 = out_dir / "gridref_x10.nc"
    if old_x10_path.is_file() and old_x10_path != old_x50_path:
        build_from_old_file(old_x10_path, out_x10, 10)
    else:
        current_x10 = gridref_path(10)
        shutil.copyfile(current_x10, out_x10)
        print(f"No old x10 sibling found at {old_x10_path} — copied current {current_x10} -> {out_x10} unchanged "
              f"(this test then only validates x50)")

    # BaseDataModule.__init__ always needs gridref_x1.nc (native grid, used
    # only to build the land mask) regardless of which levels are under
    # test — copy it unchanged, the mask isn't what this test is checking.
    current_x1 = gridref_path(1)
    out_x1 = out_dir / "gridref_x1.nc"
    shutil.copyfile(current_x1, out_x1)
    print(f"Copied {current_x1} -> {out_x1} unchanged (needed for mask-building, not under test here)")


if __name__ == "__main__":
    main()
