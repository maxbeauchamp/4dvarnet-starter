#!/usr/bin/env python3
"""Build the static CROSCIM grid-reference files (gridref_x{level}.nc) from a
single ASIP file.

These pin the spatial grid (xc/yc/lon/lat) used by the dataloaders to ASIP's
native 500m grid, coarsened to each nominal `multires` level — independently
of which sources a given xp actually declares in `satellite_vars`. See
`gridref_path()` in contrib/CROSCIM/dataloaders/load_data.py for how they get
consumed at runtime.

Run once (the underlying ASIP grid is static); re-run only if that grid
itself changes.

Level 1 (native asip resolution, no coarsening) must be included — it's the
base grid BaseDataModule.__init__ uses to build the land mask, needed even
when no xp uses `resize=1`/`multires` level 1 directly.

Usage:
    python build_grid_reference.py --asip-dir /Odyssey/public/CROSCIM_dataset/ASIP_L3 \
        --out-dir contrib/CROSCIM/gridref --levels 1 2 10 50
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import xarray as xr

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from contrib.CROSCIM.dataloaders.load_data import _GRIDREF_DIR_DEFAULT, fast_coarsen_xr

DEFAULT_LEVELS = [1, 2, 10, 50]
DEFAULT_ASIP_DIR = "/Odyssey/public/CROSCIM_dataset/ASIP_L3"
DEFAULT_OUT_DIR = _GRIDREF_DIR_DEFAULT


def build_gridref(asip_file, level, out_path):
    with xr.open_dataset(asip_file) as ds:
        grid = xr.Dataset(coords={"xc": ds["xc"], "yc": ds["yc"], "lon": ds["lon"], "lat": ds["lat"]})
    if level != 1:
        grid = fast_coarsen_xr(grid, factor_y=level, factor_x=level)
    grid.to_netcdf(out_path)
    print(f"Wrote {out_path}  (xc={grid.sizes['xc']}, yc={grid.sizes['yc']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asip-dir", default=DEFAULT_ASIP_DIR, help="Directory containing ASIP *.nc files")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where to write gridref_x{level}.nc")
    ap.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS,
                     help="Nominal multires levels (500m units) to build, e.g. 2 10 50")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.asip_dir, "*nc")))
    if not files:
        raise FileNotFoundError(f"No ASIP files found in {args.asip_dir}")
    asip_file = files[0]
    print(f"Building grid reference from {asip_file}")

    os.makedirs(args.out_dir, exist_ok=True)
    for level in args.levels:
        out_path = os.path.join(args.out_dir, f"gridref_x{level}.nc")
        build_gridref(asip_file, level, out_path)


if __name__ == "__main__":
    main()
