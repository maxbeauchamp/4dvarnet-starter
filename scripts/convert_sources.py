#!/usr/bin/env python3
"""Convert the CROSCIM source NetCDF files into the Zarr stores read by the
cascade dataloader (croscim/data.py).

Format change only: crop to the cascade domain [-3200, 3200] km and rechunk
by day. No regridding, no averaging, no target or statistic: native values
are kept (ASIP 500 m bytes, 5 km floats), everything else is computed on the
fly at training time.

Each store is written next to the NetCDF directory of its source, under
--src-root (see STORES in cascade/data.py; an existing store is overwritten):

    ASIP_L3_zarr/{sic,std,acq,days}                          (day, 12800, 12800)
    {data_noise,out_CRISTAL,atm_data}_zarr/{<var>,xc,yc,days} (day, ny5, nx5)

ASIP `acq` = acquisition time in minutes since 00:00 of the file day (int16,
-32768 = missing); it defines the swaths of the input / target partition.
The 5 km grids are checked to be in the ASIP projection (EPSG:3411) by
comparing their lon/lat with the projected xc/yc.

Usage:
    python convert_cascade_sources.py --src-root /Odyssey/public/CROSCIM_dataset \
        --start 2021-01-01 --end 2023-12-31 --workers 16
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import xarray as xr
import zarr

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from croscim import grid as g
from croscim.data import EPOCH, SOURCES, STORES

FILES = {"asip": "ASIP_L3/*.nc", "cimr": "data_noise/CIMR5km_*.nc",
         "cristal": "out_CRISTAL/CRISTAL5km_*.nc", "era5": "atm_data/atm5km_*.nc"}
H = g.HALF_EXTENT_M


def file_day(path):
    m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", os.path.basename(path))
    return (dt.date(*map(int, m.groups())) - EPOCH).days


def list_files(src_root, src, start, end):
    files = {file_day(p): p for p in glob(os.path.join(src_root, FILES[src]))}
    return sorted((d, p) for d, p in files.items() if start <= d <= end)


def asip_window(path):
    with xr.open_dataset(path, mask_and_scale=False, decode_times=False) as ds:
        xc, yc = ds.xc.values, ds.yc.values
    ix = np.flatnonzero((xc > -H) & (xc < H))
    iy = np.flatnonzero((yc > -H) & (yc < H))
    n = 2 * H // 500
    assert len(ix) == len(iy) == n and xc[ix[0]] == -H + 250 and yc[iy[0]] == H - 250, \
        "ASIP grid is not the expected 500 m EPSG:3411 grid covering the domain"
    return slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)


def grid5_window(path):
    from pyproj import Transformer
    with xr.open_dataset(path) as ds:
        xc, yc = ds.xc.values, ds.yc.values
        lon, lat = ds.lon.values[::50, ::50], ds.lat.values[::50, ::50]
    x, y = Transformer.from_crs("EPSG:4326", g.PROJ4, always_xy=True).transform(lon, lat)
    err = np.nanmax(np.hypot(x - xc[None, ::50], y - yc[::50, None]))
    assert err < 500, f"{path}: 5 km grid not in the ASIP projection (max error {err:.0f} m)"
    assert np.allclose(np.diff(xc), 5000) and np.allclose(np.diff(yc), -5000)
    ix = np.flatnonzero((xc > -H - 2500) & (xc < H + 2500))
    iy = np.flatnonzero((yc > -H - 2500) & (yc < H + 2500))
    return slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1), xc[ix], yc[iy]


def convert_asip_day(store, k, day, path, win):
    with xr.open_dataset(path, mask_and_scale=False, decode_times=False) as ds:
        ds = ds.isel(time=0, yc=win[0], xc=win[1])
        sic = ds.sic.values.astype(np.int8).view(np.uint8)  # fill -1 -> 255
        std = ds.standard_deviation_sic.values.astype(np.int8).view(np.uint8)
        acq_s = ds.acq_time.values
    acq = np.round((acq_s - day * 86400.0) / 60.0)
    acq = np.where(acq_s == 0, -32768, acq).astype(np.int16)
    for name, vals in (("sic", sic), ("std", std), ("acq", acq)):
        zarr.open_array(os.path.join(store, name), mode="r+")[k] = vals


def convert_5km_day(store, src, k, path, win):
    with xr.open_dataset(path) as ds:
        for name in SOURCES[src]:
            vals = ds[name].squeeze().values[win[0], win[1]].astype(np.float32)
            zarr.open_array(os.path.join(store, name), mode="r+")[k] = vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sources", nargs="+", default=["asip", *SOURCES])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    start, end = ((dt.date.fromisoformat(d) - EPOCH).days for d in (args.start, args.end))

    for src in args.sources:
        files = list_files(args.src_root, src, start, end)
        if not files:
            raise FileNotFoundError(f"No {src} files in {args.src_root} between {args.start} and {args.end}")
        days = np.array([d for d, _ in files], np.int64)
        path = os.path.join(args.src_root, STORES[src])
        if src == "asip":
            win = asip_window(files[0][1])
            n = 2 * H // 500
            for name, dtype, fill in (("sic", "u1", 255), ("std", "u1", 255), ("acq", "i2", -32768)):
                zarr.open_array(os.path.join(path, name), mode="w", shape=(len(days), n, n),
                                chunks=(1, 1024, 1024), dtype=dtype, fill_value=fill)
        else:
            *win, xc, yc = grid5_window(files[0][1])
            for name in SOURCES[src]:
                zarr.open_array(os.path.join(path, name), mode="w", shape=(len(days), len(yc), len(xc)),
                                chunks=(1, 320, 320), dtype="f4", fill_value=np.nan)
            zarr.open_array(os.path.join(path, "xc"), mode="w", shape=xc.shape, dtype="f8")[:] = xc
            zarr.open_array(os.path.join(path, "yc"), mode="w", shape=yc.shape, dtype="f8")[:] = yc
        zarr.open_array(os.path.join(path, "days"), mode="w", shape=days.shape, dtype="i8")[:] = days

        with ProcessPoolExecutor(args.workers) as pool:
            if src == "asip":
                jobs = [pool.submit(convert_asip_day, path, k, d, p, win) for k, (d, p) in enumerate(files)]
            else:
                jobs = [pool.submit(convert_5km_day, path, src, k, p, win) for k, (_, p) in enumerate(files)]
            t0, step = time.perf_counter(), max(1, len(jobs) // 20)
            for n, j in enumerate(as_completed(jobs), 1):
                j.result()
                if n % step == 0 or n == len(jobs):
                    dt = time.perf_counter() - t0
                    print(f"{src}: {n}/{len(jobs)} days, {dt / n:.1f} s/day (wall), "
                          f"ETA {(len(jobs) - n) * dt / n / 60:.0f} min", flush=True)
        print(f"{src}: {len(days)} days -> {path}")


if __name__ == "__main__":
    main()
