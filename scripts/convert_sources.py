#!/usr/bin/env python3
"""Convert the CROSCIM source NetCDF files into the Zarr stores read by the
cascade dataloader (croscim/data.py).

Format change only: crop to the cascade domain [-3200, 3200] km and rechunk
(few, large chunks). No regridding, no averaging, no target or statistic: native values
are kept (ASIP 500 m bytes, 5 km floats), everything else is computed on the
fly at training time.

Each store is written next to the NetCDF directory of its source, under
--src-root (see STORES in croscim/layout.py; an existing store is overwritten):

    ASIP_L3_zarr/{sic,std,acq,days}                          (day, 12800, 12800)
    ASIP_L3_zarr/{m_mean,m_w}  4 km maps of each day         (day, 1600, 1600)
    {data_noise,out_CRISTAL,atm_data}_zarr/{<var>,xc,yc,days} (day, ny5, nx5)

ASIP `acq` = acquisition time in minutes since 00:00 of the file day (int16,
-32768 = missing); it defines the swaths of the input / target partition.
The 5 km grids are checked to be in the ASIP projection (EPSG:3411) by
comparing their lon/lat with the projected xc/yc.

No small accesses on the network file system (they dominated: ~70 s of
NetCDF block reads and ~70 s of chunk writes per ASIP day on Odyssey): each
NetCDF file is first copied to a local --work-dir (one sequential read), the
stores are written there, then copied to --src-root with parallel file copies
once a source is done.

Usage:
    python scripts/convert_sources.py --src-root /Odyssey/public/CROSCIM_dataset \
        --start 2021-01-01 --end 2023-12-31 --workers 8 --work-dir /SCRATCH/$USER/$SLURM_JOB_ID
"""
from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from croscim.asip import M_PER_500M, asip_to_m
from croscim.layout import EPOCH, HALF_EXTENT_M, PROJ4, SOURCES, STORES  # no torch: fast start

FILES = {"asip": "ASIP_L3/*.nc", "cimr": "data_noise/CIMR5km_*.nc",
         "cristal": "out_CRISTAL/CRISTAL5km_*.nc", "era5": "atm_data/atm5km_*.nc"}
H = HALF_EXTENT_M
# Few, large files (copying many small files to /SCRATCH is what costs):
# ASIP one day x 2048^2 per chunk, chunks that are entirely missing not
# written; 5 km sources 8 days x 512^2 per chunk (the dataloader reads 15
# consecutive days at once), written by groups of 8 days so that two workers
# never write the same chunk.
ASIP_CHUNKS = (1, 2048, 2048)
ASIP_M_CHUNKS = (1, 800, 800)  # stored 4 km maps (1600^2 per day)
DAYS_PER_CHUNK_5KM = 8
CHUNKS_5KM = (DAYS_PER_CHUNK_5KM, 512, 512)


def local_copy(path, work_dir):
    """Copy a source file to the local work dir (one sequential read) and
    return the copy's path; the caller removes it."""
    dst = os.path.join(work_dir, f"src_{os.getpid()}_{os.path.basename(path)}")
    shutil.copyfile(path, dst)
    return dst


def publish(local, dest, n_threads=16):
    """Replace the store `dest` by the local store `local` (parallel copies:
    the network file system is slow per file, not per byte)."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    files = [os.path.relpath(os.path.join(d, f), local) for d, _, fs in os.walk(local) for f in fs]
    for d in {os.path.dirname(f) for f in files}:
        os.makedirs(os.path.join(dest, d), exist_ok=True)
    with ThreadPoolExecutor(n_threads) as pool:
        list(pool.map(lambda f: shutil.copyfile(os.path.join(local, f), os.path.join(dest, f)), files))
    shutil.rmtree(local)
    return len(files)


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
    x, y = Transformer.from_crs("EPSG:4326", PROJ4, always_xy=True).transform(lon, lat)
    err = np.nanmax(np.hypot(x - xc[None, ::50], y - yc[::50, None]))
    assert err < 500, f"{path}: 5 km grid not in the ASIP projection (max error {err:.0f} m)"
    assert np.allclose(np.diff(xc), 5000) and np.allclose(np.diff(yc), -5000)
    ix = np.flatnonzero((xc > -H - 2500) & (xc < H + 2500))
    iy = np.flatnonzero((yc > -H - 2500) & (yc < H + 2500))
    return slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1), xc[ix], yc[iy]


def convert_asip_day(store, k, day, path, win, work_dir):
    """One ASIP day. Each variable is read once for the cropped domain (each
    NetCDF chunk decompressed once, whatever the file chunking); the
    acquisition-time conversion (float64 temporaries), the 4 km maps and the
    writes go by bands of zarr chunk rows. Peak ~1.5 GB per process."""
    arrs = {name: zarr.open_array(os.path.join(store, name), mode="r+", write_empty_chunks=False)
            for name in ("sic", "std", "acq", "m_mean", "m_w")}
    src = local_copy(path, work_dir)
    with xr.open_dataset(src, mask_and_scale=False, decode_times=False) as ds:
        ds = ds.isel(time=0, yc=win[0], xc=win[1])
        sic = ds.sic.values.astype(np.int8).view(np.uint8)  # fill -1 -> 255
        std = ds.standard_deviation_sic.values.astype(np.int8).view(np.uint8)
        acq_s = ds.acq_time.values  # float32
    os.remove(src)
    rows = ASIP_CHUNKS[1]
    for r0 in range(0, sic.shape[0], rows):
        a = acq_s[r0:r0 + rows]
        acq = np.where(a == 0, -32768, np.round((a - day * 86400.0) / 60.0)).astype(np.int16)
        for name, vals in (("sic", sic[r0:r0 + rows]), ("std", std[r0:r0 + rows]), ("acq", acq)):
            arrs[name][k, r0:r0 + rows] = vals
        m_mean, m_w = asip_to_m(sic[r0:r0 + rows], std[r0:r0 + rows])
        rm = slice(r0 // M_PER_500M, (r0 + rows) // M_PER_500M)
        arrs["m_mean"][k, rm], arrs["m_w"][k, rm] = m_mean, m_w
    return 1


def convert_5km_days(store, src, k0, paths, win, work_dir):
    """Days k0 .. k0 + len(paths) - 1 (one chunk along time), written at once."""
    vals = {name: [] for name in SOURCES[src]}
    for path in paths:
        local = local_copy(path, work_dir)
        with xr.open_dataset(local) as ds:
            for name in SOURCES[src]:
                vals[name].append(ds[name].squeeze().values[win[0], win[1]].astype(np.float32))
        os.remove(local)
    for name in SOURCES[src]:
        zarr.open_array(os.path.join(store, name), mode="r+")[k0:k0 + len(paths)] = np.stack(vals[name])
    return len(paths)


def log(msg, t0=None):
    """Startup log on stdout, flushed (with the duration since t0 if given)."""
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}" + ("" if t0 is None else f" ({time.perf_counter() - t0:.1f} s)"),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sources", nargs="+", default=["asip", *SOURCES])
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel processes (1: everything in this process, to see the real error of a crash)")
    ap.add_argument("--work-dir", default=None,
                    help="local disk for the source copies and the stores before publication "
                         "(default: a temporary dir under $TMPDIR or /tmp)")
    args = ap.parse_args()
    start, end = ((dt.date.fromisoformat(d) - EPOCH).days for d in (args.start, args.end))
    work_dir = tempfile.mkdtemp(prefix="croscim_convert_", dir=args.work_dir or os.environ.get("TMPDIR"))
    log(f"local work dir: {work_dir}")

    for src in args.sources:
        t0 = time.perf_counter()
        files = list_files(args.src_root, src, start, end)
        if not files:
            raise FileNotFoundError(f"No {src} files in {args.src_root} between {args.start} and {args.end}")
        days = np.array([d for d, _ in files], np.int64)
        log(f"{src}: {len(files)} files, {EPOCH + dt.timedelta(int(days[0]))} -> {EPOCH + dt.timedelta(int(days[-1]))}", t0)
        dest = os.path.join(args.src_root, STORES[src])
        path = os.path.join(work_dir, STORES[src])  # written locally, published at the end
        t0 = time.perf_counter()
        first = local_copy(files[0][1], work_dir)
        if src == "asip":
            win = asip_window(first)
            n = 2 * H // 500
            for name, dtype, fill in (("sic", "u1", 255), ("std", "u1", 255), ("acq", "i2", -32768)):
                zarr.open_array(os.path.join(path, name), mode="w", shape=(len(days), n, n),
                                chunks=ASIP_CHUNKS, dtype=dtype, fill_value=fill)
            for name in ("m_mean", "m_w"):
                nm = n // M_PER_500M
                zarr.open_array(os.path.join(path, name), mode="w", shape=(len(days), nm, nm),
                                chunks=ASIP_M_CHUNKS, dtype="f4", fill_value=0.0)
        else:
            *win, xc, yc = grid5_window(first)
            for name in SOURCES[src]:
                zarr.open_array(os.path.join(path, name), mode="w", shape=(len(days), len(yc), len(xc)),
                                chunks=CHUNKS_5KM, dtype="f4", fill_value=np.nan)
            zarr.open_array(os.path.join(path, "xc"), mode="w", shape=xc.shape, dtype="f8")[:] = xc
            zarr.open_array(os.path.join(path, "yc"), mode="w", shape=yc.shape, dtype="f8")[:] = yc
        zarr.open_array(os.path.join(path, "days"), mode="w", shape=days.shape, dtype="i8")[:] = days
        os.remove(first)
        log(f"{src}: grid checked, local store created", t0)

        if src == "asip":
            tasks = [(convert_asip_day, (path, k, d, p, win, work_dir)) for k, (d, p) in enumerate(files)]
        else:
            n = DAYS_PER_CHUNK_5KM
            tasks = [(convert_5km_days, (path, src, k0, [p for _, p in files[k0:k0 + n]], win, work_dir))
                     for k0 in range(0, len(files), n)]
        with tqdm(total=len(files), desc=src, unit="day", mininterval=10) as bar:
            if args.workers <= 1:
                for fn, a in tasks:
                    bar.update(fn(*a))
            else:
                # "spawn": fresh worker processes, not forks of a parent in which
                # HDF5 / netCDF and Blosc are already initialised
                with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
                    for j in as_completed([pool.submit(fn, *a) for fn, a in tasks]):
                        bar.update(j.result())
        log(f"{src}: {len(days)} days converted")
        t0 = time.perf_counter()
        n_files = publish(path, dest)
        log(f"{src}: {n_files} files published to {dest}", t0)
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
