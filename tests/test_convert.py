"""Test of scripts/convert_sources.py on small fake NetCDF files with the
layout of the real 5 km sources (CIMR, CRISTAL, ERA5 atm). Run from the repo
root, e.g.:

    pytest tests/test_convert.py
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import zarr

xr = pytest.importorskip("xarray")
pyproj = pytest.importorskip("pyproj")

from croscim.layout import PROJ4, SOURCES, STORES  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATES = ("2022-02-01", "2022-02-02", "2022-02-04")  # 2022-02-03 missing
PATTERN = {"cimr": ("data_noise", "CIMR5km_{}_mod.nc"), "cristal": ("out_CRISTAL", "CRISTAL5km_no_N_{}.nc"),
           "era5": ("atm_data", "atm5km_{}.nc")}


def _grid5():
    xc = -3_847_500.0 + 5_000.0 * np.arange(1520)
    yc = 2_472_500.0 - 5_000.0 * np.arange(1475)
    xx, yy = np.meshgrid(xc, yc)
    lon, lat = pyproj.Transformer.from_crs(PROJ4, "EPSG:4326", always_xy=True).transform(xx, yy)
    return xc, yc, lon.astype(np.float32), lat.astype(np.float32)


def _write_sources(root):
    xc, yc, lon, lat = _grid5()
    rng = np.random.default_rng(0)
    truth = {}
    for src, (folder, pattern) in PATTERN.items():
        os.makedirs(os.path.join(root, folder))
        for date in DATES:
            data = {}
            for name in SOURCES[src]:
                v = rng.random((1475, 1520))
                v[rng.random((1475, 1520)) < 0.3] = np.nan
                data[name] = (("yc", "xc"), v)
                truth[(src, name, date)] = v
            ds = xr.Dataset(data, coords={"xc": xc, "yc": yc, "lon": (("yc", "xc"), lon), "lat": (("yc", "xc"), lat)})
            ds.to_netcdf(os.path.join(root, folder, pattern.format(date)))
    return truth, xc, yc


def test_convert_5km_sources(tmp_path):
    root = str(tmp_path)
    truth, xc, yc = _write_sources(root)
    cmd = [sys.executable, os.path.join(REPO, "scripts", "convert_sources.py"), "--src-root", root,
           "--start", DATES[0], "--end", DATES[-1], "--sources", *SOURCES, "--workers", "2"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert "cimr: 3 files" in out.stdout and "era5: 3 days converted" in out.stdout and "published to" in out.stdout

    keep_x = (xc > -3_202_500) & (xc < 3_202_500)
    keep_y = (yc > -3_202_500) & (yc < 3_202_500)
    for src in SOURCES:
        store = os.path.join(root, STORES[src])
        days = zarr.open_array(os.path.join(store, "days"), mode="r")[:]
        assert days.tolist() == [19024, 19025, 19027]
        assert np.array_equal(zarr.open_array(os.path.join(store, "xc"), mode="r")[:], xc[keep_x])
        assert np.array_equal(zarr.open_array(os.path.join(store, "yc"), mode="r")[:], yc[keep_y])
        for name in SOURCES[src]:
            arr = zarr.open_array(os.path.join(store, name), mode="r")
            assert arr.chunks == (8, 512, 512)  # several days per chunk: few files to stage
            for k, date in enumerate(DATES):
                ref = truth[(src, name, date)][np.ix_(keep_y, keep_x)].astype(np.float32)
                assert np.array_equal(arr[k], ref, equal_nan=True), (src, name, date)


def test_convert_refuses_a_grid_in_another_projection(tmp_path):
    root = str(tmp_path)
    _write_sources(root)
    path = os.path.join(root, "data_noise", "CIMR5km_2022-02-01_mod.nc")
    with xr.open_dataset(path) as f:
        ds = f.load()
    ds["lon"] = ds["lon"] + 1.0  # lon / lat no longer match xc / yc in EPSG:3411
    ds.to_netcdf(path)
    cmd = [sys.executable, os.path.join(REPO, "scripts", "convert_sources.py"), "--src-root", root,
           "--start", DATES[0], "--end", DATES[-1], "--sources", "cimr", "--workers", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert out.returncode != 0 and "not in the ASIP projection" in out.stderr
