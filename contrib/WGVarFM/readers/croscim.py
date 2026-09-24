"""Reader of the raw CROSCIM daily NetCDF files (one file per sensor and per day).

Each stream (= sensor) is read on its own native (xc, yc) grid, optionally reduced by
an integer block-mean factor `coarsen` (NaN-aware, no interpolation), normalized, and
returned as valid points (lat, lon, values).

Stream config keys used here:
    files:    glob of the daily files, e.g. /.../ASIP_L3/*nc
    date_fmt: date format found in the file names, e.g. "%Y%m%d"
    vars:     variables (= channels) to read
    coarsen:  integer block-mean factor (default 1 = native)
    norm:     {var: {type: minmax, min, max} | {type: zscore, mean, std}}
"""

import warnings
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr


def _normalize(x, stats):
    if stats["type"] == "minmax":
        return (x - stats["min"]) / (stats["max"] - stats["min"])
    if stats["type"] == "zscore":
        return (x - stats["mean"]) / stats["std"]
    raise ValueError(f"Unknown normalization type {stats['type']}")


def _block_nanmean(a, f):
    """(..., ny, nx) -> (..., ny // f, nx // f) NaN-aware block mean (trailing rows/cols cropped)."""
    ny, nx = a.shape[-2] // f * f, a.shape[-1] // f * f
    a = a[..., :ny, :nx].reshape(*a.shape[:-2], ny // f, f, nx // f, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks -> NaN
        return np.nanmean(a, axis=(-3, -1))


def _block_mean_latlon(lat, lon, f):
    """Block mean of positions done on the unit sphere (safe across the antimeridian)."""
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    x, y, z = _block_nanmean(xyz, f)
    return np.rad2deg(np.arctan2(z, np.hypot(x, y))), np.rad2deg(np.arctan2(y, x))


class CroscimReader:
    def __init__(self, streams, domain_limits=None):
        self.streams = streams
        self.domain_limits = domain_limits
        self.files = {}
        for name, sc in streams.items():
            by_date = {}
            for path in sorted(glob(sc["files"])):
                by_date.update(self._dates_in_name(path, sc["date_fmt"]))
            if not by_date:
                raise FileNotFoundError(f"No file matching {sc['files']} for stream {name}")
            self.files[name] = by_date

    @staticmethod
    def _dates_in_name(path, date_fmt):
        """{date: path} for the first date of format `date_fmt` found in the file name."""
        name = path.rsplit("/", 1)[-1]
        width = len(pd.Timestamp("2000-01-01").strftime(date_fmt))
        for i in range(len(name) - width + 1):
            try:
                d = pd.to_datetime(name[i : i + width], format=date_fmt)
            except ValueError:
                continue
            return {np.datetime64(d.date(), "D"): path}
        return {}

    def dates(self):
        common = set.intersection(*(set(f.keys()) for f in self.files.values()))
        return np.array(sorted(common), dtype="datetime64[D]")

    def read(self, stream_name, date):
        sc = self.streams[stream_name]
        ds = xr.open_dataset(self.files[stream_name][date])
        try:
            if self.domain_limits is not None:
                ds = ds.sel(**self.domain_limits)
            t = ds["time"].values[0]
            values = np.stack([np.squeeze(ds[v].values) for v in sc["vars"]]).astype(np.float32)
            lat = ds["lat"].values.astype(np.float64)
            lon = ds["lon"].values.astype(np.float64)
        finally:
            ds.close()

        f = int(sc.get("coarsen", 1))
        if f > 1:
            values = _block_nanmean(values, f)
            lat, lon = _block_mean_latlon(lat, lon, f)

        for c, v in enumerate(sc["vars"]):
            values[c] = _normalize(values[c], sc["norm"][v])

        values = values.reshape(len(sc["vars"]), -1).T
        lat, lon = lat.ravel(), lon.ravel()
        valid = np.isfinite(values).all(-1) & np.isfinite(lat) & np.isfinite(lon)
        n = int(valid.sum())
        return dict(
            lat=lat[valid].astype(np.float32),
            lon=lon[valid].astype(np.float32),
            values=values[valid],
            datetimes=np.full(n, np.datetime64(t, "ns")),
        )
