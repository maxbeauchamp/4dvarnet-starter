"""
ocean.py
========
Ocean fraction omega_H of the 1 km cells, from Natural Earth land polygons
(weights of R, formulation section 1). Computed once at datamodule setup: it
is a constant of the grid, not a data product.

Land polygons are projected onto the grid CRS and filled on a supersampled
grid by an even-odd scanline rule (crossing parity), then block-averaged.
"""

from __future__ import annotations

import numpy as np

from . import grid as g


def land_fraction_h(rings: list[np.ndarray], supersample: int = 4, strip_rows: int = 2048) -> np.ndarray:
    """Land fraction [6400, 6400] (float32) of the H cells covered by the
    union of `rings` (closed (N, 2) arrays of projected x, y in metres;
    exterior rings and holes alike, even-odd rule)."""
    d = g.DX_M["H"] / supersample
    n = g.shape("H")[0] * supersample
    rows, cols = [], []
    for ring in rings:
        # continuous pixel coordinates; pixel centres at integers
        c = (ring[:, 0] + g.HALF_EXTENT_M) / d - 0.5
        r = (g.HALF_EXTENT_M - ring[:, 1]) / d - 0.5
        r0, r1, c0, c1 = r[:-1], r[1:], c[:-1], c[1:]
        lo, hi = np.ceil(np.minimum(r0, r1)), np.ceil(np.maximum(r0, r1))  # rows k with min <= k < max
        lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
        span = (hi - lo).astype(np.int64)
        if span.sum() == 0:
            continue
        e = np.repeat(np.arange(len(span)), span)
        k = lo[e] + (np.arange(span.sum()) - np.repeat(np.cumsum(span) - span, span))
        t = (k - r0[e]) / (r1[e] - r0[e])
        xc = c0[e] + t * (c1[e] - c0[e])
        first = np.clip(np.ceil(xc), 0, n)  # first pixel centre right of the crossing
        keep = first < n
        rows.append(k[keep].astype(np.int64))
        cols.append(first[keep].astype(np.int64))
    rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)

    out = np.empty(g.shape("H"), np.float32)
    strip_rows -= strip_rows % supersample
    for s0 in range(0, n, strip_rows):
        nr = min(strip_rows, n - s0)
        sel = (rows >= s0) & (rows < s0 + nr)
        count = np.bincount((rows[sel] - s0) * n + cols[sel], minlength=nr * n).reshape(nr, n)
        inside = (np.cumsum(count, axis=1) & 1).astype(np.float32)
        out[s0 // supersample:(s0 + nr) // supersample] = inside.reshape(
            nr // supersample, supersample, n // supersample, supersample).mean((1, 3))
    return out


def ocean_fraction_h(natural_earth_zip: str, supersample: int = 4) -> np.ndarray:
    """omega_H = 1 - land fraction, from the Natural Earth land shapefile
    (assets/ne_10m_land.zip)."""
    import geopandas as gpd
    from pyproj import Transformer

    land = gpd.read_file(f"zip://{natural_earth_zip}").explode(index_parts=False)
    land = land.cx[:, 30:]  # polygons reaching north of 30N
    to_grid = Transformer.from_crs("EPSG:4326", g.PROJ4, always_xy=True)
    rings = []
    for poly in land.geometry:
        for ring in [poly.exterior, *poly.interiors]:
            lon, lat = np.asarray(ring.coords).T
            rings.append(np.stack(to_grid.transform(lon, np.maximum(lat, 0.0)), -1))
    return 1.0 - land_fraction_h(rings, supersample)
