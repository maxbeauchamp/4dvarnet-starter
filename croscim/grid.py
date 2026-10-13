"""
grid.py
=======
Nested polar-stereographic grid and fixed scale operators of the CROSCIM
flow-matching cascade (CROSCIM_FM_cascade/formulation, section 1).

Levels C / M / H = 32 / 4 / 1 km on the square domain [-3200, 3200] km in
projected coordinates of ASIP's native 500 m grid (EPSG:3411: lon0 = -45,
true scale at 70N, Hughes 1980 ellipsoid). Edges are aligned: every C cell is exactly 8x8 M
cells and 32x32 H cells, every H cell is exactly a 2x2 block of ASIP pixels.
Row 0 is the top of the domain (largest y), as in the existing CROSCIM grids.

Operators act on the last two (spatial) dims of a tensor, with ocean-fraction
weights w broadcastable against it:

    (R x)(k) = sum_{i in ch(k)} w_i x_i / sum_{i in ch(k)} w_i
    (P x)(i) = x(par(i))
    RP = I,   Pi0 = I - PR,   R Pi0 = 0

and the state decomposition X = (X_C, dX_M, dX_H) with exact recomposition.
"""

from __future__ import annotations

import torch
from torch import Tensor

PROJ4 = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs"
HALF_EXTENT_M = 3_200_000
LEVELS = ("C", "M", "H")
DX_M = {"C": 32_000, "M": 4_000, "H": 1_000}


def factor(fine: str, coarse: str) -> int:
    return DX_M[coarse] // DX_M[fine]


def shape(level: str) -> tuple[int, int]:
    n = 2 * HALF_EXTENT_M // DX_M[level]
    return n, n


def edges(level: str) -> Tensor:
    """Cell edges along x (along y they are the same values, reversed)."""
    return torch.arange(-HALF_EXTENT_M, HALF_EXTENT_M + 1, DX_M[level], dtype=torch.float64)


def centers(level: str) -> tuple[Tensor, Tensor]:
    """Cell centres (x increasing with column, y decreasing with row)."""
    e = edges(level)
    x = 0.5 * (e[:-1] + e[1:])
    return x, x.flip(0)


def cell_index(x: Tensor, y: Tensor, level: str) -> tuple[Tensor, Tensor, Tensor]:
    """(row, col) of the cell containing projected points (x, y), and a mask
    of points inside the domain (out-of-domain indices are meaningless)."""
    dx = DX_M[level]
    ix = torch.floor((x + HALF_EXTENT_M) / dx).long()
    iy = torch.floor((HALF_EXTENT_M - y) / dx).long()
    n = shape(level)[0]
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    return iy, ix, inside


def tile_slices(iy0: int, ix0: int, n_c: int) -> dict[str, tuple[slice, slice]]:
    """Slices of the n_c x n_c C tile starting at C cell (iy0, ix0), and of
    the M and H tiles covering exactly the same area."""
    out = {}
    for level in LEVELS:
        f = factor(level, "C")
        out[level] = (slice(iy0 * f, (iy0 + n_c) * f), slice(ix0 * f, (ix0 + n_c) * f))
    return out


def tile_edges(level: str, iy0: int, ix0: int, n_c: int) -> tuple[Tensor, Tensor]:
    """(y edges, x edges) of the level-`level` cells of a C tile (y decreasing)."""
    f, dx = factor(level, "C"), DX_M[level]
    k = torch.arange((n_c * f) + 1, dtype=torch.float64)
    return HALF_EXTENT_M - (iy0 * f + k) * dx, -HALF_EXTENT_M + (ix0 * f + k) * dx


def _blocks(x: Tensor, f: int) -> Tensor:
    *lead, ny, nx = x.shape
    return x.reshape(*lead, ny // f, f, nx // f, f)


def block_sum(x: Tensor, f: int) -> Tensor:
    """Sum over the f x f children of each parent."""
    return _blocks(x, f).sum((-3, -1))


def restrict(x: Tensor, w: Tensor, f: int) -> tuple[Tensor, Tensor]:
    """R: weighted mean over f x f children. Returns (R x, parent weights),
    the parent weights being the summed child weights, so that restrictions
    compose (R_{C<-M} R_{M<-H} = R_{C<-H}). Parents with zero weight (all
    land, or no observation) are set to 0."""
    wp = block_sum(w, f)
    num = block_sum(w * x, f)
    ocean = wp > 0
    return torch.where(ocean, num / torch.where(ocean, wp, torch.ones_like(wp)), torch.zeros_like(num)), wp


def prolong(x: Tensor, f: int) -> Tensor:
    """P: injection of each parent value into its f x f children."""
    *lead, ny, nx = x.shape
    return x[..., :, None, :, None].expand(*lead, ny, f, nx, f).reshape(*lead, ny * f, nx * f)


def pi0(x: Tensor, w: Tensor, f: int) -> Tensor:
    """Pi0 = I - PR: detail projector (R Pi0 x = 0 on parents with w > 0)."""
    return x - prolong(restrict(x, w, f)[0], f)


def decompose(x_h: Tensor, w_h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """X_H -> (X_C, dX_M, dX_H) with R dX = 0, using ocean weights w_h at H."""
    x_m, w_m = restrict(x_h, w_h, factor("H", "M"))
    x_c, _ = restrict(x_m, w_m, factor("M", "C"))
    return x_c, x_m - prolong(x_c, factor("M", "C")), x_h - prolong(x_m, factor("H", "M"))


def recompose(x_c: Tensor, dx_m: Tensor, dx_h: Tensor) -> Tensor:
    """X_H = P_{C->H} X_C + P_{M->H} dX_M + dX_H (exact, no learned decoder)."""
    x_m = prolong(x_c, factor("M", "C")) + dx_m
    return prolong(x_m, factor("H", "M")) + dx_h
