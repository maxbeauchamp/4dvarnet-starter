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
    (P x)(i) = x(par(i))                         injection
    P' x = S x + P (x - R S x)                   smooth prolongation (S: bilinear
                                                 interpolation of the parents)
    R P = R P' = I,   Pi0 = I - P'R,   R Pi0 = 0

The state is recomposed with P' (piecewise-constant P left 32 km / 4 km
steps that the details had to undo).

and the state decomposition X = (X_C, dX_M, dX_H) with exact recomposition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .layout import HALF_EXTENT_M, PROJ4  # noqa: F401  (re-exported)

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


def interp(x: Tensor, f: int, valid: Tensor) -> Tensor:
    """S: bilinear interpolation of the parents x[..., n, n] to the f x f
    finer grid (cell centres, edges clamped), ignoring the parents where
    `valid` (broadcastable) is False (land, unobserved)."""
    *lead, ny, nx = x.shape
    m = valid.to(x.dtype).expand_as(x)
    up = lambda a: F.interpolate(a.reshape(-1, 1, ny, nx), scale_factor=f, mode="bilinear",  # noqa: E731
                                 align_corners=False).reshape(*lead, ny * f, nx * f)
    num, den = up(x * m), up(m)
    return torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(num))


def prolong_s(x: Tensor, w: Tensor, f: int) -> Tensor:
    """P' x = S x + P (x - R S x): smooth prolongation with R P' = I exactly
    (R with the child weights w), so that P'R and Pi0 = I - P'R keep the
    exact decomposition while P' x has no steps at the parent edges."""
    sx = interp(x, f, block_sum(w, f) > 0)
    return sx + prolong(x - restrict(sx, w, f)[0], f)


def pi0(x: Tensor, w: Tensor, f: int) -> Tensor:
    """Pi0 = I - P'R: detail projector (R Pi0 x = 0 on parents with w > 0)."""
    return x - prolong_s(restrict(x, w, f)[0], w, f)


def decompose(x_h: Tensor, w_h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """X_H -> (X_C, dX_M, dX_H) with R dX = 0, using ocean weights w_h at H."""
    fm, fh = factor("M", "C"), factor("H", "M")
    x_m, w_m = restrict(x_h, w_h, fh)
    x_c, _ = restrict(x_m, w_m, fm)
    return x_c, x_m - prolong_s(x_c, w_m, fm), x_h - prolong_s(x_m, w_h, fh)


def recompose(x_c: Tensor, dx_m: Tensor, dx_h: Tensor, w_h: Tensor) -> Tensor:
    """X_H = P'(P' X_C + dX_M) + dX_H (exact, no learned decoder)."""
    fm, fh = factor("M", "C"), factor("H", "M")
    w_m = block_sum(w_h, fh)
    return prolong_s(prolong_s(x_c, w_m, fm) + dx_m, w_h, fh) + dx_h
