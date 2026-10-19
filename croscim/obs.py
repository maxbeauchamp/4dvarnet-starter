"""
obs.py
======
On-the-fly observation operators of the CROSCIM cascade
(CROSCIM_FM_cascade/formulation, sections 3c, 4 and 8): gridding G_l,
input / target partition, detail targets and sub-grid statistics S_C.

Everything is torch and device-agnostic: it runs on the GPU after the batch
transfer. Nothing is interpolated: dense fields aligned with the grid are
block-averaged, other dense fields are remapped by exact 1D overlap matrices,
point observations are binned into the cell containing them. Gridded fields
come as (sigma^-2-weighted mean, sum of weights); a weight of 0 means "no
observation" and the mean is then 0.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import grid as g

# ---- sub-grid statistics S_C (formulation section 8) ----------------------
S_NAMES = ("h_ow", "h_thin", "h_mid", "h_pack", "sic_std", "edge_density", "aniso_c", "aniso_s", "sit_std")
SIC_BIN_EDGES = (0.15, 0.5, 0.85)  # histogram classes [0,.15) [.15,.5) [.5,.85) [.85,1]
EDGE_JUMP = 0.1                    # |SIC difference| between 1 km neighbours marking a lead / floe edge
                                   # (0.3: non-zero in 1 % of the cells on 2022-02 ASIP, 0.1: 7 %)
ANISO_E0 = 1e-3                    # gradient-energy floor of the anisotropy (no structure -> 0):
                                   # ~ (3 % SIC per km)^2, the retrieval noise level (1e-2 flattened it)
# Uncertainty of the targets S^: sigma_S^2 = sigma_0^2 + sigma_samp(S^)^2 / n_eff, with
# sigma_samp the standard error of the statistic for one independent sample:
# sqrt(p (1 - p)) for the fractions (histogram, edge_density), S / sqrt(2) for
# the standard deviations (sic_std, sit_std), 1 - a^2 for the anisotropy
# (as for a correlation coefficient, a = magnitude). sigma_0: 10 % of the
# channel std measured on 2022-01-15 .. 2022-10-31, 20 % for the texture
# channels (sic_std, edge_density, anisotropy): with 10 % (and worst-case
# constants before that) their ensemble spread was 2-10 x below the error.
S_SIGMA0 = (0.044, 0.0103, 0.0143, 0.046, 0.012, 0.0064, 0.0196, 0.0187, 0.0062)
GAMMA_S = 0.5           # min. observed ocean fraction of a C cell for the ASIP-based channels
ASIP_PIX_PER_SAMPLE = 4  # 1 km pixels per independent sample (n_eff)
MIN_SIT_OBS = 4          # min. observed M cells of a C cell for sit_std


def _div(num: Tensor, den: Tensor) -> Tensor:
    return torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(num))


def overlap_matrix(src_centers: Tensor, src_dx: float, dst_edges: Tensor) -> Tensor:
    """A[i, j] = fraction of the (1D) source cell j lying in destination
    cell i. Orientation-free: dst_edges may be increasing or decreasing."""
    s_lo, s_hi = src_centers - src_dx / 2, src_centers + src_dx / 2
    d_lo = torch.minimum(dst_edges[:-1], dst_edges[1:])
    d_hi = torch.maximum(dst_edges[:-1], dst_edges[1:])
    ov = torch.minimum(d_hi[:, None], s_hi[None]) - torch.maximum(d_lo[:, None], s_lo[None])
    return ov.clamp_min(0) / src_dx


def grid_dense(v: Tensor, sigma: Tensor, ay: Tensor, ax: Tensor) -> tuple[Tensor, Tensor]:
    """Exact area remap of a dense field v[..., ny, nx] (NaN = missing) with
    per-value sigma (broadcastable), using overlap matrices ay [n, ny], ax [m, nx]."""
    w = torch.isfinite(v) * sigma ** -2
    num = ay @ (w * torch.nan_to_num(v)) @ ax.T
    den = ay @ w @ ax.T
    return _div(num, den), den


def grid_points(x: Tensor, y: Tensor, v: Tensor, w: Tensor, group: Tensor, n_group: int,
                y_edges: Tensor, x_edges: Tensor) -> tuple[Tensor, Tensor]:
    """Bin points into the regular cells defined by y_edges (decreasing) and
    x_edges (increasing). `group` (e.g. day x variable) indexes independent
    output maps. Returns (mean, sum of weights), each [n_group, ny, nx]."""
    ny, nx = len(y_edges) - 1, len(x_edges) - 1
    dx = float(x_edges[1] - x_edges[0])
    ix = torch.floor((x - x_edges[0]) / dx).long()
    iy = torch.floor((y_edges[0] - y) / dx).long()
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    k = ((group * ny + iy) * nx + ix)[ok]
    num = torch.zeros(n_group * ny * nx, dtype=v.dtype, device=v.device).index_add_(0, k, (w * v)[ok])
    den = torch.zeros_like(num).index_add_(0, k, w[ok].to(v.dtype))
    return _div(num, den).view(n_group, ny, nx), den.view(n_group, ny, nx)


def grid_asip(sic: Tensor, std: Tensor, keep: Tensor, sigma_floor: float) -> tuple[Tensor, Tensor]:
    """ASIP 500 m (uint8 %, 255 = missing) -> H: sigma^-2-weighted 2x2 means.
    `keep` selects the pixels used (e.g. Y^in or Y^tgt)."""
    keep = keep & (sic != 255)
    sigma = (std.float() / 100).clamp_min(sigma_floor)
    v = torch.where(keep, sic.float() / 100, torch.zeros_like(sigma))
    return g.restrict(v, keep * sigma ** -2, 2)


def combine(*fields: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """sigma^-2 combination of several (mean, weight) estimates of one field."""
    num = sum(m * w for m, w in fields)
    den = sum(w for _, w in fields)
    return _div(num, den), den


# ---- partition Y = Y^in u Y^tgt ------------------------------------------

def unit_partition(unit: Tensor, valid: Tensor, p_tgt: float, gen: torch.Generator) -> Tensor:
    """Each distinct value of `unit` among valid entries (swath / pass / block
    id) goes to the target set with probability p_tgt. Returns the target mask."""
    ids, inv = torch.unique(unit[valid], return_inverse=True)
    pick = torch.rand(len(ids), generator=gen, device=unit.device) < p_tgt
    tgt = torch.zeros_like(valid)
    tgt[valid] = pick[inv]
    return tgt


# ---- targets -------------------------------------------------------------

def detail_target(xhat: Tensor, w_obs: Tensor, omega: Tensor, f: int, gamma: float) -> tuple[Tensor, Tensor]:
    """dX^ = X^ - P R X^ with R weighted by the observed ocean fraction, on
    parents whose observed ocean fraction is >= gamma. Returns (dX^, mask);
    R_{omega*mask} dX^ = 0."""
    valid = (w_obs > 0) & (omega > 0)
    w = omega * valid
    cov, _ = g.restrict(valid.to(omega.dtype), omega, f)
    mask = valid & g.prolong(cov >= gamma, f)
    d = g.pi0(torch.where(valid, xhat, torch.zeros_like(xhat)), w, f)
    return torch.where(mask, d, torch.zeros_like(d)), mask


def _step(x: Tensor, tau: float | None) -> Tensor:
    """1[x > 0], or its sigmoid relaxation of width tau (differentiable S)."""
    return (x > 0).to(x.dtype) if tau is None else torch.sigmoid(x / tau)


def sic_stats(x: Tensor, w: Tensor, tau: float | None = None,
              edge_jump: float = EDGE_JUMP, e0: float = ANISO_E0) -> Tensor:
    """The 8 SIC channels of S_C (h_ow .. aniso_s) of a 1 km field x with
    weights w (ocean fraction x observed), [..., 8, ny/32, nx/32]. With tau,
    the indicators are sigmoids of width tau: the soft S used by L_cyc."""
    f = g.factor("H", "C")

    def mean(field):
        return g.restrict(field, w, f)[0]

    lo = (None,) + SIC_BIN_EDGES
    hi = SIC_BIN_EDGES + (None,)
    hist = [mean((_step(x - a, tau) if a is not None else 1) * (_step(b - x, tau) if b is not None else 1)
                 * torch.ones_like(x)) for a, b in zip(lo, hi)]
    m1 = mean(x)
    var = (mean(x * x) - m1 * m1).clamp_min(0)
    sic_std = var.sqrt() if tau is None else (var + 1e-8).sqrt()  # finite gradient at 0

    # neighbour differences, assigned to the left / upper pixel; pairs
    # straddling two C cells are dropped
    pad = torch.nn.functional.pad
    inner_x = (torch.arange(x.shape[-1], device=x.device) % f) != f - 1
    inner_y = (torch.arange(x.shape[-2], device=x.device) % f) != f - 1
    gx = pad(x[..., :, 1:] - x[..., :, :-1], (0, 1))
    gy = pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    wx = pad(w[..., :, 1:] * w[..., :, :-1], (0, 1)) * inner_x
    wy = pad(w[..., 1:, :] * w[..., :-1, :], (0, 0, 0, 1)) * inner_y[:, None]
    sx, sy = g.block_sum(wx, f), g.block_sum(wy, f)
    edge = _div(g.block_sum(wx * _step(gx.abs() - edge_jump, tau), f)
                + g.block_sum(wy * _step(gy.abs() - edge_jump, tau), f), sx + sy)
    # structure tensor from the gradients at the corners of each 2 x 2 block:
    # unlike forward differences sharing a pixel, unbiased (J_xy = 0) for
    # uncorrelated noise
    a, b, c, d = x[..., :-1, :-1], x[..., :-1, 1:], x[..., 1:, :-1], x[..., 1:, 1:]
    cx, cy = pad(0.5 * (b - a + d - c), (0, 1, 0, 1)), pad(0.5 * (c - a + d - b), (0, 1, 0, 1))
    wc = pad(w[..., :-1, :-1] * w[..., :-1, 1:] * w[..., 1:, :-1] * w[..., 1:, 1:], (0, 1, 0, 1))
    wc = wc * inner_x * inner_y[:, None]
    sc = g.block_sum(wc, f)
    jxx, jyy, jxy = (_div(g.block_sum(wc * p, f), sc) for p in (cx * cx, cy * cy, cx * cy))
    aniso_c = (jxx - jyy) / (jxx + jyy + e0)
    aniso_s = 2 * jxy / (jxx + jyy + e0)
    return torch.stack(hist + [sic_std, edge, aniso_c, aniso_s], -3)


def sit_std_stat(sit_m: Tensor, w_m: Tensor) -> tuple[Tensor, Tensor]:
    """sit_std of S_C from a 4 km SIT field with weights w_m; and the number of
    M cells used."""
    fm = g.factor("M", "C")
    vm = (w_m > 0).to(sit_m.dtype)
    sm = sit_m * vm
    mean = g.restrict(sm, vm, fm)[0]
    return (g.restrict(sm * sm, vm, fm)[0] - mean ** 2).clamp_min(0).sqrt(), g.block_sum(vm, fm)


def sampling_std(s: Tensor) -> Tensor:
    """Standard error of each S_C channel for one independent sample, from
    its value s [..., 9, ny, nx] (see S_SIGMA0)."""
    p = s[..., :4, :, :].clamp(0, 1)
    e = s[..., 5:6, :, :].clamp(0, 1)
    a2 = (s[..., 6:7, :, :] ** 2 + s[..., 7:8, :, :] ** 2).clamp(max=1)
    return torch.cat([(p * (1 - p)).sqrt(), s[..., 4:5, :, :] / 2 ** 0.5, (e * (1 - e)).sqrt(),
                      1 - a2, 1 - a2, s[..., 8:9, :, :] / 2 ** 0.5], -3)


def subgrid_stats(sic_h: Tensor, w_h: Tensor, omega_h: Tensor,
                  sit_m: Tensor, w_m: Tensor, omega_m: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """S^_C = S_C(X^_H) from the target SIC at H (and the target SIT at M for
    sit_std). Inputs [..., ny, nx]; returns (S, mask, sigma_S), each
    [..., 9, ny/32, nx/32], in physical units, 0 where masked."""
    f = g.factor("H", "C")
    valid = (w_h > 0) & (omega_h > 0)
    w = omega_h * valid
    x = torch.where(valid, sic_h, torch.zeros_like(sic_h))
    sit_std, n_m = sit_std_stat(sit_m, ((w_m > 0) & (omega_m > 0)).to(sit_m.dtype))

    cov, w_sum = g.restrict(valid.to(x.dtype), omega_h, f)[0], g.block_sum(w, f)
    s = torch.cat([sic_stats(x, w), sit_std[..., None, :, :]], -3)
    n_eff = torch.stack([w_sum / ASIP_PIX_PER_SAMPLE] * 8 + [n_m], -3)
    mask = torch.stack([cov >= GAMMA_S] * 8 + [n_m >= MIN_SIT_OBS], -3)
    s0 = torch.tensor(S_SIGMA0, dtype=x.dtype, device=x.device)[:, None, None]
    sigma = (s0 ** 2 + sampling_std(s) ** 2 / n_eff.clamp_min(1e-6)).sqrt()
    zero = torch.zeros_like(s)
    return torch.where(mask, s, zero), mask, torch.where(mask, sigma, zero)
