"""Phase 2 tests for the on-the-fly observation operators (gridding,
partition, detail targets, sub-grid statistics) and the Natural Earth ocean
fraction. Run from the repo root, e.g.:

    pytest tests/test_obs.py
"""
import numpy as np
import torch

from croscim import grid as g
from croscim import obs
from croscim.ocean import land_fraction_h

TOL = 1e-12
f64 = dict(dtype=torch.float64)


def _omega(shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    w = torch.rand(shape, generator=gen, **f64)
    w[w < 0.2] = 0.0
    w[w > 0.6] = 1.0
    return w


# ---------------------------------------------------------------- gridding

def test_overlap_matrix_conserves_area():
    # 5 km cells onto a 32 km C tile (y decreasing): every source cell fully
    # inside the tile is split with total weight 1, none is lost.
    y_edges, x_edges = g.tile_edges("C", 10, 20, 4)
    xc = x_edges[0] - 7_500 + 5_000 * torch.arange(30, **f64)
    a = obs.overlap_matrix(xc, 5_000.0, x_edges)
    inside = (xc - 2_500 >= x_edges[0]) & (xc + 2_500 <= x_edges[-1])
    assert torch.allclose(a.sum(0)[inside], torch.ones(int(inside.sum()), **f64))
    yc = y_edges[0] + 2_500 - 5_000 * torch.arange(30, **f64)
    ay = obs.overlap_matrix(yc, 5_000.0, y_edges)
    assert torch.allclose(ay.sum(), torch.tensor(4 * 32 / 5, **f64))


def test_grid_dense_constant_and_sigma_weighting():
    y_edges, x_edges = g.tile_edges("C", 0, 0, 2)
    c = 0.5 * (x_edges[0] + x_edges[-1]) + 5_000 * (torch.arange(16, **f64) - 7.5)
    ax = obs.overlap_matrix(c, 5_000.0, x_edges)
    ay = obs.overlap_matrix(0.5 * (y_edges[0] + y_edges[-1]) - (c - c.mean()), 5_000.0, y_edges)
    v = torch.full((16, 16), 3.0, **f64)
    v[::3] = float("nan")
    m, w = obs.grid_dense(v, torch.tensor(0.5, **f64), ay, ax)
    assert torch.allclose(m, torch.full_like(m, 3.0)) and torch.all(w > 0)
    m2, _ = obs.combine((m, w), (torch.full_like(m, 6.0), 3 * w))
    assert torch.allclose(m2, torch.full_like(m, 5.25))


def test_grid_points_mean_per_cell_and_group():
    y_edges, x_edges = g.tile_edges("M", 0, 0, 1)  # 8 x 8 M cells
    x = x_edges[0] + torch.tensor([1_000.0, 3_000.0, 1_000.0, 9_000.0, -1.0], **f64)
    y = y_edges[0] - torch.tensor([1_000.0, 2_000.0, 1_000.0, 1_000.0, 1_000.0], **f64)
    v = torch.tensor([1.0, 3.0, 10.0, 7.0, 99.0], **f64)
    w = torch.tensor([1.0, 3.0, 1.0, 1.0, 1.0], **f64)
    grp = torch.tensor([0, 0, 1, 0, 0])
    m, den = obs.grid_points(x, y, v, w, grp, 2, y_edges, x_edges)
    assert m.shape == (2, 8, 8)
    assert m[0, 0, 0] == 2.5 and den[0, 0, 0] == 4.0      # (1*1 + 3*3) / 4
    assert m[1, 0, 0] == 10.0 and m[0, 0, 2] == 7.0
    assert den.sum() == 6.0                                # out-of-tile point dropped


def test_grid_asip_weighted_block_mean():
    sic = torch.tensor([[20, 40], [255, 80]], dtype=torch.uint8)
    std = torch.tensor([[10, 20], [0, 0]], dtype=torch.uint8)
    keep = torch.tensor([[True, True], [True, False]])
    m, w = obs.grid_asip(sic, std, keep, 0.05)
    w1, w2 = 0.1 ** -2, 0.2 ** -2
    assert torch.allclose(m, torch.tensor([[(0.2 * w1 + 0.4 * w2) / (w1 + w2)]]))
    assert torch.allclose(w, torch.tensor([[w1 + w2]]))


# --------------------------------------------------------------- partition

def test_unit_partition_keeps_units_whole():
    gen = torch.Generator().manual_seed(0)
    unit = torch.randint(0, 20, (64, 64))
    valid = torch.rand(64, 64, generator=gen) > 0.3
    tgt = obs.unit_partition(unit, valid, 0.4, gen)
    assert not tgt[~valid].any()
    for u in unit[valid].unique():
        sel = tgt[valid & (unit == u)]
        assert sel.all() or not sel.any()
    assert 0 < tgt.sum() < valid.sum()


# ----------------------------------------------------------------- targets

def test_detail_target_has_zero_restriction():
    omega = _omega((64, 64))
    gen = torch.Generator().manual_seed(3)
    x = torch.rand(64, 64, generator=gen, **f64)
    w_obs = (torch.rand(64, 64, generator=gen, **f64) > 0.4).double()
    w_obs[:8, :8] = 0  # an unobserved parent
    d, m = obs.detail_target(x, w_obs, omega, 8, 0.5)
    rd, _ = g.restrict(d, omega * m, 8)
    assert rd.abs().max() < TOL
    cov, _ = g.restrict(((w_obs > 0) & (omega > 0)).double(), omega, 8)
    assert torch.equal(g.block_sum(m.double(), 8) > 0, cov >= 0.5)
    assert not m[:8, :8].any() and torch.all(d[~m] == 0)
    assert torch.all(w_obs[m] > 0)  # never a target where nothing is observed


def test_subgrid_stats_on_simple_fields():
    n = 64  # 2 x 2 C cells
    omega_h = torch.ones(n, n, **f64)
    sic = torch.ones(n, n, **f64)
    sic[:32, 32:][:, ::4] = 0.0          # C cell (0, 1): open leads every 4 km along y
    sic[32:, :32] = 0.6                  # C cell (1, 0): uniform 0.6
    w_h = torch.ones(n, n, **f64)
    w_h[32:, 32:] = 0.0                  # C cell (1, 1): unobserved
    sit_m = torch.zeros(16, 16, **f64)
    sit_m[:8, :8] = torch.arange(64, **f64).view(8, 8) % 2  # SIT 0 / 1 m in C cell (0, 0)
    w_m = torch.zeros(16, 16, **f64)
    w_m[:8, :8] = 1.0
    s, m, sig = obs.subgrid_stats(sic, w_h, omega_h, sit_m, w_m, torch.ones(16, 16, **f64))
    assert s.shape == m.shape == sig.shape == (9, 2, 2)
    hist = s[:4]
    assert torch.allclose(hist[:, m[0]].sum(0), torch.ones(int(m[0].sum()), **f64))
    assert torch.allclose(hist[:, 0, 0], torch.tensor([0, 0, 0, 1.0], **f64))
    assert torch.allclose(hist[:, 1, 0], torch.tensor([0, 0, 1.0, 0], **f64))
    assert torch.allclose(hist[0, 0, 1], torch.tensor(0.25, **f64))
    # uniform cells: no texture, no structure
    assert torch.all(s[4:8, 0, 0] == 0) and torch.all(s[4:8, 1, 0] == 0)
    # leads along y: 15 jumps over 31 x-pairs per row, none over the y-pairs;
    # x-gradients only -> anisotropy close to +1, orientation term 0
    assert torch.allclose(s[5, 0, 1], torch.tensor(15 / 62, **f64)) and s[6, 0, 1] > 0.95 and s[7, 0, 1].abs() < TOL
    assert not m[:8, 1, 1].any()
    assert m[8, 0, 0] and torch.allclose(s[8, 0, 0], torch.tensor(0.5, **f64))
    assert not m[8, 0, 1]
    assert torch.all(sig[m] > 0)


# ------------------------------------------------------------ ocean frac.

def test_land_fraction_of_a_square_with_a_hole():
    # 10 x 10 km square island with a 2 x 2 km lake (edges on cell edges:
    # exact), plus a 1 km right triangle: 6 of the 16 sub-pixel centres
    # (supersample 4) are inside it.
    x0, y0 = 0.0, 0.0
    sq = np.array([[x0, y0], [x0 + 10e3, y0], [x0 + 10e3, y0 + 10e3], [x0, y0 + 10e3], [x0, y0]])
    lake = np.array([[2e3, 2e3], [4e3, 2e3], [4e3, 4e3], [2e3, 4e3], [2e3, 2e3]])
    tri = np.array([[-20e3, -20e3], [-19e3, -20e3], [-20e3, -19e3], [-20e3, -20e3]])
    land = land_fraction_h([sq, lake, tri], supersample=4)
    assert land.shape == (6400, 6400)
    assert land.sum() == 100 - 4 + 6 / 16
    iy, ix, _ = g.cell_index(torch.tensor([5e3, 3e3]), torch.tensor([5e3, 3e3]), "H")
    assert land[iy[0], ix[0]] == 1.0 and land[iy[1], ix[1]] == 0.0


def test_anisotropy_is_unbiased_for_uncorrelated_noise():
    gen = torch.Generator().manual_seed(5)
    x = torch.rand(256, 256, generator=gen, **f64)
    s = obs.sic_stats(x, torch.ones_like(x), e0=1e-4)
    assert s[6].abs().mean() < 0.05 and s[7].abs().mean() < 0.05
