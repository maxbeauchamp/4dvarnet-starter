"""Phase 1 tests for the nested C/M/H grid and the fixed scale operators
R, P, Pi0 (CROSCIM_FM_cascade/formulation, section 1). Run from the repo
root, e.g.:

    pytest tests/test_grid.py
"""
import pytest
import torch

from croscim import grid as g

TOL = 1e-15
_F64 = dict(dtype=torch.float64)


def _ocean_weights(shape, seed=0):
    """Ocean fraction in [0, 1] with fully-land cells, coastal (partial)
    cells and one fully-land C parent (top-left 32x32 H block)."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.rand(shape, generator=gen, dtype=torch.float64)
    w[w < 0.2] = 0.0
    w[w > 0.8] = 1.0
    w[:32, :32] = 0.0
    return w


def _field(shape, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return 3.0 * torch.randn(shape, generator=gen, dtype=torch.float64) + 1.0


# ---------------------------------------------------------------- geometry

def test_level_shapes_and_factors():
    assert g.shape("C") == (200, 200)
    assert g.shape("M") == (1600, 1600)
    assert g.shape("H") == (6400, 6400)
    assert g.factor("M", "C") == 8
    assert g.factor("H", "M") == 4
    assert g.factor("H", "C") == 32


def test_edges_are_nested_and_cover_the_domain():
    ex = {lvl: g.edges(lvl) for lvl in g.LEVELS}
    for lvl in g.LEVELS:
        assert ex[lvl][0] == -g.HALF_EXTENT_M and ex[lvl][-1] == g.HALF_EXTENT_M
    assert torch.equal(ex["M"][::8], ex["C"])
    assert torch.equal(ex["H"][::4], ex["M"])


def test_h_edges_are_asip_500m_edges():
    # ASIP native 500 m cells have edges on multiples of 500 m (centres at
    # +-250 m, e.g. xc=-3849750): each 1 km cell is an exact 2x2 ASIP block.
    assert torch.all(torch.remainder(g.edges("H"), 500) == 0)


def test_centers_y_axis_points_down():
    # Row 0 is the top (largest y), as in the existing CROSCIM grids.
    x, y = g.centers("C")
    assert x[0] < x[-1] and y[0] > y[-1]
    assert x[0] == -g.HALF_EXTENT_M + 16_000 and y[0] == g.HALF_EXTENT_M - 16_000


def test_cell_index_is_consistent_across_levels():
    gen = torch.Generator().manual_seed(2)
    x = (torch.rand(10_000, generator=gen, dtype=torch.float64) * 2 - 1) * g.HALF_EXTENT_M
    y = (torch.rand(10_000, generator=gen, dtype=torch.float64) * 2 - 1) * g.HALF_EXTENT_M
    iy, ix, inside = {}, {}, {}
    for lvl in g.LEVELS:
        iy[lvl], ix[lvl], inside[lvl] = g.cell_index(x, y, lvl)
        assert inside[lvl].all()
    assert torch.equal(iy["H"] // 32, iy["C"]) and torch.equal(ix["H"] // 32, ix["C"])
    assert torch.equal(iy["H"] // 4, iy["M"]) and torch.equal(ix["H"] // 4, ix["M"])
    # The point lands inside the cell it is assigned to.
    cx, cy = g.centers("H")
    assert torch.all((x - cx[ix["H"]]).abs() <= 500) and torch.all((y - cy[iy["H"]]).abs() <= 500)


def test_cell_index_flags_points_outside_the_domain():
    x = torch.tensor([0.0, 3_300_000.0, -3_200_000.0, 3_200_000.0])
    y = torch.tensor([0.0, 0.0, 0.0, 0.0])
    _, _, inside = g.cell_index(x, y, "C")
    assert inside.tolist() == [True, False, True, False]


def test_tile_slices_are_nested():
    sl = g.tile_slices(iy0=3, ix0=5, n_c=4)
    assert sl["C"] == (slice(3, 7), slice(5, 9))
    assert sl["M"] == (slice(24, 56), slice(40, 72))
    assert sl["H"] == (slice(96, 224), slice(160, 288))


# --------------------------------------------------------------- operators

@pytest.mark.parametrize("f", [4, 8, 32])
def test_RP_is_identity_on_ocean_parents(f):
    w = _ocean_weights((128, 128))
    xc = _field((128 // f, 128 // f))
    rpx, wp = g.restrict(g.prolong(xc, f), w, f)
    ocean = wp > 0
    assert torch.allclose(rpx[ocean], xc[ocean], rtol=0, atol=TOL * xc.abs().max())
    assert torch.all(rpx[~ocean] == 0)


@pytest.mark.parametrize("f", [4, 8])
def test_R_of_detail_vanishes(f):
    w = _ocean_weights((128, 128))
    x = _field((2, 3, 128, 128))
    d = g.pi0(x, w, f)
    rd, _ = g.restrict(d, w, f)
    assert rd.abs().max() < TOL * x.abs().max()


def test_pi0_is_a_projector():
    w = _ocean_weights((128, 128))
    x = _field((128, 128))
    d = g.pi0(x, w, 8)
    assert torch.allclose(g.pi0(d, w, 8), d, rtol=0, atol=TOL * x.abs().max())


def test_restriction_composes():
    # R_{C<-M} R_{M<-H} = R_{C<-H}, provided the M weights are the summed
    # H weights returned by restrict.
    w = _ocean_weights((128, 128))
    x = _field((128, 128))
    xm, wm = g.restrict(x, w, 4)
    xc_two_steps, wc_two_steps = g.restrict(xm, wm, 8)
    xc_direct, wc_direct = g.restrict(x, w, 32)
    assert torch.allclose(xc_two_steps, xc_direct, rtol=0, atol=TOL * x.abs().max())
    assert torch.allclose(wc_two_steps, wc_direct, rtol=0, atol=TOL * wc_direct.max())


def test_decompose_recompose_is_exact():
    w = _ocean_weights((128, 128))
    x = _field((2, 128, 128))
    x_c, dx_m, dx_h = g.decompose(x, w)
    assert x_c.shape == (2, 4, 4) and dx_m.shape == (2, 32, 32) and dx_h.shape == (2, 128, 128)
    assert torch.allclose(g.recompose(x_c, dx_m, dx_h, w), x, rtol=0, atol=TOL * x.abs().max())
    _, w_m = g.restrict(x, w, 4)
    assert g.restrict(dx_h, w, 4)[0].abs().max() < TOL * x.abs().max()
    assert g.restrict(dx_m, w_m, 8)[0].abs().max() < TOL * x.abs().max()


def test_tiled_restriction_matches_global():
    w = _ocean_weights((256, 256))
    x = _field((256, 256))
    x_c, _ = g.restrict(x, w, 32)
    sl = g.tile_slices(iy0=2, ix0=4, n_c=4)
    x_c_tile, _ = g.restrict(x[sl["H"]], w[sl["H"]], 32)
    assert torch.equal(x_c_tile, x_c[sl["C"]])


def test_weights_broadcast_over_leading_dims_float32():
    w = _ocean_weights((64, 64)).float()
    x = _field((2, 3, 5, 64, 64)).float()
    x_c, wp = g.restrict(x, w, 32)
    assert x_c.shape == (2, 3, 5, 2, 2) and wp.shape == (2, 2)
    assert g.restrict(g.pi0(x, w, 32), w, 32)[0].abs().max() < 1e-5 * x.abs().max()


@pytest.mark.parametrize("f", [4, 8])
def test_smooth_prolongation_is_exact_and_smooth(f):
    w = _ocean_weights((128, 128))
    n = 128 // f
    yy, xx = torch.meshgrid(torch.arange(n, **_F64), torch.arange(n, **_F64), indexing="ij")
    ramp = 0.5 * xx + 0.2 * yy
    p_s, p_0 = g.prolong_s(ramp, w, f), g.prolong(ramp, f)
    ocean = g.block_sum(w, f) > 0
    # R P' = I on ocean parents, whatever the land
    assert torch.allclose(g.restrict(p_s, w, f)[0][ocean], ramp[ocean], rtol=0, atol=TOL * ramp.abs().max())
    # no step at parent edges: along a row of open ocean, P' of a ramp is much
    # smoother than the injection
    sea = torch.ones_like(w)
    jump = lambda z: (z[..., :, 1:] - z[..., :, :-1]).abs().max()  # noqa: E731
    assert jump(g.prolong_s(ramp, sea, f)) < 0.5 * jump(g.prolong(ramp, f))
    assert torch.equal(p_0, g.prolong(ramp, f))
