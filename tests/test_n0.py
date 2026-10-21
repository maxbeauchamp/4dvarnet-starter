"""Phase 3-5 tests: encoders / pyramid operators and the N0 cascade
(LitCascadeN0) with tiny Swin backbones on a synthetic batch in the
`build_batch` format. Run from the repo root, e.g.:

    pytest tests/test_n0.py
"""
import pytest
import torch

from croscim import cascade as C
from croscim import data as D
from croscim import grid as g
from croscim import obs
from croscim.encoders import AsipHEncoder, Down, PointCellEncoder, Up

TINY = dict(base_dim=8, encoder_depths=[1, 1, 1, 1], decoder_depths=[1, 1, 1], num_heads=[1, 1, 1, 1],
            window_size=8, mlp_ratio=1.0, time_dim=16, modulation_rank=4, drop_path_rate=0.0,
            activation_checkpointing=False)
NORM = {**{k: [0.0, 1.0] for k in C.Z_NAMES}, "SIT": [0.57, 0.82], "HS": [0.155, 0.14],
        **{k: [0.0, 1.0] for k in C.FORCING}, "dSIC_M": [0.0, 0.1], "dSIT_M": [0.0, 0.3], "dSIC_H": [0.0, 0.1]}


def make_batch(bsz=1, seed=0):
    """Synthetic batch in the build_batch format (physical units)."""
    gen = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.rand(*shape, generator=gen)

    nm, nh = 512, 2048
    omega_h = (r(bsz, nh // 32, nh // 32) > 0.1).float().repeat_interleave(32, -1).repeat_interleave(32, -2)
    omega_m = g.restrict(omega_h, torch.ones_like(omega_h), 4)[0]
    omega_c = g.restrict(omega_m, torch.ones_like(omega_m), 8)[0]
    iy0, ix0 = torch.tensor([60] * bsz), torch.tensor([70] * bsz)
    n_pts = 500
    y_edges, x_edges = g.tile_edges("C", 60, 70, 64)
    pts = torch.stack([x_edges[0] + r(bsz, n_pts) * 2.048e6, y_edges[0] - r(bsz, n_pts) * 2.048e6,
                       r(bsz, n_pts) * 2, (r(bsz, n_pts) > 0.5).float(), -torch.floor(r(bsz, n_pts) * D.W)], -1).float()
    b = {"iy0": iy0, "ix0": ix0, "day": torch.tensor([19000] * bsz),
         "omega_h": omega_h, "omega_m": omega_m, "omega_c": omega_c,
         "in_H": r(bsz, D.W_H, nh, nh), "in_H_w": (r(bsz, D.W_H, nh, nh) > 0.6).float(),
         "in_M_asip": r(bsz, D.W, nm, nm), "in_M_asip_w": (r(bsz, D.W, nm, nm) > 0.5).float(),
         "in_C_cimr": r(bsz, D.W, 2, 64, 64), "in_C_cimr_w": (r(bsz, D.W, 2, 64, 64) > 0.3).float(),
         "pts": pts, "pts_valid": r(bsz, n_pts) > 0.1, "f_C": r(bsz, D.W, 4, 64, 64),
         "tgt_C": r(bsz, 3, 64, 64), "tgt_C_m": r(bsz, 3, 64, 64) > 0.7,
         "tgt_C_sigma": torch.cat([0.15 * (r(bsz, 1, 64, 64) > 0.5), torch.zeros(bsz, 2, 64, 64)], 1),
         "tgt_S": r(bsz, 9, 64, 64), "tgt_S_m": r(bsz, 9, 64, 64) > 0.5, "tgt_S_sigma": 0.05 + 0.1 * r(bsz, 9, 64, 64),
         "tgt_M": r(bsz, 2, nm, nm), "tgt_M_m": r(bsz, 2, nm, nm) > 0.5,
         "tgt_H": r(bsz, 1, nh, nh), "tgt_H_m": r(bsz, 1, nh, nh) > 0.5}
    b["f_C"][:, :, :, :5] = float("nan")
    for key, lvl, f in (("tgt_dM", "M", 8), ("tgt_dH", "H", 4)):
        x, w = b[f"tgt_{lvl}"], b[f"tgt_{lvl}_m"].float()
        om = b["omega_m"] if lvl == "M" else b["omega_h"]
        ds, ms = zip(*[obs.detail_target(x[:, c], w[:, c], om, f, 0.1) for c in range(x.shape[1])])
        b[key], b[f"{key}_m"] = torch.stack(ds, 1), torch.stack(ms, 1)
    return b


def make_model(**kw):
    torch.manual_seed(0)
    args = dict(norm=NORM, swin={"C": TINY, "M": TINY, "H": TINY}, d=8, d_psi=2, n_steps=1, cond_steps=1,
                n_members=2)
    args.update(kw)
    return C.LitCascadeN0(**args)


@pytest.fixture(scope="module")
def batch():
    return make_batch()


@pytest.fixture(scope="module")
def model():
    return make_model()


# ---------------------------------------------------------------- encoders

def test_point_cell_encoder_matches_per_cell_softmax():
    torch.manual_seed(0)
    enc = PointCellEncoder(4, 2, "C", 15)
    y_edges, x_edges = g.tile_edges("C", 10, 20, 2)
    # 3 points in cell (0, 0), 1 in cell (1, 1), one outside the tile, one invalid
    xs = x_edges[0] + torch.tensor([1e3, 5e3, 20e3, 40e3, 70e3, 3e3])
    ys = y_edges[0] - torch.tensor([1e3, 9e3, 30e3, 40e3, 1e3, 3e3])
    pts = torch.stack([xs, ys, torch.rand(6), torch.tensor([0., 1., 0., 1., 0., 0.]), -torch.arange(6.)], -1)[None].float()
    valid = torch.tensor([[True, True, True, True, True, False]])
    a, m = enc(pts, valid, torch.tensor([10]), torch.tensor([20]), 2)
    assert m[0].tolist() == [[True, False], [False, True]]
    assert torch.all(a[0, :, 0, 1] == 0) and torch.all(a[0, :, 1, 0] == 0)
    # brute force for cell (0, 0)
    feat = []
    for n in range(3):
        cx, cy = (xs[n] - x_edges[0]) / 32e3, (y_edges[0] - ys[n]) / 32e3
        feat.append(torch.cat([pts[0, n, 2:3], torch.nn.functional.one_hot(pts[0, n, 3].long(), 2).float(),
                               torch.stack([cx - cx.floor() - 0.5, cy - cy.floor() - 0.5]).float(), pts[0, n, 4:5] / 15]))
    e = enc.mlp(torch.stack(feat))
    att = torch.softmax(enc.wk(e) @ enc.q / 2.0, 0)
    assert torch.allclose(a[0, :, 0, 0], (att[:, None] * enc.wv(e)).sum(0), atol=1e-6)


def test_up_excludes_masked_children():
    torch.manual_seed(0)
    up = Up(4, 8, 3)
    h = torch.randn(1, 4, 16, 16)
    m = torch.rand(1, 16, 16) > 0.5
    m[:, :8, :8] = False  # parent (0, 0) has no observed child
    out, pm = up(h, m)
    h2 = torch.where(m[:, None], h, torch.randn_like(h) * 100)
    out2, _ = up(h2, m)
    assert torch.allclose(out, out2, atol=1e-5) and not pm[0, 0, 0] and torch.all(out[0, :, 0, 0] == 0)


def test_down_injects_parent_plus_child_position():
    down = Down(3, 4, 8)
    gp = torch.randn(2, 3, 2, 2)
    out = down(gp)
    assert out.shape == (2, 4, 16, 16)
    # same child position in two parents: difference is the parent term only
    lin = down.lin(gp)
    assert torch.allclose(out[:, :, 3, 5] - out[:, :, 11, 13], lin[:, :, 0, 0] - lin[:, :, 1, 1], atol=1e-5)


def test_asip_encoder_only_nonempty_cells():
    torch.manual_seed(0)
    enc = AsipHEncoder(2, 4)
    x = torch.randn(1, 2, 64, 64)
    mask = torch.zeros(1, 64, 64, dtype=torch.bool)
    mask[:, 3, 40] = True  # only C cell (0, 1) observed
    out, mm = enc(x, mask)
    assert out.shape == (1, 4, 16, 16) and mm.sum() == 1
    dense = enc.net(x)
    assert torch.allclose(out[..., :8, 8:], (dense * mm[:, None])[..., :8, 8:], atol=1e-6)
    assert torch.all(out[..., :, :8] == 0) and torch.all(out[..., 8:, :] == 0)


# ---------------------------------------------------------------------- N0

def test_losses_all_levels_backward(model, batch):
    out = model.losses(batch)
    for k in ("X_C", "S_C", "X_M", "X_H", "cyc", "psi", "loss"):
        assert torch.isfinite(out[k]), k
    model.zero_grad()
    out["loss"].backward()
    for name in ("gen.C", "gen.M", "gen.H", "e_pts_m", "e_asip_h", "up_cm", "psi", "down_h"):
        mods = dict(model.nets.named_modules())[name]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mods.parameters()), name


def test_phase3_configuration_c_only_without_pyramid(batch):
    m = make_model(levels=["C"], pyramid=False)
    out = m.losses(batch)
    assert set(out) == {"X_C", "S_C", "psi", "loss"} and torch.isfinite(out["loss"])


def test_phase4_does_not_train_the_c_side(batch):
    m = make_model(levels=["M", "H"], pyramid=False, corrector=True)
    m.losses(batch)["loss"].backward()
    mods = dict(m.nets.named_modules())
    for name in ("gen.C", "e_cimr", "e_pts_c", "e_f", "fuse_c", "psi"):
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in mods[name].parameters()), name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mods["gen.M"].parameters())


def test_sample_cascade_details_are_exact(model, batch):
    s = model.sample_cascade(batch)
    assert s["z_c"].shape == (1, 12, 64, 64) and s["d_m"].shape == (1, 2, 512, 512)
    assert s["x_h"].shape == (1, 1, 2048, 2048)
    w_m, w_h = batch["omega_m"][:, None], batch["omega_h"][:, None]
    assert g.restrict(s["d_m"], w_m, 8)[0].abs().max() < 1e-5
    assert g.restrict(s["d_h"], w_h, 4)[0].abs().max() < 1e-5
    # exact recomposition: R X_H = X_M, R X_M = X_C on ocean parents
    rm, wm = g.restrict(s["x_h"], w_h, 4)
    assert ((rm - s["x_m"][:, :1]).abs() * (wm > 0)).max() < 1e-5
    rc, wc = g.restrict(s["x_m"], w_m, 8)
    assert ((rc - s["z_c"][:, :2]).abs() * (wc > 0)).max() < 1e-5


def test_corrector_losses_and_bounded_samples(batch):
    m = make_model(corrector=True)
    out = m.losses(batch)
    for k in ("X_C", "X_M", "X_H", "cyc", "loss"):
        assert torch.isfinite(out[k]), k
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.nets.gen["M"].parameters())
    s = m.sample_cascade(batch)
    # corrections censored at the physical bounds of the corrected fields
    assert s["x_m"][:, 0].min() >= -1e-5 and s["x_m"][:, 0].max() <= 1 + 1e-5 and s["x_m"][:, 1].min() >= -1e-5
    assert s["x_h"].min() >= -1e-5 and s["x_h"].max() <= 1 + 1e-5
    # no R dX = 0 constraint: the corrections may change the coarse means
    assert g.restrict(s["d_m"], batch["omega_m"][:, None], 8)[0].abs().max() > 1e-3


def test_weak_cimr_targets_are_not_scored(batch):
    m = C.strict_mask(batch)
    assert not (m[:, 0] & (batch["tgt_C_sigma"][:, 0] > 0)).any() and m[:, 0].any()


def test_paired_noise_is_reproducible(model, batch):
    keep = model.keep_groups(1, "cpu", drop=False)
    ctx = model.encode(model.ema.module, batch)
    a = model.sample_c(model.ema.module, ctx, batch, keep, 1, generator=torch.Generator().manual_seed(3))
    b = model.sample_c(model.ema.module, ctx, batch, keep, 1, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_scores_keys(model, batch):
    out = model.scores(batch)
    assert {"C_rmse_SIC", "M_crps_SIT", "H_rmse_SIC"} <= set(out)
    assert set(model._ranks) == {"C", "M", "H"}
