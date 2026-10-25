"""Phase 6-7 tests: the transition network N1 (LitCascadeN1), mode A / B
rollouts, interventions and diagnostics, with tiny backbones on synthetic
batches. Run from the repo root, e.g.:

    pytest tests/test_n1.py
"""
import pytest
import torch

from croscim import rollout as R
from croscim.cascade import N_X, N_Z, Z_NAMES
from croscim.transition import LitCascadeN1, only_last_day
from tests.test_n0 import TINY, make_batch, make_model


@pytest.fixture(scope="module")
def days():
    return [make_batch(seed=k) for k in range(3)]


@pytest.fixture(scope="module")
def n1():
    torch.manual_seed(0)
    return LitCascadeN1(n0=make_model(), swin=TINY, k_max=2, n_steps=1, push_steps=1, n_members=2)


def test_init_from_n0(n1):
    assert n1.nets.n_loaded > 0  # G_C^0 weights reused wherever shapes match
    assert not any(p.requires_grad for p in n1.nets.enc.parameters())
    assert not any(p.requires_grad for p in n1.n0.parameters())


def test_only_last_day_masks_the_window(days):
    b = only_last_day(days[1])
    assert b["in_C_cimr_w"][:, :-1].sum() == 0 and b["in_H_w"][:, :-1].sum() == 0
    assert torch.all(b["pts"][..., 4][b["pts_valid"]] == 0)
    assert torch.isnan(b["f_C"][:, :-2]).all() and torch.equal(b["f_C"][:, -1:].isnan(), days[1]["f_C"][:, -1:].isnan())


def test_pushforward_loss_backward(n1, days, monkeypatch):
    monkeypatch.setattr(n1, "k_now", lambda n: 2)
    out = n1.losses(days)
    assert {"X_k1", "S_k1", "psi_k1", "X_k2"} <= set(out) and torch.isfinite(out["loss"])
    n1.zero_grad()
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in n1.nets.gen.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in n1.nets.e_state.parameters())


def test_rollout_mode_a_paired_noise_and_guidance(n1, days):
    z1, psi = R.rollout(n1, days, 2, w=1.0, seed=4)
    z2, _ = R.rollout(n1, days, 2, w=1.0, seed=4)
    z0, _ = R.rollout(n1, days, 2, w=0.0, seed=4)
    assert z1.shape == (1, 3, N_Z, 64, 64) and psi.shape == (1, 3, n1.d_psi, 64, 64)
    assert torch.equal(z1, z2)
    assert torch.equal(z1[:, 0], z0[:, 0]) and not torch.equal(z1[:, 1], z0[:, 1])
    hist = z1[:, :, Z_NAMES.index("h_ow"):Z_NAMES.index("h_pack") + 1].sum(2)
    assert torch.allclose(hist, torch.ones_like(hist), atol=1e-5)


def test_prescription_holds(n1, days):
    names = ["edge_density"]
    s_star = torch.full((1, 1, 64, 64), 0.3)
    z, _ = R.rollout(n1, days, 2, seed=1, intervention=R.prescribe(lambda k: s_star, names))
    assert torch.allclose(z[:, 1:, Z_NAMES.index("edge_density")], torch.full((1, 2, 64, 64), 0.3))


def test_mode_b_replaces_s_by_fine_statistics(n1, days):
    za, _ = R.rollout(n1, days, 1, seed=2, mode="A")
    zb, _ = R.rollout(n1, days, 1, seed=2, mode="B")
    assert torch.equal(za[:, 1, :N_X], zb[:, 1, :N_X]) and not torch.equal(za[:, 1, N_X:], zb[:, 1, N_X:])


def test_diagnostics_run(n1, days):
    gaps = R.loss_gaps(n1, days)
    assert set(gaps) == {"I_up", "I_down_f", "I_down_X"} and all(torch.isfinite(v) for v in gaps.values())
    resp = R.response(n1, days, torch.tensor(0.1), ["edge_density"], 2, n_members=2)
    assert resp.shape == (1, 3, N_X, 64, 64) and torch.all(resp[:, 0] == 0)
    z, psi = R.rollout(n1, days, 0, seed=0)
    slope = R.controllability(n1.n0, days[0], z[:, 0], psi[:, 0], "sic_std", 0.05)
    assert torch.isfinite(slope)
    cmp = R.compare_modes(n1, days, 1)
    assert len(cmp) == 1 and cmp[0]["k"] == 1


def test_forcing_scenario(days):
    out = R.modify_forcing(days, d_t2m=2.0, rotate_deg=90.0)
    f0, f1 = days[0]["f_C"], out[0]["f_C"]
    ok = torch.isfinite(f0[:, :, 0])
    assert torch.allclose((f1[:, :, 0] - f0[:, :, 0])[ok], torch.tensor(2.0))
    assert torch.allclose(f1[:, :, 2][ok], -f0[:, :, 3][ok], atol=1e-6)


def test_frozen_and_ema_copies_stay_in_eval_mode(n1):
    n1.train()
    assert n1.nets.training and not n1.ema.training and not n1.n0.training and not n1.n0.ema.training
    assert not any(m.training for m in n1.n0.modules()) and not any(m.training for m in n1.ema.modules())


def test_mixed_precision_bf16(days):
    # trainer precision bf16-mixed / overfit.py: everything under autocast
    torch.manual_seed(0)
    n0 = make_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = n0.losses(days[0])
        out["loss"].backward()
        assert torch.isfinite(out["loss"])
        assert all(torch.isfinite(v) for v in n0.scores(days[0]).values())
        n1 = LitCascadeN1(n0=n0, swin=TINY, k_max=2, n_steps=1, push_steps=1, n_members=2)
        assert torch.isfinite(n1.losses(days)["loss"])
        z, _ = R.rollout(n1, days, 1, seed=0)
        assert torch.isfinite(z).all()


def test_batch_without_any_target_still_backpropagates(n1, days, monkeypatch):
    monkeypatch.setattr(n1, "k_now", lambda n: 1)
    empty = [dict(d) for d in days]
    for d in empty:
        for k in ("tgt_C_m", "tgt_S_m", "tgt_dM_m", "tgt_dH_m", "tgt_M_m", "tgt_H_m"):
            d[k] = torch.zeros_like(d[k])
        d["in_C_cimr_w"], d["in_M_asip_w"], d["in_H_w"] = (torch.zeros_like(d[k]) for k in
                                                           ("in_C_cimr_w", "in_M_asip_w", "in_H_w"))
        d["pts_valid"] = torch.zeros_like(d["pts_valid"])
    out = n1.losses(empty)
    assert out["loss"] == 0 and out["loss"].requires_grad
    out["loss"].backward()


# ----------------------------------------------------- N1b (increment, state_input, raw_obs)

@pytest.fixture(scope="module")
def n1b():
    torch.manual_seed(0)
    return LitCascadeN1(n0=make_model(), swin=TINY, k_max=2, n_steps=1, push_steps=1, n_members=2,
                        increment=True, state_input=True, raw_obs=True)


def test_n1b_loss_backward(n1b, days, monkeypatch):
    monkeypatch.setattr(n1b, "k_now", lambda n: 2)
    out = n1b.losses(days)
    assert torch.isfinite(out["loss"])
    n1b.zero_grad()
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in n1b.nets.gen.parameters())


def test_n1b_zero_increment_is_persistence_and_bounds_hold(n1b, days, monkeypatch):
    nets = n1b.ema.module
    z, psi = n1b.analysis(days[0], 1)
    keep = n1b.keep(1, "cpu", train=False)
    real = nets.gen.sample
    monkeypatch.setattr(nets.gen, "sample", lambda cond, valid, n, **kw: torch.zeros(1, N_Z + n1b.d_psi, 64, 64))
    z1, psi1 = n1b.step(nets, days[1], z, psi, keep, 1)
    assert torch.allclose(z1, z, atol=1e-5) and torch.allclose(psi1, psi, atol=1e-5)
    monkeypatch.setattr(nets.gen, "sample", real)
    z1, _ = n1b.step(nets, days[1], z, psi, keep, 1)
    sic = z1[:, 0]
    assert sic.min() >= -1e-4 and sic.max() <= 1 + 1e-4 and z1[:, 1].min() >= -1e-4  # SIC in [0, 1], SIT >= 0
