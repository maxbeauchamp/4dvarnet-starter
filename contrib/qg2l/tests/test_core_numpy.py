"""
Numpy-only tests of the QG testbed core (no torch / hydra / xarray needed).

Run from the repository root:
    python -m pytest contrib/qg2l/tests -q
(the conftest puts ``contrib`` on sys.path so that ``qg2l`` is importable without
triggering ``contrib/__init__.py``).
"""
import math

import numpy as np
import pytest

from qg2l.qgmodel import QG2L, spectral_coarsen, isotropic_spectrum
from qg2l import obs as O
from qg2l import lyapunov as LY
from qg2l import metrics as M
from qg2l.fourdvar import isotropic_layer_covariance, sqrt_cov_field
from qg2l.simulate import model_kwargs


# ---------------------------------------------------------------------------
# QG model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spun_up():
    m = QG2L(**model_kwargs("energetic", nx=32, dt=7200.0, L=5e5))
    p = m.random_init(0, amp=1e-5)
    for _ in range(400):
        p = m.forecast_days(p, 1)
    return m, p


def test_inversion_roundtrip():
    m = QG2L(nx=32)
    rng = np.random.default_rng(0)
    q = rng.standard_normal((2, 32, 32))
    q -= q.mean(axis=(-1, -2), keepdims=True)
    assert np.allclose(m.psi_to_q(m.q_to_psi(q)), q, atol=1e-10)


def test_energy_quadratic_form():
    m = QG2L(nx=32)
    p = m.random_init(1, amp=1.0)
    z = LY.energy_coords(m, p)
    assert np.isclose(0.5 * z @ z, m.energy(p), rtol=1e-10)


def test_spun_up_is_turbulent_and_stable(spun_up):
    m, p = spun_up
    assert np.isfinite(p).all()
    assert m.energy(p) > 1e-4
    assert m.cfl(p) < 0.8


def test_spectral_coarsen_preserves_large_scales():
    m = QG2L(nx=64)
    p = m.random_init(2, amp=1.0)
    pc = spectral_coarsen(p, 32)
    k1, s1 = isotropic_spectrum(p[0], m.dx)
    k2, s2 = isotropic_spectrum(pc[0], 2 * m.dx)
    assert np.allclose(s1[:10], s2[:10], rtol=1e-6)


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
def test_observation_sampling():
    nx, L, n_days = 64, 1e6, 42
    ssh = np.random.default_rng(0).standard_normal((n_days, nx, nx))
    ob = O.build_observations(ssh, L, lat=35.0, seed=0)
    seen = ob["sat_id"] >= 0
    assert np.array_equal(seen, np.isfinite(ob["obs"]))
    assert np.array_equal(seen, np.isfinite(ob["obs_sigma"]))
    cov = O.coverage_stats(ob["sat_id"], ob["names"])
    assert cov["swot"] > 3 * cov["jason3"]  # wide swath
    # every nadir satellite completes its repeat cycle -> each track seen once per cycle
    for i, name in enumerate(ob["names"]):
        assert (ob["sat_id"] == i).any(), name
    # noise level matches the specification
    for i, name in enumerate(ob["names"]):
        e = (ob["obs"] - ssh)[ob["sat_id"] == i]
        assert abs(e.std() - O.DEFAULT_CONSTELLATION[name]["noise"]) < 0.2 * O.DEFAULT_CONSTELLATION[name]["noise"]


# ---------------------------------------------------------------------------
# Lyapunov machinery, validated on Lorenz-96 (n=40, F=8)
# ---------------------------------------------------------------------------
class L96:
    """Adapter exposing the QG2L interface used by lyapunov.py: state (1, 1, n)."""

    class xp:
        name = "numpy"
        asarray = staticmethod(lambda x, cplx=False: np.asarray(x, float))
        stack = staticmethod(lambda xs, axis=0: np.stack(xs, axis=axis))

    def __init__(self, n=40, F=8.0, dt=0.01, day=0.2):
        self.n, self.F, self.dt, self.nsteps = n, F, dt, int(round(day / dt))

    def f(self, x):
        return (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + self.F

    def forecast_days(self, x, n_days=1):
        for _ in range(n_days * self.nsteps):
            k1 = self.f(x)
            k2 = self.f(x + 0.5 * self.dt * k1)
            k3 = self.f(x + 0.5 * self.dt * k2)
            k4 = self.f(x + self.dt * k3)
            x = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def energy_components(self, x):
        return x  # Euclidean norm, (..., 1, 1, n)


@pytest.fixture(scope="module")
def l96_run():
    m = L96()
    x = 8 + np.random.default_rng(0).standard_normal((1, 1, 40))
    x = m.forecast_days(x, 100)
    traj = [x]
    for _ in range(400):
        traj.append(m.forecast_days(traj[-1], 1))
    traj = np.stack(traj)
    rng = np.random.default_rng(1)
    errors = dict(noise=0.1 * rng.standard_normal(traj.shape))
    out = LY.lyapunov_vectors(m, traj, k=20, n_transient_fwd=100, n_transient_bwd=100, errors=errors,
                              snapshot_index=[150], verbose=False)
    return m, traj, out


def test_l96_exponents(l96_run):
    _, _, out = l96_run
    lam = out["exponents"] / 0.2  # per model time unit
    assert 1.4 < lam[0] < 2.0  # literature: ~1.69
    n_pos = (lam > 0.05).sum()
    assert 11 <= n_pos <= 15  # literature: 13
    assert np.all(np.diff(lam) < 0.1)  # (almost) ordered


def test_l96_clv_covariance(l96_run):
    """CLVs are covariant: M_n CLV_n is parallel to CLV_{n+1}."""
    m, traj, out = l96_run
    tlm = LY.make_tlm_fd(m, 1)
    n = 150
    j = n - out["index"][0]
    V0 = out["snapshots"][n]  # BLVs at n, (k,1,1,40)
    clv0 = np.einsum("kd,kj->jd", V0.reshape(20, -1), out["C"][j])
    MV = tlm(traj[n], clv0.reshape(20, 1, 1, 40)).reshape(20, -1)
    # BLVs at n+1 are not stored: recompute through QR of M V0 (= BLVs at n+1 up to sign)
    W = tlm(traj[n], V0).reshape(20, -1)
    Qn1, R = np.linalg.qr(W.T)
    Qn1 = Qn1 * np.sign(np.diag(R))  # same sign convention as lyapunov_vectors
    clv1 = Qn1 @ out["C"][j + 1]  # (40, k)
    for i in range(5):
        u = MV[i] / np.linalg.norm(MV[i])
        cos = abs(u @ clv1[:, i]) / np.linalg.norm(clv1[:, i])
        assert cos > 0.99, (i, cos)


def test_projection_fractions(l96_run):
    _, _, out = l96_run
    d = LY.decompose(out, "noise", neutral_tol=0.02)
    s = d["blv_unstable"] + d["blv_neutral"] + d["blv_stable_resolved"] + d["residual"]
    assert np.allclose(s, 1.0)
    # isotropic noise: energy fraction ~ k / n in the leading 20 directions
    assert abs(np.mean(1 - d["residual"]) - 20 / 40) < 0.05
    assert np.all(d["clv_unstable"] >= 0)


def test_projection_of_a_blv_is_exact(l96_run):
    m, traj, _ = l96_run
    n = 150
    # an error equal to BLV #3 at time n projects entirely on index 3
    out = LY.lyapunov_vectors(m, traj[:201], k=20, n_transient_fwd=100, n_transient_bwd=10,
                              snapshot_index=[n], verbose=False, compute_clv=False)
    V = out["snapshots"][n]
    err = np.full(traj[:201].shape, np.nan)
    err[n] = V[3]
    out2 = LY.lyapunov_vectors(m, traj[:201], k=20, n_transient_fwd=100, n_transient_bwd=10,
                               errors=dict(e=err), verbose=False, compute_clv=False)
    c = out2["proj"]["e"]["c"][n - 100]
    assert np.isclose(c[3] ** 2 / (c**2).sum(), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# 4DVar background covariance (numpy part)
# ---------------------------------------------------------------------------
def test_sqrt_b_reproduces_spectrum(spun_up):
    m, p = spun_up
    traj = [p]
    for _ in range(60):
        traj.append(m.forecast_days(traj[-1], 1))
    traj = np.stack(traj)
    cov = isotropic_layer_covariance(traj, m.dx)
    S = sqrt_cov_field(cov, m.nx, scale=1.0)
    rng = np.random.default_rng(0)
    v = rng.standard_normal((400, 2, m.nx, m.nx))
    vh = np.fft.rfft2(v)
    a = S[0, 0] * vh[:, 0] + S[0, 1] * vh[:, 1]
    b = S[1, 0] * vh[:, 0] + S[1, 1] * vh[:, 1]
    samp = np.fft.irfft2(np.stack([a, b], 1), s=(m.nx, m.nx))
    for l in range(2):
        r = samp[:, l].std() / traj[:, l].std()
        assert 0.85 < r < 1.15, (l, r)
    # layer correlation reproduced
    c_true = np.corrcoef(traj[:, 0].ravel(), traj[:, 1].ravel())[0, 1]
    c_samp = np.corrcoef(samp[:, 0].ravel(), samp[:, 1].ravel())[0, 1]
    assert abs(c_true - c_samp) < 0.15


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_metrics():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((5, 1, 32, 32))
    assert M.all_scores(ref, ref, 1e4)["rmse_ssh"] == 0
    s = M.all_scores(ref + 0.1 * rng.standard_normal(ref.shape), ref, 1e4)
    assert 0.85 < s["mu_ssh"] < 0.95
    # ideal low-pass reconstruction -> effective resolution = cut-off wavelength
    n, dx = 64, 1e4
    ref = rng.standard_normal((20, n, n))
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    kk = np.sqrt(k[None, :] ** 2 + k[:, None] ** 2)
    kc = 2 * np.pi / 100e3  # 100 km
    rec = np.fft.ifft2(np.fft.fft2(ref) * (kk < kc)).real
    lam = M.psd_score(rec, ref, dx)[2]
    assert 85 < lam < 115, lam
