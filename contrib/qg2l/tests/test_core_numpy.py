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


def test_clv_split_bounded_and_exact(l96_run):
    m, traj, out = l96_run
    d = LY.decompose(out, "noise", neutral_tol=0.02)
    for key in ["clv_unstable", "clv_neutral", "clv_stable_resolved"]:
        assert np.all((d[key] >= -1e-12) & (d[key] <= 1 + 1e-12)), key
    assert np.all((d["angle_us_deg"] > 0) & (d["angle_us_deg"] <= 90))
    # an error in the unstable subspace: clv_unstable = 1, its stable projection = cos^2 of its angle to S
    uns, neu, _ = LY.classify_exponents(out["exponents"], 0.02)
    c = np.zeros((1, 20))
    c[0, 0], c[0, 2] = 1.0, -0.5
    sp = LY._clv_split(out["C"][:1], c, int(uns.sum()), int(neu.sum()))
    assert np.isclose(sp["clv_unstable"][0], 1.0)
    assert sp["clv_stable_resolved"][0] <= np.cos(np.radians(sp["angle_us_deg"][0])) ** 2 + 1e-9


def test_blend_with_padded_windows():
    """Windows sticking out of the test period: full, exact coverage."""
    import collections, sys, types
    # data.py needs torch / lightning / src.data only for class definitions: stub them if missing
    try:
        from qg2l import data as D
    except ImportError:
        stub = types.ModuleType("torch")
        stub.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object, DataLoader=None))
        pl = types.ModuleType("pytorch_lightning"); pl.LightningDataModule = object
        sd = types.ModuleType("src.data"); sd.TrainingItem = collections.namedtuple("TrainingItem", ["input", "tgt"])
        sys.modules.update({"torch": stub, "torch.utils": stub.utils, "torch.utils.data": stub.utils.data,
                            "pytorch_lightning": pl, "src": types.ModuleType("src"), "src.data": sd})
        from qg2l import data as D
    rng = np.random.default_rng(0)
    state = rng.standard_normal((100, 2, 8, 8)).astype(np.float32)
    obs = state[:, 0].copy()
    T, t0, t1 = 7, 40, 60
    ds = D.QGWindowDataset(state, obs, obs * 0, t0, t1, T, pad=T - 1)
    assert ds.starts[0] == t0 - T + 1 and ds.starts[-1] == t1 - 1
    outs = np.stack([np.moveaxis(state[s:s + T], 1, 0) for s in ds.starts])
    w = D.qg_rec_weight(T, [1, 2], 8).reshape(2, T, 8, 8)[:, :, 0, 0]
    rec = D.blend_windows(outs, ds.starts, t0, t1 - t0, w)
    assert np.allclose(rec, state[t0:t1], atol=1e-6)


# ---------------------------------------------------------------------------
# realism options
# ---------------------------------------------------------------------------
def test_band_limited_and_unresolved_signal():
    rng = np.random.default_rng(0)
    f = O.band_limited_field(64, 1e6, (100.0, 200.0), rng)
    assert np.isclose(f.std(), 1.0)
    k = np.fft.fftfreq(64, 1e6 / 64) * 1e3  # cycles / km
    kk = np.sqrt(k[None, :] ** 2 + k[:, None] ** 2)
    p = np.abs(np.fft.fft2(f)) ** 2
    assert p[(kk < 1 / 200 - 1e-9) | (kk > 1 / 100 + 1e-9)].sum() < 1e-10 * p.sum()
    s = O.unresolved_signal(400, 32, 5e5, ssh_std=0.01, wavelength_km=(50, 150), decorrelation_days=5.0, rng=rng)
    assert abs(s.std() - 0.01) < 0.002
    r = np.corrcoef(s[:-1].ravel(), s[1:].ravel())[0, 1]
    assert abs(r - np.exp(-1 / 5)) < 0.05


def test_truth_forcing_and_hr_block_obs():
    from qg2l.simulate import simulate_truth

    common = dict(preset="energetic", nx=16, nx_truth=32, dt_truth=3600.0, L=2.5e5, spinup_days=60, n_days=20,
                  backend="numpy", log_every=0)
    a = simulate_truth(**common)
    b = simulate_truth(**common, forcing=dict(ssh_std=0.002, wavelength_km=(60.0, 200.0)), obs_source="hr_block")
    assert a["ssh_obs_src"] is None and b["ssh_obs_src"].shape == (20, 16, 16)
    assert not np.allclose(a["psi"], b["psi"])  # forcing changes the truth
    m = QG2L(**model_kwargs("energetic", nx=16, dt=7200.0, L=2.5e5))
    ssh_lr = m.ssh(b["psi"].astype(np.float64))
    diff = b["ssh_obs_src"] - ssh_lr  # representativeness error: small but non zero, mostly small scales
    assert 0 < diff.std() < 0.5 * ssh_lr.std()


def test_free_run_b():
    from qg2l.fourdvar import free_run, isotropic_layer_covariance

    kw = model_kwargs("energetic", nx=16, dt=7200.0, L=2.5e5)
    m = QG2L(**kw)
    p = m.random_init(0, amp=1e-5)
    for _ in range(300):
        p = m.forecast_days(p, 1)
    fr = free_run(kw, p, 50, every=5, discard=5)
    assert fr.shape == (10, 2, 16, 16) and np.isfinite(fr).all()
    cov = isotropic_layer_covariance(fr, m.dx)
    assert np.all(cov["Sigma"][1:, 0, 0] >= 0)


def test_conjugate_gradient():
    from qg2l.fourdvar import conjugate_gradient

    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 30))
    A = np.eye(30) + A @ A.T  # I + G^T G structure
    b = rng.standard_normal(30)
    x, it = conjugate_gradient(lambda p: A @ p, b, 60, rtol=1e-12)
    assert np.allclose(A @ x, b, atol=1e-8) and it <= 60
    # few iterations already reduce the quadratic cost a lot (preconditioned problem)
    x5, _ = conjugate_gradient(lambda p: A @ p, b, 5)
    q = lambda v: 0.5 * v @ A @ v - b @ v
    assert q(x5) < 0.5 * q(np.zeros(30)) + 0.5 * q(x)


def test_lyapunov_checkpoint_resume(tmp_path):
    """Interrupted forward sweep + resume == uninterrupted run."""
    m = L96()
    x = m.forecast_days(8 + np.random.default_rng(0).standard_normal((1, 1, 40)), 50)
    traj = [x]
    for _ in range(80):
        traj.append(m.forecast_days(traj[-1], 1))
    traj = np.stack(traj)
    err = dict(e=0.1 * np.random.default_rng(1).standard_normal(traj.shape))
    kw = dict(k=12, n_transient_fwd=20, n_transient_bwd=20, errors=err, snapshot_index=[40], verbose=False)
    ref = LY.lyapunov_vectors(m, traj, **kw)

    calls = {"n": 0}
    base = LY.make_tlm_fd(m, 1)

    def crashing_tlm(x, V):
        calls["n"] += 1
        if calls["n"] == 47:
            raise RuntimeError("simulated GPU crash")
        return base(x, V)

    ck = tmp_path / "ck"
    with pytest.raises(RuntimeError):
        LY.lyapunov_vectors(m, traj, tlm=crashing_tlm, checkpoint_dir=ck, checkpoint_key="a", checkpoint_every=10, **kw)
    assert (ck / "state.npz").exists()
    calls["n"] = -10**9  # no more crash
    res = LY.lyapunov_vectors(m, traj, tlm=crashing_tlm, checkpoint_dir=ck, checkpoint_key="a", checkpoint_every=10, **kw)
    assert calls["n"] == -10**9 + (80 - 40)  # resumed at step 40: only 40 more TLM calls
    assert np.allclose(res["exponents"], ref["exponents"])
    assert np.allclose(res["C"], ref["C"])
    assert np.allclose(res["proj"]["e"]["c"], ref["proj"]["e"]["c"], equal_nan=True)
    assert np.allclose(res["snapshots"][40], ref["snapshots"][40])
    # another configuration key is not resumed
    res2 = LY.lyapunov_vectors(m, traj, checkpoint_dir=ck, checkpoint_key="b", **kw)
    assert np.allclose(res2["exponents"], ref["exponents"])


def test_block_bootstrap_ci():
    from qg2l.stats import block_bootstrap_indices, subspace_ci

    rng = np.random.default_rng(0)
    idx = block_bootstrap_indices(90, 10, 5, rng)
    assert idx.shape == (5, 90) and idx.max() < 90
    assert np.all(np.diff(idx[0][:10]) % 90 == 1)  # contiguous (circular) blocks
    # correlated daily series: method b is 2x worse than a, with a common AR(1) fluctuation
    T = 200
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = 0.9 * z[t - 1] + rng.standard_normal()
    base = np.exp(0.3 * z)
    series = {"a": {"total": base * (1 + 0.05 * rng.standard_normal(T))},
              "b": {"total": 2 * base * (1 + 0.05 * rng.standard_normal(T))}}
    rows = {r["method"]: r for r in subspace_ci(series, ref="b", block=10, n_boot=1000)}
    assert 1.8 < rows["a"]["ratio_lo"] < rows["a"]["ratio"] < rows["a"]["ratio_hi"] < 2.2  # paired: tight
    assert rows["a"]["p_better"] == 1.0 and abs(rows["b"]["ratio"] - 1) < 1e-12
    assert rows["a"]["lo"] < rows["a"]["mean"] < rows["a"]["hi"]


def test_desroziers_and_oracle_obs_std():
    from qg2l.fourdvar import desroziers_obs_std, oracle_obs_std, sigma_from_satellite_std

    rng = np.random.default_rng(0)
    B, R, n = 1.0, 0.25, 200000  # optimal scalar analysis: E[d_a d_b] = R
    x = rng.normal(0, 1, n)
    y = x + rng.normal(0, R**0.5, n)
    xa = B / (B + R) * y
    sid = np.zeros((1, 1, n), int)
    sid[0, 0, ::2] = 1
    est = desroziers_obs_std([(0, y.reshape(1, 1, n), (y - xa).reshape(1, 1, n))], sid)
    assert abs(est[0] - 0.5) < 0.01 and abs(est[1] - 0.5) < 0.01
    obs = (x + 0.1 * rng.standard_normal(n)).reshape(1, 1, n)
    assert abs(oracle_obs_std(obs, x.reshape(1, 1, n), sid, 0, 1)[0] - 0.1) < 0.005
    sig = sigma_from_satellite_std(sid, np.full((1, 1, n), 0.03), {1: 0.07})
    assert np.all(sig[sid == 1] == 0.07) and np.all(sig[sid == 0] == 0.03)
