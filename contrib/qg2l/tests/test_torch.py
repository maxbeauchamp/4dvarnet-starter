"""
Torch-dependent tests (skipped when torch is not installed). Run from the repo root:
    python -m pytest contrib/qg2l/tests -q
They use tiny grids and run in about a minute on CPU.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qg2l.qgmodel import QG2L  # noqa: E402
from qg2l.simulate import model_kwargs  # noqa: E402
from qg2l import lyapunov as LY  # noqa: E402
from qg2l.fourdvar import FourDVar, isotropic_layer_covariance  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    kw = model_kwargs("energetic", nx=32, dt=7200.0, L=5e5)
    m = QG2L(**kw)
    p = m.random_init(0, amp=1e-5)
    for _ in range(400):
        p = m.forecast_days(p, 1)
    traj = [p]
    for _ in range(40):
        traj.append(m.forecast_days(traj[-1], 1))
    return kw, m, np.stack(traj)


def test_torch_matches_numpy(setup):
    kw, m, traj = setup
    mt = QG2L(backend="torch", dtype="float64", **kw)
    a = m.forecast_days(traj[0], 3)
    b = mt.forecast_days(mt.xp.asarray(traj[0]), 3).numpy()
    assert np.allclose(a, b, rtol=1e-9, atol=1e-6 * np.abs(a).max())


def test_jvp_matches_fd(setup):
    kw, _, traj = setup
    mt = QG2L(backend="torch", dtype="float64", **kw)
    x = mt.xp.asarray(traj[0])
    V = mt.xp.asarray(np.random.default_rng(0).standard_normal((3,) + traj.shape[1:]))
    a = LY.make_tlm_torch(mt)(x, V)
    b = LY.make_tlm_fd(mt)(x, V)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5 * float(a.abs().max()))


def test_4dvar_adjoint_gradient(setup):
    """Autograd gradient of the 4DVar cost vs a finite difference along a random direction."""
    kw, m, traj = setup
    cov = isotropic_layer_covariance(traj, m.dx)
    da = FourDVar(kw, cov, b_scale=0.5, window_days=3, n_iter=1, verbose=False)
    ssh = m.ssh(traj[:3])
    obs = np.where(np.random.default_rng(1).random(ssh.shape) < 0.1, ssh, np.nan)
    sig = np.where(np.isfinite(obs), 0.01, np.nan)
    xb = torch.as_tensor(traj[0] * 0.9, dtype=torch.float64)
    y = torch.as_tensor(np.nan_to_num(obs))
    msk = torch.as_tensor(np.isfinite(obs))

    def J(v):
        x = xb + da._apply_sqrt(da.Sb, v)
        out = [x]
        for _ in range(2):
            x = da.model.forecast_days(x, 1)
            out.append(x)
        innov = (y - da.ssh(torch.stack(out))) / 0.01
        return 0.5 * (v**2).sum() + 0.5 * (innov[msk] ** 2).sum()

    v = torch.randn(2, 32, 32, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(J(v), v)[0]
    d = torch.randn_like(v)
    h = 1e-4
    fd = (J(v + h * d) - J(v - h * d)) / (2 * h)
    assert abs(float((g * d).sum()) - float(fd)) < 1e-4 * abs(float(fd))


@pytest.mark.parametrize("mode", ["strong", "weak"])
def test_4dvar_twin_reduces_error(setup, mode):
    """Perfect-model twin: analysis closer to the truth than the background."""
    kw, m, traj = setup
    cov = isotropic_layer_covariance(traj, m.dx)
    ssh = m.ssh(traj)
    rng = np.random.default_rng(2)
    obs = np.where(rng.random(ssh.shape) < 0.15, ssh + 0.005 * rng.standard_normal(ssh.shape), np.nan)
    sig = np.where(np.isfinite(obs), 0.005, np.nan)
    da = FourDVar(kw, cov, b_scale=0.5, mode=mode, q_scale=0.02, window_days=4, n_iter=30, verbose=False)
    xb = torch.as_tensor(traj[10] + 0.5 * (traj[0] - traj[10]), dtype=torch.float64)
    a, info = da.analyse_window(xb, obs[10:14], sig[10:14])
    err_b = np.sqrt(((xb.numpy() - traj[10]) ** 2).mean())
    err_a = np.sqrt(((a[0].numpy() - traj[10]) ** 2).mean())
    assert info["J"] < info["J0"]
    assert err_a < 0.7 * err_b, (err_a, err_b)


def test_datamodule_and_litmodule(tmp_path, setup):
    """Synthetic netCDF -> QG2LDataModule -> one 4DVarNet forward / loss / test stitching."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("pytorch_lightning")
    import json
    from qg2l import obs as O
    from qg2l.simulate import to_dataset
    from qg2l.data import QG2LDataModule, qg_rec_weight, n_channels
    import src.models
    from qg2l.models import periodic

    kw, m, traj = setup
    psi = np.concatenate([traj] * 2)[:80].astype(np.float32)
    ob = O.build_observations(m.ssh(psi.astype(np.float64)), kw["L"], seed=0)
    splits = dict(train=[0, 40], val=[40, 55], test=[55, 75], total=80)
    ds = to_dataset(dict(psi=psi, truth_params=kw), ob, kw, O.DEFAULT_CONSTELLATION, splits)
    path = tmp_path / "qg.nc"
    ds.to_netcdf(path)

    dm = QG2LDataModule(str(path), n_time=5, dl_kw=dict(batch_size=4, num_workers=0))
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    C = n_channels(5, [1, 2])
    assert batch.input.shape == (4, C, 32, 32)
    assert torch.isnan(batch.input[:, 5:]).all()

    solver = src.models.GradSolver(
        prior_cost=periodic(src.models.BilinAEPriorCost(C, 16, downsamp=2)),
        obs_cost=src.models.BaseObsCost(),
        grad_mod=periodic(src.models.ConvLstmGradModel(C, 16)),
        n_step=2, lr_grad=1e3,
    )
    out = solver(batch)
    assert out.shape == batch.tgt.shape and torch.isfinite(out).all()
    w = qg_rec_weight(5, [1, 2], 32)
    assert w.shape == (C, 32, 32)
    json.loads(xr.open_dataset(path, decode_times=False).attrs["model_kw"])["f0"]
