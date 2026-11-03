"""
OSSE generation for the two-layer QG testbed.

truth  : QG run at ``nx_truth`` (default 128) spectrally truncated to the DA grid
         ``nx`` (default 64)  -> model error of the LR model = unresolved scales
         (``nx_truth == nx`` gives a perfect-model twin experiment)
obs    : nadir + SWOT sampling of the truth SSH (see obs.py)
output : one netCDF file with dims (time, layer, y, x) holding
         psi (truth streamfunction, LR grid), ssh, obs_ssh, obs_sigma, sat_id
         + attrs: model / DA-model parameters, constellation, split indices.

Run:  python main.py xp=qg2l_generate
"""
import json
import time
from pathlib import Path

import numpy as np

from .qgmodel import QG2L, spectral_coarsen
from . import obs as obs_mod

PRESETS = {
    # pyqg "eddy" configuration (Ross et al. 2023): weakly chaotic at nx=64 (lambda1 ~ 0.035/day)
    "eddy": dict(beta=1.5e-11, rd=15e3, delta=0.25, H1=500.0, U1=0.025, U2=0.0, rek=5.787e-7),
    # doubled vertical shear: lambda1 ~ 0.09/day at nx=64 (tau_lambda ~ nadir revisit time)
    "energetic": dict(beta=1.5e-11, rd=15e3, delta=0.25, H1=500.0, U1=0.05, U2=0.0, rek=5.787e-7),
}

DEFAULT_SPLITS = dict(  # in days after spin-up
    train=(0, 1095),
    val=(1095, 1277),
    test=(1307, 1672),
    total=1762,  # 90-day margin after test for the CLV backward iteration
)


def model_kwargs(preset="energetic", nx=64, dt=7200.0, L=1e6, **overrides):
    """Full QG2L constructor kwargs (all physical parameters, incl. f0 and g)."""
    kw = dict(PRESETS[preset])
    kw.update(nx=nx, dt=dt, L=L)
    kw.update(overrides)
    kw = dict(QG2L(**kw).params)
    return kw


def simulate_truth(
    preset="energetic",
    nx=64,
    nx_truth=128,
    dt_truth=3600.0,
    L=1e6,
    spinup_days=3 * 365,
    n_days=DEFAULT_SPLITS["total"],
    seed=0,
    backend="numpy",
    device="cpu",
    dtype="float64",
    truth_overrides=None,
    keep_hr=False,
    log_every=100,
):
    """Integrate the truth model and return daily LR-grid fields (numpy)."""
    kw = model_kwargs(preset, nx=nx_truth, dt=dt_truth, L=L, **(truth_overrides or {}))
    m = QG2L(backend=backend, device=device, dtype=dtype, **kw)
    t0 = time.time()
    psi = m.random_init(seed=seed)
    energy = []

    def log(d, p):
        energy.append(float(m.energy(p)))
        if log_every and d % log_every == 0:
            print(f"[truth] day {d}: E={energy[-1]:.3e} m2/s2  cfl={m.cfl(p):.2f}  ({time.time()-t0:.0f}s)", flush=True)

    to_np = (lambda a: a.detach().cpu().numpy()) if backend == "torch" else np.asarray
    for d in range(spinup_days):
        psi = m.forecast_days(psi, 1)
        if log_every and (d + 1) % (5 * log_every) == 0:
            print(f"[truth] spin-up day {d+1}: E={float(m.energy(psi)):.3e}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[truth] spin-up done ({spinup_days} d): E={float(m.energy(psi)):.3e}", flush=True)
    traj = [to_np(psi)]
    log(0, psi)
    for d in range(1, n_days):
        psi = m.forecast_days(psi, 1)
        traj.append(to_np(psi))
        log(d, psi)
    traj = np.stack(traj)
    lr = spectral_coarsen(traj, nx) if nx_truth != nx else traj
    return dict(
        psi=lr.astype(np.float32),
        psi_hr=traj.astype(np.float32) if keep_hr and nx_truth != nx else None,
        energy=np.array(energy),
        truth_params=m.params,
    )


def estimate_lambda1(model_kw, psi0, n_days=120, skip=30, seed=0):
    """Leading Lyapunov exponent (1/day) from the growth of a tiny perturbation (numpy)."""
    m = QG2L(**model_kw)
    rng = np.random.default_rng(seed)
    p = np.stack([psi0, psi0 + 1e-6 * np.abs(psi0).max() * rng.standard_normal(psi0.shape)])
    d = []
    for _ in range(n_days):
        p = m.forecast_days(p, 1)
        d.append(m.energy(p[1] - p[0]))
    ld = 0.5 * np.log(np.array(d))
    return float((ld[-1] - ld[skip]) / (n_days - 1 - skip))


def revisit_time(sat_id):
    """Mean revisit time (days) per pixel and its domain median."""
    seen = sat_id >= 0
    n = seen.sum(0)
    rt = np.where(n > 0, seen.shape[0] / np.maximum(n, 1), np.inf)
    return rt, float(np.median(rt[np.isfinite(rt)]))


def to_dataset(truth, obs, model_kw, constellation, splits, extra_attrs=None):
    import xarray as xr

    psi = truth["psi"]
    n_days, _, ny, nx = psi.shape
    L = model_kw["L"]
    x = (np.arange(nx) + 0.5) * L / nx / 1e3
    m = QG2L(**model_kw)
    ssh = m.ssh(psi.astype(np.float64)).astype(np.float32)
    ds = xr.Dataset(
        dict(
            psi=(("time", "layer", "y", "x"), psi, dict(units="m2 s-1", long_name="truth streamfunction (DA grid)")),
            ssh=(("time", "y", "x"), ssh, dict(units="m")),
            obs_ssh=(("time", "y", "x"), obs["obs"].astype(np.float32), dict(units="m")),
            obs_sigma=(("time", "y", "x"), obs["obs_sigma"].astype(np.float32), dict(units="m")),
            sat_id=(("time", "y", "x"), obs["sat_id"].astype(np.int8)),
        ),
        coords=dict(time=np.arange(n_days, dtype=np.float32), layer=[1, 2], y=x, x=x),
    )
    ds.time.attrs.update(units="day", long_name="days after the end of the spin-up")  # no "since": not a CF date
    ds.x.attrs.update(units="km")
    ds.y.attrs.update(units="km")
    ds.attrs.update(
        model_kw=json.dumps(model_kw),
        truth_params=json.dumps(truth["truth_params"]),
        constellation=json.dumps(constellation),
        sat_names=json.dumps(obs["names"]),
        splits=json.dumps(splits),
        **(extra_attrs or {}),
    )
    return ds


def generate(
    out_dir="data/qg2l",
    preset="energetic",
    nx=64,
    dt=7200.0,
    nx_truth=128,
    dt_truth=3600.0,
    L=1e6,
    spinup_days=3 * 365,
    splits=None,
    constellation=None,
    lat=35.0,
    seed=0,
    backend="torch",
    device="cuda",
    dtype="float64",
    truth_overrides=None,
    make_figures=True,
    animation_days=60,
):
    """Generate the OSSE dataset + figures. Returns the path of the netCDF file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = dict(splits or DEFAULT_SPLITS)
    constellation = constellation or obs_mod.DEFAULT_CONSTELLATION
    constellation = {k: dict(v) for k, v in constellation.items()}
    if backend == "torch":
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[generate] CUDA not available, falling back to CPU")
            device = "cpu"

    truth = simulate_truth(
        preset=preset, nx=nx, nx_truth=nx_truth, dt_truth=dt_truth, L=L, spinup_days=spinup_days,
        n_days=splits["total"], seed=seed, backend=backend, device=device, dtype=dtype,
        truth_overrides=truth_overrides,
    )
    model_kw = model_kwargs(preset, nx=nx, dt=dt, L=L)
    m = QG2L(**model_kw)
    ssh = m.ssh(truth["psi"].astype(np.float64))
    obs = obs_mod.build_observations(ssh, L, constellation, lat=lat, seed=seed + 1)
    rt_map, rt_med = revisit_time(obs["sat_id"])
    lam1 = estimate_lambda1(model_kw, truth["psi"][splits["test"][0]].astype(np.float64))
    summary = dict(
        lambda1_per_day_LRmodel=lam1,
        lyapunov_time_days=1.0 / lam1 if lam1 > 0 else float("inf"),
        median_revisit_days=rt_med,
        lambda1_x_revisit=lam1 * rt_med,
        coverage_per_day=obs_mod.coverage_stats(obs["sat_id"], obs["names"]),
        ssh_std_m=float(ssh.std()),
    )
    print("[generate] summary:", json.dumps(summary, indent=1))
    ds = to_dataset(truth, obs, model_kw, constellation, splits, extra_attrs=dict(summary=json.dumps(summary)))
    path = out_dir / f"qg2l_{preset}_nx{nx}_truth{nx_truth}.nc"
    ds.to_netcdf(path)
    np.save(out_dir / "truth_energy.npy", truth["energy"])
    print(f"[generate] wrote {path}")

    if make_figures:
        from . import viz

        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        viz.dataset_figures(ds, fig_dir, energy=truth["energy"], revisit_map=rt_map)
        if animation_days:
            t0 = splits["test"][0]
            viz.animate(ds, fig_dir / "truth_obs.mp4", days=range(t0, t0 + animation_days))
    return str(path)
