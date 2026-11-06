"""
Uncertainty of the diagnostic: paired moving-block bootstrap over test days.

The daily error energies of all methods are resampled with the *same* blocks of days, so the
confidence intervals of the ratios between methods account for their (strong) correlation in time
and between methods. Use a block length of the order of the error decorrelation time (~ Lyapunov time).
"""
import numpy as np

SUBSPACES = ["total", "blv_unstable_abs", "blv_neutral_abs", "blv_stable_resolved_abs", "residual_abs"]


def block_bootstrap_indices(n, block, n_boot, rng):
    """(n_boot, n) day indices made of circular blocks of length ``block``."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)) % n
    return idx.reshape(n_boot, -1)[:, :n]


def subspace_ci(series, ref, block=10, n_boot=2000, level=0.9, seed=0):
    """series: dict method -> dict subspace -> (T,) daily energies (NaN-free days are used).

    Returns a list of rows: method, subspace, mean, lo, hi, ratio = E_ref / E_method (> 1: method better
    than ref), ratio_lo, ratio_hi, p_better = bootstrap probability that ratio > 1."""
    rng = np.random.default_rng(seed)
    methods = list(series)
    subs = [s for s in SUBSPACES if all(s in series[m] for m in methods)]
    good = np.ones(len(series[ref][subs[0]]), bool)
    for m in methods:
        for s in subs:
            good &= np.isfinite(series[m][s])
    T = int(good.sum())
    idx = block_bootstrap_indices(T, block, n_boot, rng)
    a = (1 - level) / 2
    rows = []
    for s in subs:
        ref_b = series[ref][s][good][idx].mean(1)
        for m in methods:
            x = series[m][s][good]
            boot = x[idx].mean(1)
            ratio = ref_b / boot
            rows.append(dict(method=m, subspace=s.replace("_abs", "").replace("blv_", ""), mean=float(x.mean()),
                             lo=float(np.quantile(boot, a)), hi=float(np.quantile(boot, 1 - a)),
                             ratio=float(series[ref][s][good].mean() / x.mean()),
                             ratio_lo=float(np.quantile(ratio, a)), ratio_hi=float(np.quantile(ratio, 1 - a)),
                             p_better=float((ratio > 1).mean())))
    return rows


def subspace_ci_from_file(path, ref="4dvar_strong", methods=None, **kw):
    """Same, reading diag/lyapunov_projection.nc (written by diag.run_diag)."""
    import json

    import xarray as xr

    ds = xr.open_dataset(path, decode_times=False)
    methods = methods or json.loads(ds.attrs["methods"])
    series = {m: {s: ds[f"{m}__{s}"].values for s in SUBSPACES if f"{m}__{s}" in ds} for m in methods}
    return subspace_ci(series, ref, **kw)
