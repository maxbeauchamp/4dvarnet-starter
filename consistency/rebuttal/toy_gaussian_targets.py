"""Gaussian toy check of the reviewers' diagnosis (XETn W1/W4, qyTj, aayn Q2). Numpy only, runs on CPU in seconds.

Data x0 ~ N(0, 1), VE noising x_s = x0 + s*eps. In this linear-Gaussian case every L2-optimal map is linear,
g(x; s, s') = a(s, s') x, so each training target can be solved in closed form and checked by Monte Carlo.

* exact PF-ODE map       : a = sqrt((1 + s'^2) / (1 + s^2))          (preserves the marginal N(0, 1 + s'^2))
* shared-noise target    : minimise E[(a x_s - x_s')^2] with the SAME eps  ->  a = (1 + s s') / (1 + s^2)
  (this is what Alg. 3/5 and the pairwise CM code do: student input and target share one draw of eps)
* VarCM `l_long` anchor  : minimise E[(g(x_T) - x0)^2]  ->  g = E[x0 | x_T], a = 1 / (1 + T^2)
* ODE-teacher target     : a = exact map (what CTM-style targets converge to)

Prints, for each target: one-step output std at s' = 0 (should be 1), semigroup gap |a(s,0) - a(s,m) a(m,0)|,
and the CRPS of the one-step ensemble against x0 relative to the exact posterior sampler.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 400_000


def a_exact(s, sp):
    return np.sqrt((1 + sp**2) / (1 + s**2))


def a_shared(s, sp):
    return (1 + s * sp) / (1 + s**2)


def crps_ensemble(samples, obs):
    """CRPS of an ensemble (members on axis 0) for each observation, mean over observations."""
    m = samples.shape[0]
    t1 = np.abs(samples - obs).mean(0)
    s = np.sort(samples, axis=0)
    i = np.arange(1, m + 1)[:, None]
    t2 = ((2 * i - m - 1) * s).sum(0) / (m * m)
    return float((t1 - t2).mean())


def mc_check(s, sp):
    x0 = rng.standard_normal(N); eps = rng.standard_normal(N)
    xs, xsp = x0 + s * eps, x0 + sp * eps
    return float((xs * xsp).mean() / (xs * xs).mean())


print("Monte Carlo check of the shared-noise optimum a = (1 + s s')/(1 + s^2):")
for s, sp in [(10, 1), (80, 5), (5, 0), (80, 0)]:
    print(f"  s={s:>3} -> s'={sp}: closed form {a_shared(s, sp):.5f}   Monte Carlo {mc_check(s, sp):.5f}   exact ODE {a_exact(s, sp):.5f}")

T, mid = 80.0, 5.0
print(f"\nSampling from s_max = {T:g} to 0 (one step), composition via s = {mid:g}:")
rows = []
for name, a in [("exact PF-ODE / ODE-teacher target", a_exact), ("shared-noise target (current)", a_shared)]:
    direct, composed = a(T, 0), a(T, mid) * a(mid, 0)
    rows.append((name, direct * np.sqrt(1 + T**2), abs(direct - composed), direct, composed))
a_long = 1 / (1 + T**2)
rows.append(("VarCM l_long anchor E[x0|x_T]", a_long * np.sqrt(1 + T**2), float("nan"), a_long, float("nan")))
print(f"  {'target':<36} {'output std':>10} {'semigroup gap':>14} {'a direct':>10} {'a composed':>11}")
for name, std, gap, d, c in rows:
    print(f"  {name:<36} {std:>10.4f} {gap:>14.2e} {d:>10.2e} {c:>11.2e}")

# Posterior-sampling version: y = x0 + r*eta observed; the exact posterior is N(k y, 1 - k), k = 1/(1 + r^2).
r, M, n = 0.5, 32, 20_000
k = 1 / (1 + r**2)
x0 = rng.standard_normal(n); y = x0 + r * rng.standard_normal(n)
mu, var = k * y, 1 - k
z = rng.standard_normal((M, n))
ens = {
    "exact posterior": mu + np.sqrt(var) * z,
    # x_T = mu + sqrt(var + T^2) z; the exact ODE map rescales it to the posterior, the shared-noise optimum
    # g = mu + var / (var + T^2) (x_T - mu) shrinks it almost to the mean.
    "ODE-teacher one-step map": mu + np.sqrt(var / (var + T**2)) * np.sqrt(var + T**2) * z,
    "shared-noise one-step map": mu + var / (var + T**2) * np.sqrt(var + T**2) * z,
    "l_long anchor (mean)": mu + 0 * z,
}
print(f"\nConditional version (obs noise r = {r}), {M} members, CRPS / RMSE / spread-skill ratio:")
for name, e in ens.items():
    rmse = float(np.sqrt(((e.mean(0) - x0) ** 2).mean()))
    spread = float(np.sqrt(e.var(0, ddof=1).mean()))
    print(f"  {name:<28} CRPS {crps_ensemble(e, x0):.4f}   RMSE {rmse:.4f}   SSR {spread / rmse:.3f}")
