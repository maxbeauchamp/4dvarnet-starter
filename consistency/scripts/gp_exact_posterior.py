"""Exact Gaussian posterior of the GP benchmark on its test windows (reference for Table 2).

The GP data (SPDE_diffusion_dataset_1.nc) were simulated with the implicit scheme
    x_{t+1} = M^{-1} (x_t + sigma eps_t),   M = I + dt B,   eps_t ~ N(0, I),
B the anisotropic diffusion operator (kappa = 1/3, H from the H11/H22/H12 fields of the file), and observed without
noise on tracks. Over a window x_{0:4} with a stationary start, the prior precision is block tridiagonal:
    diag (M^2, M^2 + I, M^2 + I, M^2 + I, M^2) / sigma^2,  off-diagonal  -M / sigma^2,
(the stationary precision M^2 - I of x_0 plus the first transition gives M^2). Conditioning on the observed pixels is
exact (noise-free): posterior mean and samples by solving with the unobserved block Q_uu.

dt, sigma^2 and the grid orientation are checked on the data (residuals M x_{t+1} - x_t must be white with variance
sigma^2). Writes, in the format of the notebooks (results/GP/):
    ExactPosterior_metrics.csv, ExactPosterior_metrics_calibration.csv (+ _frames)
with a 20-member ensemble of exact posterior samples, so that SSR, CRPS and ranks use the same code as the models.

    python consistency/scripts/gp_exact_posterior.py [--n-win N] [--members 20]
"""
import argparse, csv, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from calibration import calib_stats, calib_summary, write_calibration_csv   # noqa: E402
from spectral_utils import psd_spectral_score, resolved_scale              # noqa: E402

try:
    from sksparse.cholmod import cholesky as _chol
except Exception:   # pragma: no cover
    _chol = None


def pcg_solver(A, tol=1e-10, maxiter=5000):
    """Jacobi-preconditioned CG (Q_uu is SPD and sparse; direct factorisation of the 3-D window is too slow)."""
    A = sp.csr_matrix(A)
    Dinv = 1.0 / A.diagonal()
    P = spla.LinearOperator(A.shape, matvec=lambda v: Dinv * v)
    def solve(b):
        x, info = spla.cg(A, b, rtol=tol, maxiter=maxiter, M=P)
        if info != 0:
            print(f"  [cg] info={info}", flush=True)
        return x
    return solve


def banded_chol_solver(A):
    """Cholesky of a symmetric banded matrix (LAPACK pbtrf): A in an ordering where its bandwidth is small."""
    import scipy.linalg as sla
    A = sp.coo_matrix(A)
    keep = A.col >= A.row
    r, c, v = A.row[keep], A.col[keep], A.data[keep]
    u = int((c - r).max())
    ab = np.zeros((u + 1, A.shape[0]))
    ab[u + r - c, c] = v
    cb = sla.cholesky_banded(ab, overwrite_ab=True, lower=False)
    return lambda b: sla.cho_solve_banded((cb, False), b)


def factor(A):
    A = sp.csc_matrix(A)
    if _chol is not None:
        f = _chol(A)
        return lambda b: f(b)
    lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A")
    return lambda b: lu.solve(b)


def diff_operator2(nx, ny, H11, H22, H12, kappa, dx=1.0, dy=1.0):
    """Port of diff_operator2 (SPDE/baseline_torch.ipynb): node n, x-index n % nx (fast), H indexed by node."""
    N = nx * ny
    n = np.arange(N)
    hc = H12 + H12   # H12 + H21 (symmetric tensor)
    r, c, v = [], [], []
    def add(sel, off, val):
        r.append(n[sel]); c.append(n[sel] + off); v.append(val[sel])
    right = (n + 1) % nx != 0; left = n % nx != 0
    top = n < (ny - 1) * nx;   bot = n >= nx
    add(right, 1, -H11 / dx**2); add(left, -1, -H11 / dx**2)
    add(top, nx, -H22 / dx**2);  add(bot, -nx, -H22 / dx**2)
    add(right & top, nx + 1, hc / (4 * dx * dy));  add(left & top, nx - 1, -hc / (4 * dx * dy))
    add(right & bot, -nx + 1, -hc / (4 * dx * dy)); add(left & bot, -nx - 1, hc / (4 * dx * dy))
    r.append(n); c.append(n); v.append(kappa**2 + 2 * (H11 / dx**2 + H22 / dy**2))
    return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(N, N))


def test_starts(T, ws=5, stride=1, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Exactly SPDEDataModule.setup."""
    all_starts = list(range(0, T - ws + 1, stride))
    perm = np.random.default_rng(seed).permutation(len(all_starts)).tolist()
    n_train = max(1, int(len(perm) * train_ratio)); n_val = max(1, int(len(perm) * val_ratio))
    n_train = min(n_train, len(perm) - 2); n_val = min(n_val, len(perm) - n_train - 1)
    return [all_starts[i] for i in perm[n_train + n_val:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "data", "SPDE_diffusion_dataset_1.nc"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "sbatch_submission", "results", "GP"))
    ap.add_argument("--n-win", type=int, default=0, help="0: all test windows")
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--kappa", type=float, default=1.0 / 3.0)
    ap.add_argument("--budget", type=float, default=0, help="stop after this many seconds (0: no limit); results so far are written")
    a = ap.parse_args()
    t0 = time.time()
    if a.data.endswith(".npz"):
        ds = {k: type("V", (), {"values": v})() for k, v in np.load(a.data).items()}
    else:
        import xarray as xr
        ds = xr.open_dataset(a.data)
    X = ds["x"].values.astype(np.float64); Y = ds["y"].values.astype(np.float64); OI = ds["OI"].values.astype(np.float64)
    T, ny, nx = X.shape
    print("sksparse.cholmod:", _chol is not None, flush=True)   # last axis fast (C order), as reshape(n_iter, nx, ny) in the generator (nx = ny)
    N = nx * ny
    H11, H22, H12 = (ds[k].values.astype(np.float64).ravel() for k in ("H11", "H22", "H12"))
    print(f"T={T} grid={ny}x{nx} H11 in [{H11.min():.2f},{H11.max():.2f}] H12 in [{H12.min():.2f},{H12.max():.2f}]", flush=True)

    # --- identify orientation and dt from the dynamics (residuals white, variance sigma^2) ---
    best = None
    for orient in ("C", "T"):
        Xf = X.reshape(T, N) if orient == "C" else X.transpose(0, 2, 1).reshape(T, N)
        Hs = (H11, H22, H12) if orient == "C" else tuple(h.reshape(ny, nx).T.ravel() for h in (H11, H22, H12))
        B = diff_operator2(nx, ny, *Hs, a.kappa)
        for dt in (1.0, 0.25):
            M = sp.identity(N, format="csr") + dt * B
            ts = np.arange(50, T - 1)
            R = (M @ Xf[ts + 1].T).T - Xf[ts]
            var = float(R.var())
            R2 = R.reshape(len(ts), ny, nx)
            ac = float((R2[:, :, 1:] * R2[:, :, :-1]).mean() / R2.var())   # lag-1 spatial autocorrelation
            score = abs(np.log(var / dt)) + abs(ac)
            print(f"  orientation {orient} dt={dt}: residual var {var:.4f} (expected dt), lag-1 autocorr {ac:+.3f}", flush=True)
            if best is None or score < best[0]:
                best = (score, orient, dt, var, B, Xf)
    _, orient, dt, s2, B, Xf = best
    print(f"-> orientation {orient}, dt={dt}, sigma^2={s2:.4f}", flush=True)
    Yf = Y.reshape(T, N) if orient == "C" else Y.transpose(0, 2, 1).reshape(T, N)
    OIf = OI.reshape(T, N) if orient == "C" else OI.transpose(0, 2, 1).reshape(T, N)
    unflat = (lambda v: v.reshape(ny, nx)) if orient == "C" else (lambda v: v.reshape(nx, ny).T)

    I = sp.identity(N, format="csr"); M = (I + dt * B).tocsr(); MM = (M.T @ M).tocsr()   # |M x_{t+1} - x_t|^2
    nt = 5
    D = [MM] + [MM + I] * (nt - 2) + [MM]
    blocks = [[None] * nt for _ in range(nt)]
    for k in range(nt):
        blocks[k][k] = D[k]
        if k + 1 < nt:
            blocks[k][k + 1] = -M; blocks[k + 1][k] = -M.T
    Q = (sp.bmat(blocks, format="csr") / s2).tocsr()
    # space-major ordering (node, frame): Q becomes banded (bandwidth ~ nt (nx + 1)), so that the conditioned
    # block Q_uu is factorised exactly by a banded Cholesky (sparse LU of the time-major 3-D system is far too slow)
    perm = (np.arange(nt)[None, :] * N + np.arange(N)[:, None]).ravel()      # new index j -> old index perm[j]
    Qp = Q[perm][:, perm].tocsr()

    # --- prior samples of a window (stationary): run the simulation scheme from 0 with burn-in ---
    rng = np.random.default_rng(0)
    solveM = factor(M)
    prior = np.zeros((a.members, nt * N))
    for m in range(a.members):
        x = np.zeros(N)
        for _ in range(100):
            x = solveM(x + np.sqrt(s2) * rng.standard_normal(N))
        for k in range(nt):
            x = solveM(x + np.sqrt(s2) * rng.standard_normal(N)) if k else x
            prior[m, k * N:(k + 1) * N] = x
    print(f"prior samples ready ({time.time() - t0:.0f}s)", flush=True)

    starts = test_starts(T)
    if a.n_win:
        starts = starts[:a.n_win]
    rows, stats, rmse_pairs, rmse_oi_pairs, lam_pairs = [], [], [], [], []
    for w, s in enumerate(starts):
        yw = Yf[s:s + nt].ravel()[perm]; xw = Xf[s:s + nt].ravel()[perm]
        o = np.isfinite(yw); u = ~o
        Quu = Qp[u][:, u]; Quo = Qp[u][:, o]
        solve = banded_chol_solver(Quu)
        mean = xw * 0.0; mean[o] = yw[o]
        mean[u] = solve(-(Quo @ yw[o]))
        ens = np.empty((a.members, nt * N))
        for m in range(a.members):
            ps = prior[m][perm]
            cond_u = solve(-(Quo @ ps[o]))          # E[u | prior obs]
            ens[m] = mean; ens[m, u] = mean[u] + (ps[u] - cond_u)
        inv = np.argsort(perm)
        xw, mean, ens = xw[inv], mean[inv], ens[:, inv]
        for k in range(nt):
            sl = slice(k * N, (k + 1) * N)
            gt = unflat(xw[sl]); mu = unflat(mean[sl]); e = np.stack([unflat(v[sl]) for v in ens])
            rmse_pairs.append(float(np.sqrt(np.mean((mu - gt) ** 2))))
            rmse_oi_pairs.append(float(np.sqrt(np.mean((unflat(OIf[s + k]) - gt) ** 2))) if np.all(np.isfinite(OIf[s + k])) else np.nan)
            wl, _, _, spec = psd_spectral_score(mu, gt, dx=1.0)
            lam_pairs.append(resolved_scale(wl, spec, threshold=0.5))
            _t = k   # read by calib_stats for the per-frame summary
            stats.append(calib_stats(e, gt, np.ones_like(gt, dtype=bool)))
        print(f"  window {w + 1}/{len(starts)} (start {s}): RMSE {np.mean(rmse_pairs[-nt:]):.4f}"
              f"  OI {np.nanmean(rmse_oi_pairs[-nt:]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if (w + 1) % 5 == 0 or w + 1 == len(starts) or (a.budget and time.time() - t0 > a.budget):
            write(a, stats, rmse_pairs, rmse_oi_pairs, lam_pairs, w + 1, orient, dt, s2, t0)
        if a.budget and time.time() - t0 > a.budget:
            print("time budget reached", flush=True); break


def write(a, stats, rmse_pairs, rmse_oi_pairs, lam_pairs, n_done, orient, dt, s2, t0):
    summ = calib_summary(stats)
    os.makedirs(a.out, exist_ok=True)
    agg = lambda v: (float(np.nanmean(v)), float(np.nanstd(v)))
    (rm, rs), (lm, ls) = agg(rmse_pairs), agg(lam_pairs)
    with open(os.path.join(a.out, "ExactPosterior_metrics.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["Method", "RMSE", "lambda_x", "CRPS"])
        wr.writerow([f"Exact posterior ({n_done} test windows)", f"{rm:.4f} $\\pm$ {rs:.4f}", f"{lm:.1f} $\\pm$ {ls:.1f}", f"{summ['crps_all']:.4f}"])
    write_calibration_csv(os.path.join(a.out, "ExactPosterior_metrics_calibration.csv"), "ExactPosterior", summ, a.members)
    print(f"\nEXACT POSTERIOR on {n_done} test windows: RMSE {rm:.4f}  lambda_x {lm:.1f}  CRPS(all px) {summ['crps_all']:.4f}"
          f"  SSR {summ['ssr']:.3f}  rank_dev {summ['rank_dev']:.2f}  | OI of the file on the same windows: RMSE {np.nanmean(rmse_oi_pairs):.4f}"
          f"  | orientation {orient} dt {dt} sigma2 {s2:.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
