"""
Backward (BLV) and covariant (CLV) Lyapunov vectors along a reference trajectory,
computed in the energy norm of the QG model, and projection of reconstruction
errors onto the unstable / neutral / stable subspaces.

Conventions
-----------
* ``traj``: reference states x_n (n = 0..N) one ``interval_days`` apart (daily).
  In ``diag.py`` it is the (coarse-grained) truth, and the tangent linear model
  is that of the low-resolution DA model along it.
* Flow map between consecutive references: Phi_n(x) = model.forecast_days(x, 1),
  M_n = dPhi_n/dx (x_n), applied to a block of k perturbations at once.
* Orthonormality in the energy norm: <a, b>_E = <S a, S b>,
  S = model.energy_components (linear map psi -> 5 energy fields).

Benettin / QR step:   M_n V_n = V_{n+1} R_{n+1},  S V_{n+1} orthonormal (k columns)
Exponents:            lambda_i = mean_n log R_{n+1}[i, i] / interval
Ginelli (CLV):        C_n = R_{n+1}^{-1} C_{n+1} (columns renormalised),  CLV_n = V_n C_n

Projections are done on the fly during the forward sweep (only the k BLV
coefficients of each error are kept, never the vectors themselves), and turned
into CLV (oblique) coefficients after the backward sweep: since
span(CLV_1..j) = span(BLV_1..j), P e = Q c = Q C a  =>  a = C^{-1} c.
"""
import numpy as np


# ---------------------------------------------------------------------------
# tangent-linear propagators
# ---------------------------------------------------------------------------
def make_tlm_torch(model, n_days=1):
    """Forward-mode TLM (exact) with torch.func.jvp. V: (k, 2, ny, nx) torch tensor."""
    import torch

    def phi(x):
        return model.forecast_days(x, n_days)

    def tlm(x, V):
        k = V.shape[0]
        xs = x.unsqueeze(0).expand(k, *x.shape).contiguous()
        _, MV = torch.func.jvp(phi, (xs,), (V,))
        return MV

    return tlm


def make_tlm_fd(model, n_days=1, eps=1e-6):
    """Central finite-difference TLM (any backend, use float64). eps is relative to the state rms."""

    def tlm(x, V):
        k = V.shape[0]
        xs = model.xp.stack([x] * k, axis=0)
        nx_ = float((x**2).mean() ** 0.5)
        nv = ((V**2).reshape(k, -1).mean(1) ** 0.5).reshape(k, 1, 1, 1)
        h = eps * nx_ / nv
        fp = model.forecast_days(xs + h * V, n_days)
        fm = model.forecast_days(xs - h * V, n_days)
        return (fp - fm) / (2 * h)

    return tlm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def energy_coords(model, e):
    """Energy-norm coordinates of states e (..., 2, ny, nx) -> (..., dim_z) numpy.

    |z|^2 = 2 * domain-mean total energy of e."""
    Z = _to_np(model.energy_components(e))
    n = Z.shape[-1] * Z.shape[-2]
    return Z.reshape(Z.shape[:-3] + (-1,)) / np.sqrt(n)


def _right_solve(xp, W, R):
    """V = W R^{-1} on the batch (column) dimension of W (k, ...)."""
    Rinv = np.linalg.inv(R)
    k = W.shape[0]
    Wf = W.reshape(k, -1)
    V = Rinv.T @ Wf if xp.name == "numpy" else xp.asarray(Rinv.T) @ Wf
    return V.reshape(W.shape)


def _solve_triangular(R, C):
    from scipy.linalg import solve_triangular

    return solve_triangular(R, C, lower=False)


# ---------------------------------------------------------------------------
# main computation
# ---------------------------------------------------------------------------
def lyapunov_vectors(
    model,
    traj,
    k=64,
    tlm=None,
    n_transient_fwd=30,
    n_transient_bwd=30,
    compute_clv=True,
    interval_days=1,
    errors=None,
    snapshot_index=(),
    seed=0,
    verbose=True,
):
    """Compute k leading BLVs (and CLVs) along ``traj`` and project errors on them.

    traj:   (N+1, 2, ny, nx) reference states (numpy or backend array), daily.
    errors: optional dict name -> (N+1, 2, ny, nx) numpy error fields aligned with traj
            (NaN where unavailable).
    snapshot_index: traj indices at which the BLV fields are kept (for figures).

    Output times are n in [n_transient_fwd, N - n_transient_bwd].
    Returns a dict with
        exponents  (k,)          mean exponents [1/day] after the forward transient
        local_exp  (n_out, k)    finite-time exponents log R_ii / interval
        index      (n_out,)      traj indices of the output times
        C          (n_out, k, k) Ginelli coefficients (CLV = BLV @ C)      [if compute_clv]
        proj       dict name -> dict(c=(n_out,k) BLV coeffs, tot=(n_out,) |e|_E^2)
        snapshots  dict index -> (k, 2, ny, nx) BLV fields (numpy, energy-normalised)
    """
    xp = model.xp
    tlm = tlm or make_tlm_fd(model, interval_days)
    errors = errors or {}
    N = traj.shape[0] - 1
    n_out_start = n_transient_fwd
    n_out_end = N - n_transient_bwd
    if n_out_start < 1 or n_out_end < n_out_start:
        raise ValueError("need n_transient_fwd >= 1 and a trajectory longer than both transients")
    rng = np.random.default_rng(seed)

    V = xp.asarray(rng.standard_normal((k,) + tuple(traj.shape[1:])))
    Qz, R = np.linalg.qr(energy_coords(model, V).T)
    V = _right_solve(xp, V, R)

    proj = {name: dict(c=[], tot=[]) for name in errors}
    Rs, logs, snaps = [], [], {}
    snapshot_index = set(int(i) for i in snapshot_index)
    for n in range(N):
        W = tlm(traj[n], V)
        Qz, R = np.linalg.qr(energy_coords(model, W).T)  # (dz, k), (k, k)
        sgn = np.sign(np.diag(R))
        sgn[sgn == 0] = 1
        Qz, R = Qz * sgn, R * sgn[:, None]
        V = _right_solve(xp, W, R)  # BLVs at time n+1
        logs.append(np.log(np.abs(np.diag(R))) / interval_days)
        m = n + 1
        if m >= n_out_start:
            Rs.append(R)
        if n_out_start <= m <= n_out_end:
            for name, err in errors.items():
                ze = energy_coords(model, xp.asarray(np.nan_to_num(err[m])))
                proj[name]["c"].append(Qz.T @ ze if np.isfinite(err[m]).all() else np.full(k, np.nan))
                proj[name]["tot"].append(float(ze @ ze) if np.isfinite(err[m]).all() else np.nan)
            if m in snapshot_index:
                snaps[m] = _to_np(V).astype(np.float32)
        if verbose and (n % 20 == 0 or n == N - 1):
            lam = np.mean(logs[n_transient_fwd:] or logs, 0)
            print(f"[lyap] {n+1}/{N} lambda_1={lam[0]:.3f}/day  n_pos={(lam > 0).sum()}/{k}", flush=True)

    logs = np.array(logs)
    exps = logs[n_transient_fwd:].mean(0) if N > n_transient_fwd else logs.mean(0)
    out = dict(
        exponents=exps,
        local_exp=logs[max(n_out_start - 1, 0):n_out_end],
        index=np.arange(n_out_start, n_out_end + 1),
        proj={name: dict(c=np.array(p["c"]), tot=np.array(p["tot"])) for name, p in proj.items()},
        snapshots=snaps,
    )

    if compute_clv:
        # Rs[j] = R_m with m = n_out_start + j (maps BLV at m-1 to BLV at m)
        C = np.triu(rng.standard_normal((k, k)))
        C /= np.linalg.norm(C, axis=0, keepdims=True)
        Cs = {}
        for m in range(N, n_out_start, -1):
            if m <= n_out_end:
                Cs[m] = C.copy()
            C = _solve_triangular(Rs[m - n_out_start], C)
            C /= np.linalg.norm(C, axis=0, keepdims=True)
        Cs[n_out_start] = C.copy()
        out["C"] = np.stack([Cs[m] for m in out["index"]])
    return out


# ---------------------------------------------------------------------------
# decomposition of the projected errors
# ---------------------------------------------------------------------------
def classify_exponents(exponents, neutral_tol=0.01):
    """Masks (unstable, neutral, stable) for exponents in 1/day."""
    exponents = np.asarray(exponents)
    neutral = np.abs(exponents) <= neutral_tol
    return exponents > neutral_tol, neutral, exponents < -neutral_tol


def decompose(lyap, name, neutral_tol=0.01):
    """Energy fractions of the error ``name`` in the Lyapunov subspaces.

    Orthogonal (BLV) split, additive:
        blv_unstable + blv_neutral + blv_stable_resolved + residual = 1
        (residual = part outside the span of the k leading vectors, i.e. in the
         unresolved, most stable directions -> choose k > number of unstable exponents)
    Oblique (CLV) split of the resolved part P e (not additive, relative to |P e|^2):
        clv_unstable, clv_neutral, clv_stable_resolved
    Also returns the absolute error energies (``*_abs``, m2 s-2 per unit mass x 2)
    and the mean BLV / CLV coefficient spectrum (``spec_blv``, ``spec_clv``).
    """
    uns, neu, sta = classify_exponents(lyap["exponents"], neutral_tol)
    c, tot = lyap["proj"][name]["c"], lyap["proj"][name]["tot"]
    c2 = c**2
    res = dict(
        total=tot,
        blv_unstable=c2[:, uns].sum(-1) / tot,
        blv_neutral=c2[:, neu].sum(-1) / tot,
        blv_stable_resolved=c2[:, sta].sum(-1) / tot,
        residual=1.0 - c2.sum(-1) / tot,
        spec_blv=np.nanmean(c2 / tot[:, None], 0),
    )
    for key in ["blv_unstable", "blv_neutral", "blv_stable_resolved", "residual"]:
        res[key + "_abs"] = res[key] * tot
    if "C" in lyap:
        C = lyap["C"]
        good = np.isfinite(c).all(-1)
        a = np.full_like(c, np.nan)
        a[good] = np.linalg.solve(C[good], c[good][..., None])[..., 0]
        pnorm = c2.sum(-1)
        for key, msk in [("unstable", uns), ("neutral", neu), ("stable_resolved", sta)]:
            comp = np.einsum("nij,nj->ni", C[:, :, msk], np.nan_to_num(a[:, msk]))
            res[f"clv_{key}"] = np.where(good, (comp**2).sum(-1) / pnorm, np.nan)
        res["spec_clv"] = np.nanmean(a**2 / pnorm[:, None], 0)
    return res
