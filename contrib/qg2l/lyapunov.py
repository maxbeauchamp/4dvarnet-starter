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
# checkpointing of the forward sweep
# ---------------------------------------------------------------------------
def _save_lyap_checkpoint(ckdir, key, n_next, V, logs, Rs, n_rs_saved, proj, names, snaps):
    """R factors: append-only chunks R_<first index>.npy; everything else: state.npz (atomic)."""
    import json
    import os
    from pathlib import Path

    ckdir = Path(ckdir)
    ckdir.mkdir(parents=True, exist_ok=True)
    if len(Rs) > n_rs_saved:
        np.save(ckdir / f"R_{n_rs_saved:06d}.npy", np.stack(Rs[n_rs_saved:]))
    k = V.shape[0]
    arrs = dict(key=np.array(key), n_next=np.array(n_next), V=V, logs=np.array(logs), n_rs=np.array(len(Rs)),
                names=np.array(json.dumps(names)),
                snap_idx=np.array(sorted(snaps), dtype=int),
                snap_val=np.stack([snaps[i] for i in sorted(snaps)]) if snaps else np.zeros((0,) + V.shape, np.float32))
    for i, nm in enumerate(names):
        c = proj[nm]["c"]
        arrs[f"c_{i}"] = np.stack(c) if c else np.zeros((0, k))
        arrs[f"tot_{i}"] = np.array(proj[nm]["tot"], dtype=float)
    tmp = ckdir / "state.tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, ckdir / "state.npz")
    return len(Rs)


def _load_lyap_checkpoint(ckdir, key, names, k):
    import json
    from pathlib import Path

    f = Path(ckdir) / "state.npz"
    if not f.exists():
        return None
    ck = np.load(f, allow_pickle=False)
    if str(ck["key"]) != key or json.loads(str(ck["names"])) != names or ck["V"].shape[0] != k:
        print(f"[lyap] ignoring checkpoint in {ckdir} (other configuration)", flush=True)
        return None
    n_rs = int(ck["n_rs"])
    Rs = []
    for chunk in sorted(Path(ckdir).glob("R_*.npy")):
        if int(chunk.stem[2:]) < n_rs:
            Rs.extend(list(np.load(chunk)))
    Rs = Rs[:n_rs]
    if len(Rs) != n_rs:
        print(f"[lyap] incomplete checkpoint in {ckdir}, restarting", flush=True)
        return None
    proj = {nm: dict(c=list(ck[f"c_{i}"]), tot=list(ck[f"tot_{i}"])) for i, nm in enumerate(names)}
    snaps = {int(i): v for i, v in zip(ck["snap_idx"], ck["snap_val"])}
    return int(ck["n_next"]), ck["V"], list(ck["logs"]), Rs, proj, snaps


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
    checkpoint_dir=None,
    checkpoint_key="",
    checkpoint_every=10,
):
    """Compute k leading BLVs (and CLVs) along ``traj`` and project errors on them.

    checkpoint_dir: if given, the forward sweep is saved every ``checkpoint_every`` steps (BLVs,
        exponents, projections; the R factors needed by the backward CLV sweep in append-only chunks)
        and a call with the same ``checkpoint_key`` resumes from the last saved step.

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
    names = sorted(errors)
    n_start, n_rs_saved = 0, 0
    if checkpoint_dir is not None:
        ck = _load_lyap_checkpoint(checkpoint_dir, checkpoint_key, names, k)
        if ck is not None:
            n_start, V, logs, Rs, proj, snaps = ck
            V = xp.asarray(V)
            n_rs_saved = len(Rs)
            if verbose:
                print(f"[lyap] resuming from {checkpoint_dir} at step {n_start}/{N}", flush=True)
    for n in range(n_start, N):
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
        if checkpoint_dir is not None and ((n + 1) % checkpoint_every == 0 or n == N - 1):
            n_rs_saved = _save_lyap_checkpoint(checkpoint_dir, checkpoint_key, n + 1, _to_np(V), logs, Rs,
                                               n_rs_saved, proj, names, snaps)

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


def _orth(A, rtol=1e-10):
    """Orthonormal basis of span(A) (SVD, numerically rank-revealing)."""
    if A.shape[1] == 0:
        return A
    U, sv, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, sv > rtol * sv[0]]


def _clv_split(C, c, n_u, n_n):
    """CLV-subspace diagnostics of P e (BLV coordinates c).

    U = span(CLV_1..n_u) (= span(BLV_1..n_u)), N = span(neutral CLVs), S = span(stable CLVs).
    The *oblique* split e = e_U + e_N + e_S is ill-conditioned: the smallest principal angle
    between U and S is typically a few degrees (near-tangencies), so the oblique components are
    amplified by ~1/sin^2(angle) and cancel each other. We therefore report the *orthogonal*
    projections on each CLV subspace, |P_X e|^2 / |P e|^2 in [0, 1] (not additive), plus the
    smallest principal angle between U and S (angle_us_deg).
    Exponents are assumed ordered (first n_u unstable, then n_n neutral)."""
    n, k = c.shape
    out = {key: np.full(n, np.nan) for key in ["clv_unstable", "clv_neutral", "clv_stable_resolved", "angle_us_deg"]}
    for t in range(n):
        if not np.isfinite(c[t]).all():
            continue
        pn = (c[t] ** 2).sum()
        Bn = _orth(C[t][:, n_u:n_u + n_n])
        Bs = _orth(C[t][:, n_u + n_n:])
        out["clv_unstable"][t] = (c[t][:n_u] ** 2).sum() / pn
        out["clv_neutral"][t] = ((Bn.T @ c[t]) ** 2).sum() / pn
        out["clv_stable_resolved"][t] = ((Bs.T @ c[t]) ** 2).sum() / pn
        if n_u and Bs.shape[1]:
            cos = np.linalg.svd(Bs[:n_u], compute_uv=False).max()  # Eu^T Bs = first n_u rows of Bs
            out["angle_us_deg"][t] = np.degrees(np.arccos(min(cos, 1.0)))
    return out


def decompose(lyap, name, neutral_tol=0.01):
    """Energy fractions of the error ``name`` in the Lyapunov subspaces.

    Orthogonal (BLV) split, additive:
        blv_unstable + blv_neutral + blv_stable_resolved + residual = 1
        (residual = part outside the span of the k leading vectors, i.e. in the
         unresolved, most stable directions -> choose k > number of unstable exponents)
    CLV subspaces (see ``_clv_split``): orthogonal projections of P e on the unstable, neutral
    and stable CLV subspaces, |P_X e|^2 / |P e|^2 in [0, 1] (not additive; the oblique split is
    ill-conditioned because of near-tangencies), and angle_us_deg = smallest principal angle
    between the unstable and stable CLV subspaces.
    Also returns the absolute error energies (``*_abs``, m2 s-2 per unit mass x 2)
    and the mean BLV coefficient spectrum (``spec_blv``).
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
        res.update(_clv_split(lyap["C"], c, int(uns.sum()), int(neu.sum())))
    return res
