"""
Classical (nonlinear, full) 4DVar on the two-layer QG testbed, adjoint by torch autograd.

DA model = the low-resolution QG2L (``model_kw`` of the dataset), which differs
from the truth (high-resolution run spectrally truncated to the DA grid) by the
unresolved scales -> genuine, structured model error.

Control-variable transform (preconditioning), B homogeneous & isotropic:

    x0 = xb + B^{1/2} v,     B^{1/2}: psi_hat(k) = S(|k|) v_hat(k),  S(|k|) 2x2 lower triangular
    S(|k|) S(|k|)^T = b_scale^2 * Sigma_clim(|k|) / (nx ny)

with Sigma_clim(|k|) the 2x2 (layer x layer) isotropic spectral covariance of the
truth anomalies over the training period.

strong constraint:  J(v)    = 1/2 |v|^2 + 1/2 sum_d |R^{-1/2}(y_d - H M_{0->d}(x0))|^2
weak  constraint :  J(v, w) = 1/2 |v|^2 + 1/2 sum_d |w_d|^2 + Jo,
                     x_{d+1} = M_1day(x_d) + Q^{1/2} w_{d+1},   Q^{1/2} = (q_scale / b_scale) B^{1/2}
                     (model-error forcing formulation, daily)

H = f0/g * psi_1 at the observed pixels of day d;  R = diag((r_inflation * obs_sigma)^2).
Minimisation: L-BFGS (strong-Wolfe line search) on the control variables.
Cycling: consecutive windows of ``window_days``; the background of a window is the
1-day forecast of the last analysed state of the previous one. The reconstruction
of a window is its analysed trajectory (4D smoother within the window).
"""
import json
import time
from pathlib import Path

import numpy as np

from .qgmodel import QG2L


# ---------------------------------------------------------------------------
# background-error covariance (spectral, isotropic, 2 layers)
# ---------------------------------------------------------------------------
def isotropic_layer_covariance(psi, dx):
    """psi (T, 2, n, n) anomalies -> dict with bin edges and Sigma (n_bins, 2, 2) of rfft2 coeffs."""
    psi = np.asarray(psi, dtype=np.float64)
    n = psi.shape[-1]
    ph = np.fft.rfft2(psi)  # (T, 2, n, n//2+1)
    kx = 2 * np.pi * np.fft.rfftfreq(n, dx)
    ky = 2 * np.pi * np.fft.fftfreq(n, dx)
    kk = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    dk = 2 * np.pi / (n * dx)
    kbin = np.rint(kk / dk).astype(int)
    nb = kbin.max() + 1
    cov = np.einsum("tikl,tjkl->klij", ph, ph.conj()).real / psi.shape[0]  # (n, n//2+1, 2, 2)
    Sig = np.zeros((nb, 2, 2))
    cnt = np.bincount(kbin.ravel(), minlength=nb)
    for i in range(2):
        for j in range(2):
            Sig[:, i, j] = np.bincount(kbin.ravel(), weights=cov[..., i, j].ravel(), minlength=nb)
    Sig /= np.maximum(cnt, 1)[:, None, None]
    return dict(kbin=kbin, Sigma=Sig)


def sqrt_cov_field(cov, n, scale=1.0, eps=1e-12):
    """Per-wavenumber lower Cholesky factor S (2, 2, n, n//2+1) with S S^T = scale^2 Sigma / n^2."""
    Sig = cov["Sigma"] * scale**2 / n**2
    S = np.zeros_like(Sig)
    for b in range(len(Sig)):
        A = Sig[b] + eps * np.eye(2) * max(np.trace(Sig[b]), 1e-30)
        if b == 0 or np.trace(Sig[b]) <= 0:
            continue  # mean mode is not controlled (psi has zero mean)
        S[b] = np.linalg.cholesky(A)
    return np.moveaxis(S[cov["kbin"]], (0, 1), (2, 3)).copy()  # (2,2,n,nk)


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------
def conjugate_gradient(A, b, n_iter, rtol=1e-8):
    """Solve A x = b for a symmetric positive definite operator A (callable) by CG, x0 = 0.
    Backend agnostic (numpy arrays or torch tensors). Returns (x, number of A applications)."""
    x = b * 0
    r = b * 1
    p = r * 1
    rs = float((r * r).sum())
    b_norm = rs ** 0.5
    it = 0
    for it in range(1, n_iter + 1):
        Ap = A(p)
        alpha = rs / float((p * Ap).sum())
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float((r * r).sum())
        if rs_new ** 0.5 <= rtol * b_norm:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, it


# ---------------------------------------------------------------------------
# 4DVar
# ---------------------------------------------------------------------------
class FourDVar:
    def __init__(self, model_kw, b_cov, b_scale=0.5, mode="strong", q_scale=0.05, r_inflation=1.0,
                 window_days=10, n_iter=60, history_size=20, device="cpu", dtype="float64", verbose=True,
                 solver="lbfgs", n_outer=3, n_inner=20):
        import torch

        self.torch = torch
        self.model = QG2L(backend="torch", device=device, dtype=dtype, **model_kw)
        self.kw = model_kw
        self.n = self.model.nx
        self.mode = mode
        self.window_days = int(window_days)
        self.n_iter = int(n_iter)
        self.history_size = history_size
        if solver not in ("lbfgs", "incremental"):
            raise ValueError(solver)
        self.solver, self.n_outer, self.n_inner = solver, int(n_outer), int(n_inner)
        self.r_inflation = float(r_inflation)
        self.verbose = verbose
        self.dtype = getattr(torch, dtype)
        self.device = torch.device(device)
        self.b_cov = b_cov
        self.set_scales(b_scale, q_scale)

    def set_scales(self, b_scale, q_scale):
        self.b_scale, self.q_scale = float(b_scale), float(q_scale)
        self.Sb = self.model.xp.asarray(sqrt_cov_field(self.b_cov, self.n, self.b_scale), cplx=True)
        self.Sq = self.model.xp.asarray(sqrt_cov_field(self.b_cov, self.n, self.q_scale), cplx=True)

    def _apply_sqrt(self, S, v):
        vh = self.model.fft(v)
        a, b = self.model._mat(S, vh)
        return self.model.ifft(self.model._stack2(a, b))

    def ssh(self, psi):
        return self.model.ssh(psi)

    # -- one window ---------------------------------------------------------------
    def analyse_window(self, xb, obs, sigma, b_scale=None):
        """xb (2,n,n) torch; obs/sigma (W,n,n) numpy (NaN = no obs). Returns (traj (W,2,n,n) torch, info)."""
        torch = self.torch
        if b_scale is not None and b_scale != self.b_scale:
            Sb = self.model.xp.asarray(sqrt_cov_field(self.b_cov, self.n, b_scale), cplx=True)
        else:
            Sb = self.Sb
        W = obs.shape[0]
        msk = torch.as_tensor(np.isfinite(obs), device=self.device)
        y = torch.as_tensor(np.nan_to_num(obs), dtype=self.dtype, device=self.device)
        rinv = torch.as_tensor(1.0 / (self.r_inflation * np.where(np.isfinite(sigma), sigma, 1.0)),
                               dtype=self.dtype, device=self.device)
        n_obs = int(msk.sum())
        v = torch.zeros((2, self.n, self.n), dtype=self.dtype, device=self.device, requires_grad=True)
        params = [v]
        if self.mode == "weak":
            w = torch.zeros((W - 1, 2, self.n, self.n), dtype=self.dtype, device=self.device, requires_grad=True)
            params.append(w)

        def trajectory_of(v_, w_):
            x = xb + self._apply_sqrt(Sb, v_)
            out = [x]
            for d in range(1, W):
                x = self.model.forecast_days(x, 1)
                if self.mode == "weak":
                    x = x + self._apply_sqrt(self.Sq, w_[d - 1])
                out.append(x)
            return torch.stack(out)

        def trajectory():
            return trajectory_of(v, w if self.mode == "weak" else None)

        if self.solver == "incremental":
            return self._incremental(trajectory_of, v, w if self.mode == "weak" else None, y, rinv, msk, n_obs, W)

        def cost():
            traj = trajectory()
            innov = (y - self.ssh(traj)) * rinv
            jo = 0.5 * (innov[msk] ** 2).sum()
            jb = 0.5 * (v**2).sum()
            jq = 0.5 * (w**2).sum() if self.mode == "weak" else torch.zeros((), dtype=self.dtype)
            return jb + jq + jo, (float(jb), float(jq), float(jo))

        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=self.n_iter, history_size=self.history_size,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
        hist = []

        def closure():
            opt.zero_grad()
            J, parts = cost()
            J.backward()
            hist.append((float(J),) + parts)
            return J

        with torch.no_grad():
            J0 = float(cost()[0])
        opt.step(closure)
        with torch.no_grad():
            traj = trajectory()
            J1, parts = cost()
        info = dict(J0=J0, J=float(J1), jb=parts[0], jq=parts[1], jo=parts[2], n_obs=n_obs,
                    n_eval=len(hist), jo_per_obs=parts[2] / max(n_obs, 1),
                    n_nonlinear=len(hist), n_tl=0, n_adjoint=len(hist))
        return traj.detach(), info

    def _incremental(self, trajectory_of, v, w, y, rinv, msk, n_obs, W):
        """Incremental (Gauss-Newton) 4DVar: outer loops re-linearise the model around the current
        trajectory, inner loops solve the quadratic problem with preconditioned conjugate gradient:

            (I + G'^T G') dz = -(z - G'^T r),   r = R^{-1/2}(y - H M(xb + B^{1/2} v)),  z = (v [, w])

        The B^{1/2} control-variable transform makes the background Hessian the identity, so CG works on
        I + G'^T G' (eigenvalues >= 1). G' = tangent linear (torch.func.jvp), G'^T = adjoint (torch.func.vjp).
        Cost per inner iteration: one TL + one adjoint integration over the window."""
        torch = self.torch
        nv = v.numel()
        mskf = msk.to(self.dtype)
        shp_w = (W - 1, 2, self.n, self.n)

        def split(z):
            return z[:nv].reshape(v.shape), (z[nv:].reshape(shp_w) if w is not None else None)

        def G(z):  # scaled model equivalent of the observations (0 where no obs)
            v_, w_ = split(z)
            return self.ssh(trajectory_of(v_, w_)) * rinv * mskf

        yr = y * rinv * mskf

        def cost(z):
            with torch.no_grad():
                r = yr - G(z)
                jb = 0.5 * float((z[:nv] ** 2).sum())
                jq = 0.5 * float((z[nv:] ** 2).sum())
                jo = 0.5 * float((r ** 2).sum())
            return jb + jq + jo, (jb, jq, jo)

        z = torch.zeros(nv + (w.numel() if w is not None else 0), dtype=self.dtype, device=self.device)
        J0 = cost(z)[0]
        n_tl = n_ad = n_nl = 0
        for _ in range(self.n_outer):
            Gz, vjp_fn = torch.func.vjp(G, z)
            n_nl += 1
            grad = z - vjp_fn(yr - Gz)[0]
            n_ad += 1
            z_lin = z

            def hess_vec(d):
                _, Jd = torch.func.jvp(G, (z_lin,), (d,))
                return d + vjp_fn(Jd)[0]

            dz, it = conjugate_gradient(hess_vec, -grad, self.n_inner)
            n_tl += it
            n_ad += it
            z = (z + dz).detach()
        J1, parts = cost(z)
        n_nl += 1
        with torch.no_grad():
            traj = trajectory_of(*split(z))
        info = dict(J0=J0, J=J1, jb=parts[0], jq=parts[1], jo=parts[2], n_obs=n_obs,
                    n_eval=self.n_outer, jo_per_obs=parts[2] / max(n_obs, 1),
                    n_nonlinear=n_nl, n_tl=n_tl, n_adjoint=n_ad)
        return traj.detach(), info

    # -- cycling --------------------------------------------------------------------
    def run(self, obs, sigma, t_start, t_end, spinup_days=60, xb0=None, b_scale_first=1.0, truth=None,
            background=None):
        """Cycled 4DVar over [t_start - spinup_days, t_end). obs/sigma: full (T, n, n) arrays.

        background: optional (t_end - t_start, 2, n, n) psi (e.g. a 4DVarNet reconstruction): the
            background of every window starting in [t_start, t_end) is taken from it instead of the
            cycled forecast (hybrid 4DVarNet -> 4DVar; b_scale then sizes the 4DVarNet error).
        Returns the analysed trajectory on [t_start, t_end) (numpy, psi), and a log list.
        truth (optional, (T,2,n,n) numpy psi) is only used for logging RMSEs."""
        torch = self.torch
        t0 = t_start - spinup_days
        xb = torch.zeros((2, self.n, self.n), dtype=self.dtype, device=self.device) if xb0 is None else \
            torch.as_tensor(xb0, dtype=self.dtype, device=self.device)
        out = np.full((t_end - t_start, 2, self.n, self.n), np.nan, dtype=np.float32)
        log = []
        t, first, tic = t0, True, time.time()
        while t < t_end:
            W = min(self.window_days, t_end - t)
            if background is not None and t_start <= t < t_end:
                xb = torch.as_tensor(background[t - t_start], dtype=self.dtype, device=self.device)
                first = False
            traj, info = self.analyse_window(xb, obs[t:t + W], sigma[t:t + W],
                                             b_scale=b_scale_first if first else None)
            first = False
            tn = traj.cpu().numpy()
            info.update(t=t, W=W, elapsed=time.time() - tic)
            if truth is not None:
                f = self.kw["f0"] / self.kw["g"]
                info["rmse_ssh"] = float(np.sqrt(np.mean((f * (tn[:, 0] - truth[t:t + W, 0])) ** 2)))
                info["rmse_l2"] = float(np.sqrt(np.mean((f * (tn[:, 1] - truth[t:t + W, 1])) ** 2)))
            log.append(info)
            if self.verbose:
                print("[4dvar] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in info.items()), flush=True)
            for d in range(W):
                if t_start <= t + d < t_end:
                    out[t + d - t_start] = tn[d]
            with torch.no_grad():
                xb = self.model.forecast_days(traj[-1], 1)
            t += W
        return out, log


def free_run(model_kw, psi0, n_days, every=5, discard=30):
    """Daily free run of the DA model from psi0 (numpy); returns every ``every``-th state after ``discard`` days."""
    m = QG2L(**model_kw)
    p, out = np.asarray(psi0, dtype=np.float64), []
    for d in range(n_days + discard):
        p = m.forecast_days(p, 1)
        if d >= discard and (d - discard) % every == 0:
            out.append(p)
    return np.stack(out)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run_4dvar(data_path, out_path, mode="strong", window_days=10, b_scale=0.5, q_scale=0.05,
              r_inflation=1.0, n_iter=60, spinup_days=60, b_scale_first=1.0, split="test",
              t_range=None, device="cuda", dtype="float64", model_overrides=None, b_source="truth",
              background_path=None, method=None, solver="lbfgs", n_outer=3, n_inner=20):
    """Run the cycled 4DVar on a split of the OSSE and write a netCDF like LitQG2L's.

    b_source: "truth"    -> B from the training truth (oracle-like statistics)
              "free_run" -> B from a free run of the DA (low-resolution) model over the training
                            period length (model climatology: no knowledge of the truth statistics)
    background_path: reconstruction file (e.g. 4dvarnet.nc) providing the background of each window
              (hybrid scheme); use spinup_days=0 and a b_scale matching its error level.
    solver:   "lbfgs" (nonlinear L-BFGS on the preconditioned cost, n_iter iterations) or "incremental"
              (operational-style Gauss-Newton: n_outer outer loops x n_inner preconditioned CG iterations
              with tangent-linear / adjoint models).
    """
    import torch
    import xarray as xr

    from . import metrics as qg_metrics

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[4dvar] CUDA not available, using CPU")
        device = "cpu"
    ds = xr.open_dataset(data_path, decode_times=False).load()
    kw = json.loads(ds.attrs["model_kw"])
    kw.update(model_overrides or {})
    splits = json.loads(ds.attrs["splits"])
    t_start, t_end = t_range or splits[split]
    psi = ds.psi.values.astype(np.float64)
    tr0, tr1 = splits["train"]
    if b_source == "truth":
        b_cov = isotropic_layer_covariance(psi[tr0:tr1:5], kw["L"] / kw["nx"])
    elif b_source == "free_run":
        b_cov = isotropic_layer_covariance(free_run(kw, psi[tr0], tr1 - tr0), kw["L"] / kw["nx"])
    else:
        raise ValueError(b_source)
    background = None
    if background_path is not None:
        bg = xr.open_dataset(background_path, decode_times=False).load()
        background = bg.psi.sel(time=ds.time.values[t_start:t_end], method="nearest").values.astype(np.float64)
        if not np.isfinite(background).all():
            raise ValueError(f"{background_path}: NaN in the background (deep layer not estimated?)")
    da = FourDVar(kw, b_cov, b_scale=b_scale, mode=mode, q_scale=q_scale, r_inflation=r_inflation,
                  window_days=window_days, n_iter=n_iter, device=device, dtype=dtype, solver=solver,
                  n_outer=n_outer, n_inner=n_inner)
    rec, log = da.run(ds.obs_ssh.values, ds.obs_sigma.values, t_start, t_end, spinup_days=spinup_days,
                      b_scale_first=b_scale_first, truth=psi, background=background)
    cost = {k: int(sum(l.get(k, 0) for l in log)) for k in ["n_nonlinear", "n_tl", "n_adjoint"]}
    print("[4dvar] model integrations over the run:", cost)
    f = kw["f0"] / kw["g"]
    scores = qg_metrics.all_scores(f * rec, f * psi[t_start:t_end], dx=kw["L"] / kw["nx"])
    print("\n" + "\n".join(f"| {k:16s} | {v:10.4f} |" for k, v in scores.items()))
    cfg = dict(mode=mode, window_days=window_days, b_scale=b_scale, q_scale=q_scale, r_inflation=r_inflation,
               n_iter=n_iter, spinup_days=spinup_days, model_overrides=model_overrides or {}, b_source=b_source,
               background_path=str(background_path) if background_path else None, solver=solver,
               n_outer=n_outer, n_inner=n_inner)
    method = method or (f"hybrid_{mode}" if background_path else f"4dvar_{mode}")
    out = xr.Dataset(
        dict(psi=(("time", "layer", "y", "x"), rec, dict(units="m2 s-1", long_name=f"{method} analysis"))),
        coords=dict(time=ds.time.values[t_start:t_end], layer=[1, 2], y=ds.y.values, x=ds.x.values),
        attrs=dict(method=method, model_kw=json.dumps(kw), scores=json.dumps(scores),
                   config=json.dumps(cfg), log=json.dumps(log), layers_estimated=json.dumps([1, 2]),
                   cost=json.dumps(cost)),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    print("[4dvar] wrote", out_path)
    return str(out_path)
