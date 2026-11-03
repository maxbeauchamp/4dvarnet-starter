"""
Two-layer quasi-geostrophic model on a doubly periodic beta-plane.

Pseudo-spectral, RK4 time stepping, exponential spectral filter for small-scale
dissipation (same formulation as pyqg's ``QGModel``):

    q1 = lap(psi1) + F1 (psi2 - psi1)
    q2 = lap(psi2) + F2 (psi1 - psi2)          F1 = 1 / (rd^2 (1 + delta)),  F2 = delta F1,  delta = H1 / H2

    dq_i/dt + J(psi_i, q_i) + U_i dq_i/dx + Qy_i dpsi_i/dx = - rek * lap(psi2) * [i == 2]

    Qy1 = beta + F1 (U1 - U2),   Qy2 = beta - F2 (U1 - U2)

The same code runs with numpy (fast checks, CPU) or torch (autograd / forward-mode
tangent linear for 4DVar and Lyapunov vectors, GPU). States are physical-space
streamfunctions of shape (..., 2, ny, nx) in m^2/s; ssh = f0 / g * psi1.
"""
import math
import numpy as np

SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# minimal array backend so that the model runs with numpy or torch
# ---------------------------------------------------------------------------
class _NumpyBackend:
    name = "numpy"

    def __init__(self, dtype="float64", device=None):
        self.real = np.dtype(dtype)
        self.cplx = np.complex128 if self.real == np.float64 else np.complex64

    def asarray(self, x, cplx=False):
        return np.asarray(x, dtype=self.cplx if cplx else self.real)

    def rfft2(self, x):
        return np.fft.rfft2(x)

    def irfft2(self, xh, s):
        return np.fft.irfft2(xh, s=s).astype(self.real, copy=False)

    def stack(self, xs, axis):
        return np.stack(xs, axis=axis)

    def concat(self, xs, axis):
        return np.concatenate(xs, axis=axis)


class _TorchBackend:
    name = "torch"

    def __init__(self, dtype="float64", device="cpu"):
        import torch

        self.torch = torch
        self.real = getattr(torch, dtype)
        self.cplx = torch.complex128 if self.real == torch.float64 else torch.complex64
        self.device = torch.device(device)

    def asarray(self, x, cplx=False):
        t = self.torch
        dtype = self.cplx if cplx else self.real
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=dtype)
        return t.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def rfft2(self, x):
        return self.torch.fft.rfft2(x)

    def irfft2(self, xh, s):
        return self.torch.fft.irfft2(xh, s=s)

    def stack(self, xs, axis):
        return self.torch.stack(xs, dim=axis)

    def concat(self, xs, axis):
        return self.torch.cat(xs, dim=axis)


def get_backend(backend="numpy", dtype="float64", device="cpu"):
    if backend == "numpy":
        return _NumpyBackend(dtype)
    if backend == "torch":
        return _TorchBackend(dtype, device)
    raise ValueError(backend)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class QG2L:
    """Two-layer QG model. Default parameters = pyqg 'eddy' configuration (Ross et al. 2023)."""

    def __init__(
        self,
        nx=64,
        L=1e6,
        dt=3600.0,
        beta=1.5e-11,
        rd=15e3,
        delta=0.25,
        H1=500.0,
        U1=0.025,
        U2=0.0,
        rek=5.787e-7,
        f0=1e-4,
        g=9.81,
        filterfac=23.6,
        backend="numpy",
        dtype="float64",
        device="cpu",
    ):
        self.nx = self.ny = int(nx)
        self.L = float(L)
        self.dx = self.L / self.nx
        self.dt = float(dt)
        self.beta, self.rd, self.delta = beta, rd, delta
        self.H1 = H1
        self.H2 = H1 / delta
        self.H = self.H1 + self.H2
        self.U1, self.U2, self.rek = U1, U2, rek
        self.f0, self.g = f0, g
        self.F1 = 1.0 / (rd**2 * (1.0 + delta))
        self.F2 = delta * self.F1
        self.Qy1 = beta + self.F1 * (U1 - U2)
        self.Qy2 = beta - self.F2 * (U1 - U2)
        self.params = dict(
            nx=nx, L=L, dt=dt, beta=beta, rd=rd, delta=delta, H1=H1, U1=U1, U2=U2,
            rek=rek, f0=f0, g=g, filterfac=filterfac,
        )
        self.xp = get_backend(backend, dtype, device)
        self._build_spectral(filterfac)

    # -- setup -------------------------------------------------------------
    def _build_spectral(self, filterfac):
        nx, ny, dx = self.nx, self.ny, self.dx
        kx = 2 * np.pi * np.fft.rfftfreq(nx, dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, dx)
        K, Ly = np.meshgrid(kx, ky)  # (ny, nx//2+1)
        K2 = K**2 + Ly**2
        det = K2 * (K2 + self.F1 + self.F2)
        with np.errstate(divide="ignore"):
            idet = np.where(det > 0, 1.0 / det, 0.0)
        # psi_hat = Minv @ q_hat
        inv = np.zeros((2, 2) + K2.shape)
        inv[0, 0] = -(K2 + self.F2) * idet
        inv[0, 1] = -self.F1 * idet
        inv[1, 0] = -self.F2 * idet
        inv[1, 1] = -(K2 + self.F1) * idet
        # q_hat = M @ psi_hat
        fwd = np.zeros((2, 2) + K2.shape)
        fwd[0, 0] = -K2 - self.F1
        fwd[0, 1] = self.F1
        fwd[1, 0] = self.F2
        fwd[1, 1] = -K2 - self.F2
        # pyqg exponential filter
        cphi = 0.65 * np.pi
        wv = np.sqrt((K * dx) ** 2 + (Ly * dx) ** 2)
        filtr = np.exp(-filterfac * (wv - cphi) ** 4)
        filtr[wv <= cphi] = 1.0

        U = np.array([self.U1, self.U2])[:, None, None]
        Qy = np.array([self.Qy1, self.Qy2])[:, None, None]
        lin_q = -1j * K[None] * U  # acts on q_hat
        lin_p = -1j * K[None] * Qy + np.stack([0 * K2, self.rek * K2])  # acts on psi_hat

        xp = self.xp
        self._np = dict(K=K, L=Ly, K2=K2, inv=inv, fwd=fwd, filtr=filtr)
        self.ik = xp.asarray(1j * K, cplx=True)
        self.il = xp.asarray(1j * Ly, cplx=True)
        self.inv = xp.asarray(inv, cplx=True)
        self.fwd = xp.asarray(fwd, cplx=True)
        self.filtr = xp.asarray(filtr, cplx=True)
        self.lin_q = xp.asarray(lin_q, cplx=True)
        self.lin_p = xp.asarray(lin_p, cplx=True)
        self.K2 = xp.asarray(K2, cplx=True)

    # -- conversions ---------------------------------------------------------
    @staticmethod
    def _mat(M, xh):
        # (2,2,ny,nk) x (...,2,ny,nk) -> (...,2,ny,nk)
        a = M[0, 0] * xh[..., 0, :, :] + M[0, 1] * xh[..., 1, :, :]
        b = M[1, 0] * xh[..., 0, :, :] + M[1, 1] * xh[..., 1, :, :]
        return a, b

    def _stack2(self, a, b):
        return self.xp.stack([a, b], axis=-3)

    def qh_to_ph(self, qh):
        return self._stack2(*self._mat(self.inv, qh))

    def ph_to_qh(self, ph):
        return self._stack2(*self._mat(self.fwd, ph))

    def fft(self, x):
        return self.xp.rfft2(x)

    def ifft(self, xh):
        return self.xp.irfft2(xh, s=(self.ny, self.nx))

    def psi_to_q(self, psi):
        return self.ifft(self.ph_to_qh(self.fft(psi)))

    def q_to_psi(self, q):
        return self.ifft(self.qh_to_ph(self.fft(q)))

    def ssh(self, psi):
        return self.f0 / self.g * psi[..., 0, :, :]

    def psi1_from_ssh(self, ssh):
        return self.g / self.f0 * ssh

    # -- dynamics --------------------------------------------------------------
    def tendency(self, qh):
        ph = self.qh_to_ph(qh)
        u = self.ifft(-self.il * ph)
        v = self.ifft(self.ik * ph)
        qx = self.ifft(self.ik * qh)
        qy = self.ifft(self.il * qh)
        adv = self.fft(u * qx + v * qy)
        return -adv + self.lin_q * qh + self.lin_p * ph

    def step_qh(self, qh):
        dt = self.dt
        k1 = self.tendency(qh)
        k2 = self.tendency(qh + 0.5 * dt * k1)
        k3 = self.tendency(qh + 0.5 * dt * k2)
        k4 = self.tendency(qh + dt * k3)
        return self.filtr * (qh + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))

    def forward(self, psi, n_steps):
        """Integrate n_steps model steps from psi (..., 2, ny, nx)."""
        qh = self.ph_to_qh(self.fft(psi))
        for _ in range(int(n_steps)):
            qh = self.step_qh(qh)
        return self.ifft(self.qh_to_ph(qh))

    @property
    def steps_per_day(self):
        n = SECONDS_PER_DAY / self.dt
        if abs(n - round(n)) > 1e-9:
            raise ValueError("dt must divide one day")
        return int(round(n))

    def forecast_days(self, psi, n_days=1):
        return self.forward(psi, n_days * self.steps_per_day)

    def trajectory(self, psi0, n_days, save_every=1, callback=None):
        """Daily trajectory (n_days//save_every + 1, ..., 2, ny, nx) including psi0."""
        out = [psi0]
        qh = self.ph_to_qh(self.fft(psi0))
        spd = self.steps_per_day
        for d in range(1, n_days + 1):
            for _ in range(spd):
                qh = self.step_qh(qh)
            if d % save_every == 0:
                psi = self.ifft(self.qh_to_ph(qh))
                out.append(psi)
                if callback is not None:
                    callback(d, psi)
        return self.xp.stack(out, axis=0)

    # -- diagnostics -------------------------------------------------------------
    def energy_components(self, psi):
        """Map psi -> z such that 0.5 * mean(z**2) * 5 = total energy per unit mass (m^2/s^2).

        z = [sqrt(d1) psi1_x, sqrt(d1) psi1_y, sqrt(d2) psi2_x, sqrt(d2) psi2_y, sqrt(d1 F1) (psi1 - psi2)]
        with d_i = H_i / H. This linear map defines the energy norm used for Lyapunov vectors.
        """
        ph = self.fft(psi)
        d1, d2 = self.H1 / self.H, self.H2 / self.H
        p1x = self.ifft(self.ik * ph[..., 0, :, :])
        p1y = self.ifft(self.il * ph[..., 0, :, :])
        p2x = self.ifft(self.ik * ph[..., 1, :, :])
        p2y = self.ifft(self.il * ph[..., 1, :, :])
        ape = psi[..., 0, :, :] - psi[..., 1, :, :]
        return self.xp.stack(
            [math.sqrt(d1) * p1x, math.sqrt(d1) * p1y, math.sqrt(d2) * p2x, math.sqrt(d2) * p2y,
             math.sqrt(d1 * self.F1) * ape],
            axis=-3,
        )

    def energy(self, psi):
        z = self.energy_components(psi)
        return 0.5 * (z**2).sum(-3).mean(-1).mean(-1)

    def vorticity(self, psi):
        return self.ifft(-self.K2 * self.fft(psi))

    def velocity(self, psi):
        ph = self.fft(psi)
        return self.ifft(-self.il * ph), self.ifft(self.ik * ph)

    def cfl(self, psi):
        u, v = self.velocity(psi)
        umax = float(abs(u).max() + abs(self.U1))
        return umax * self.dt / self.dx

    def random_init(self, seed=0, amp=1e-6, n_batch=None):
        """Small random PV perturbation (pyqg style)."""
        rng = np.random.default_rng(seed)
        shape = (2, self.ny, self.nx) if n_batch is None else (n_batch, 2, self.ny, self.nx)
        q = amp * rng.standard_normal(shape)
        q = q - q.mean(axis=(-1, -2), keepdims=True)
        return self.q_to_psi(self.xp.asarray(q))


# ---------------------------------------------------------------------------
# grid utilities
# ---------------------------------------------------------------------------
def spectral_coarsen(field, n_out):
    """Spectral truncation of a doubly periodic field (..., n, n) to (..., n_out, n_out) (numpy)."""
    field = np.asarray(field)
    n = field.shape[-1]
    if n_out == n:
        return field.copy()
    assert n_out < n and n % n_out == 0
    fh = np.fft.rfft2(field)
    h = n_out // 2
    out = np.zeros(field.shape[:-2] + (n_out, h + 1), dtype=fh.dtype)
    out[..., :h, :h] = fh[..., :h, :h]
    out[..., -h + 1:, :h] = fh[..., -h + 1:, :h]  # drop the Nyquist row/column
    out *= (n_out / n) ** 2
    return np.fft.irfft2(out, s=(n_out, n_out))


def isotropic_spectrum(field, dx):
    """Isotropic power spectrum of (..., n, n) field. Returns (k [rad/m], E(k)) averaged over leading dims."""
    field = np.asarray(field)
    n = field.shape[-1]
    fh = np.fft.fft2(field) / n**2
    p = (np.abs(fh) ** 2).reshape(-1, n, n).mean(0)
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    kk = np.sqrt(k[None, :] ** 2 + k[:, None] ** 2)
    dk = 2 * np.pi / (n * dx)
    bins = np.arange(0.5, n // 2) * dk
    idx = np.digitize(kk.ravel(), bins)
    spec = np.bincount(idx, weights=p.ravel(), minlength=len(bins) + 1)[1:len(bins)] / dk
    kc = 0.5 * (bins[1:] + bins[:-1])
    return kc, spec
