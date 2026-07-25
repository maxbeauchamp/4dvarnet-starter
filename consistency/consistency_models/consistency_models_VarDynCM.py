from typing import NamedTuple, Optional, Tuple
import torch.nn.functional as F
from .utils import *

# VarDynCM — Variational Dynamical Consistency Model
#
# Fusion of VarCM (pairwise self-consistency, no prescribed forward SDE)
# and DynCM (spinup / physical time structure with Karras parametrisation).
#
# TIME AXIS  (same as DynCM v6):
#   Diffusion time t ∈ [0, 1]:  t > sb = spin-up (noise),  t ≤ sb = physical
#   Solver progress s = 1 - t:  s=0 (noise) → s=1 (clean)
#
# KEY DIFFERENCES from DynCM:
#   Spin-up  : x_start = τ·σ_max·ε + (1-τ)·IC   where τ(t)=(t-sb)/(1-sb)
#              Derived from the boundary-relative c_skip (self-consistent,
#              NOT an SDE assumption).  At t=1: pure noise.  At t=sb: clean IC.
#   Physical : x_start = x_k               (GT frame directly, NO interpolation)
#
# VARIATIONAL GRADIENT CONDITIONING:
#   conditioning_mode = "none" : no extra gradient (same as DynCM)
#   conditioning_mode = "grad" : ∇_x J(x, y) passed to model as separate signal
#     — GATED BY PHYSICAL TIME:
#       spin-up (t > sb)  : grad_cond = 0  (noise has no physical meaning)
#       physical (t ≤ sb) : grad_cond = ∇_x J(x_k, y) / std(∇J)
#
#   J(x, y, t) = J_o(x, y, t) + λ_reg · J_b(x)
#   J_o = Σ_k w_k(t) · ||x - y_k||²_{mask_k}   (temporally-weighted obs cost)
#   J_b = ||x - Phi(x)||²                        (prior / AE regularisation)
#
# LOSS:
#   L = L_comp + λ_IC · L_IC + λ_phys · L_phys + λ_obs · L_obs + λ_ae · L_ae
#
#   L_ae = MSE(Φ(x_frame), x_frame)  — trains prior_cost directly.
#   prior_cost parameters receive zero gradient from the other terms (autograd
#   is taken w.r.t. a detached x_leaf only), so this explicit term is required.
#
# Pairwise preconditioner (regime-dependent, NO SDE / NO interpolant assumption):
#   Spin-up  (t > sb): c_skip = (t'-sb)/(t-sb), c_out = (t-t')/(t-sb)
#     Telescopes: ∏(t'_k - sb)/(t_k - sb) = 0  → noise fully eliminated at t=sb
#   Physical (t ≤ sb): c_skip = t'/t,  c_out = 1 - t'/t
#   output = c_skip · x  +  c_out · F_θ(x, y, t, t'; grad_cond)
#
# v8 fix: Student takes the BIGGER step (current → next), teacher takes
# the SMALLER step (intermediate → next). Matches CM pairwise design.


# ── Types ─────────────────────────────────────────────────────────────────────

class VarDynCMOutput(NamedTuple):
    loss:          Tensor
    loss_comp:     Tensor
    loss_IC:       Tensor
    loss_phys:     Tensor
    loss_obs:      Tensor   # MSE at observed pixels only — breaks smoothness
    loss_ae:       Tensor   # AE reconstruction — trains prior_cost
    num_timesteps: int
    times:         Tensor


# ── Variational cost modules ──────────────────────────────────────────────────

class VarDynCMObsCost(nn.Module):
    """
    Physical-time observation cost for VarDynCM.

    x : (B, 1, H, W)  current physical state (single frame at time t)
    y : (B, C, H, W)  full observation window (NaN = unobserved)
    t : (B,)          diffusion time in physical regime (t ≤ spinup_boundary)

    J_o(x, y, t) = Σ_k w_k(t) · ||x − y_k||²_{mask_k}

    w_k(t) mirrors _compute_obs_weights: w points to the observation frame(s)
    nearest to the physical time t.  This makes J_o consistent with the
    temporal structure used by the UNet's w_obs input channel.
    """

    def __init__(self, sigma_obs: float = 1.0, spinup_boundary: float = 0.7):
        super().__init__()
        self.sigma_obs        = sigma_obs
        self.spinup_boundary  = spinup_boundary

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        B, C, H, W = y.shape
        sb   = self.spinup_boundary
        dev, dtype = y.device, x.dtype

        mask   = (~torch.isnan(y)).to(dtype=dtype)
        y_fill = torch.nan_to_num(y, nan=0.0)

        # Temporal weights — same schedule as _make_regime_input
        phys_step = sb / max(C - 1, 1)
        phys_steps = torch.linspace(0.0, sb, C, device=dev, dtype=dtype)
        t_c  = t.clamp(0.0, sb).unsqueeze(1)
        idx  = torch.searchsorted(
            phys_steps.unsqueeze(0).expand(B, -1).contiguous(), t_c
        ).squeeze(1).clamp(1, C - 1)
        b_idx = torch.arange(B, device=dev)
        a    = ((phys_steps[idx] - t.clamp(0., sb)) / phys_step).clamp(0., 1.)
        w    = torch.zeros(B, C, device=dev, dtype=dtype)
        w[b_idx, C - idx]     = a
        w[b_idx, C - 1 - idx] = 1.0 - a       # (B, C)

        cost = x.new_zeros(B)
        for k in range(C):
            wk   = w[:, k].view(B, 1, 1, 1)
            mk   = mask[:, [k], :, :]
            diff = mk * (x - y_fill[:, [k], :, :])
            cost = cost + wk * diff.pow(2).sum(dim=(-1, -2, -3))

        n_obs = mask.sum(dim=(-1, -2, -3)).clamp(min=1.0)
        return (cost / n_obs / (2.0 * self.sigma_obs ** 2)).mean()


class VarDynCMPriorCost(nn.Module):
    """
    Prior cost for single-frame VarDynCM: J_b(x) = ||x − Φ(x)||²

    Φ is a lightweight bilinear autoencoder (1-channel in, 1-channel out).
    """

    def __init__(self, dim_ae: int = 32):
        super().__init__()
        self.enc       = nn.Sequential(
            nn.Conv2d(1, dim_ae, 3, padding=1), nn.SiLU(),
            nn.Conv2d(dim_ae, dim_ae, 3, padding=1), nn.SiLU(),
        )
        self.dec_lin   = nn.Sequential(
            nn.Conv2d(dim_ae, dim_ae, 3, padding=1), nn.SiLU(),
            nn.Conv2d(dim_ae, 1, 3, padding=1),
        )
        self.dec_quad  = nn.Sequential(
            nn.Conv2d(dim_ae, dim_ae, 3, padding=1), nn.SiLU(),
            nn.Conv2d(dim_ae, 1, 3, padding=1),
        )

    def forward_ae(self, x: Tensor) -> Tensor:
        z = self.enc(x)
        return self.dec_lin(z) + self.dec_quad(z) ** 2

    def forward(self, x: Tensor) -> Tensor:
        return F.mse_loss(x, self.forward_ae(x)) * x.numel()


# ── Local observation conditioning helper ────────────────────────────────────

def _get_local_obs(y: Tensor, t: Tensor, spinup_boundary: float) -> Tensor:
    """
    Build (B, 2, H, W) = cat(y_local_filled, mask_local) for conditioning modes
    "obs" and "obs+grad".

    The "local" frame is the nearest physical anchor frame to diffusion time t,
    using the same frame↔time mapping as _make_regime_input:
        physical_steps[k] = sb * k / (C-1)  →  frame index C-1-k

    Returns zeros for spin-up elements (t > spinup_boundary).
    """
    B, C, H, W = y.shape
    device, dtype = y.device, t.dtype

    phys_steps = torch.linspace(0.0, spinup_boundary, C, device=device, dtype=dtype)
    t_c  = t.clamp(0.0, spinup_boundary)
    # argmin distance to physical anchor grid → physical-step index k
    k_idx = (phys_steps.unsqueeze(0) - t_c.unsqueeze(1)).abs().argmin(dim=1)  # (B,)
    frame_idx = (C - 1 - k_idx).clamp(0, C - 1)                               # frame in y

    y_fill = torch.nan_to_num(y, nan=0.0).to(dtype=dtype)
    mask   = (~torch.isnan(y)).to(dtype=dtype)

    b_idx   = torch.arange(B, device=device)
    y_local = y_fill[b_idx, frame_idx].unsqueeze(1)   # (B, 1, H, W)
    m_local = mask  [b_idx, frame_idx].unsqueeze(1)   # (B, 1, H, W)

    # Zero out spin-up elements (no physical meaning)
    is_spinup = (t > spinup_boundary).view(B, 1, 1, 1)
    y_local = y_local * (~is_spinup)
    m_local = m_local * (~is_spinup)

    return torch.cat([y_local, m_local], dim=1)        # (B, 2, H, W)


# ── Physical-time gradient of variational cost ────────────────────────────────

def _compute_physical_grad_J(
    x:               Tensor,             # (B, 1, H, W)  physical state
    y:               Tensor,             # (B, C, H, W)  observations
    t:               Tensor,             # (B,)          physical diffusion time
    obs_cost:        nn.Module,
    prior_cost:      Optional[nn.Module],
    lambda_reg:      Optional[nn.Parameter],
    spinup_boundary: float,
) -> Tensor:
    """
    Compute ∇_x J(x, y, t) for elements in the physical regime (t ≤ spinup_boundary).

    J = J_o(x, y, t) + λ_reg · J_b(x)

    Returns the normalised gradient (unit std), shape (B, 1, H, W).
    Only meaningful when called with t ≤ spinup_boundary.
    """
    # Two independent leaf tensors → two independent graphs → no retain_graph needed.
    # Normalise each term independently so neither J_o nor J_b dominates.
    xl_Jo = x.detach().float().requires_grad_(True)
    xl_Jb = x.detach().float().requires_grad_(True)
    y_f   = y.detach().float()
    t_f   = t.detach().float()

    try:
        cost_dtype = next(obs_cost.parameters()).dtype
    except StopIteration:
        cost_dtype = torch.float32

    # Snap t to the nearest physical anchor grid point so that the obs-cost
    # temporal weight is exactly 1 on one frame.  x is always the nearest GT
    # frame to t (from _get_start), so the gradient must compare against that
    # same frame's observation only — using interpolated weights mixes two
    # different frames and breaks temporal coherence.
    C_y = y.shape[1]
    phys_grid = torch.linspace(0.0, spinup_boundary, C_y, device=xl_Jo.device, dtype=torch.float32)
    t_snap = phys_grid[
        (phys_grid.unsqueeze(0) - t_f.clamp(0.0, float(spinup_boundary)).unsqueeze(1))
        .abs().argmin(dim=1)
    ]  # (B,) — each t rounded to nearest physical frame time

    with torch.enable_grad():
        Jo   = obs_cost(xl_Jo.to(cost_dtype), y_f.to(cost_dtype), t_snap.to(cost_dtype))
        g_Jo = torch.autograd.grad(Jo.float(), xl_Jo, create_graph=False, allow_unused=True)[0]
        if prior_cost is not None:
            Jb   = prior_cost(xl_Jb.to(cost_dtype))
            g_Jb = torch.autograd.grad(Jb.float(), xl_Jb, create_graph=False, allow_unused=True)[0]
        else:
            g_Jb = None

    if g_Jo is None:
        g_Jo = torch.zeros_like(x)
    if g_Jb is None:
        g_Jb = torch.zeros_like(x)

    g_Jo = g_Jo.detach()
    g_Jb = g_Jb.detach()
    return (
        g_Jo / g_Jo.std().clamp(min=1e-8)
        + g_Jb / g_Jb.std().clamp(min=1e-8)
    ).to(dtype=x.dtype)


# ── Forward wrapper with variational gradient conditioning ────────────────────

def _fwd(
    model:            nn.Module,
    x:                Tensor,
    y:                Tensor,
    t:                Tensor,
    t_prime:          Tensor,
    spinup_boundary:  float,
    # variational conditioning (optional)
    obs_cost:         Optional[nn.Module]    = None,
    prior_cost:       Optional[nn.Module]    = None,
    lambda_reg:       Optional[nn.Parameter] = None,
    conditioning_mode: str                   = "none",
    **kw,
) -> Tensor:
    """
    VarDynCM single-step forward pass with regime-dependent pairwise
    preconditioning (no SDE / no interpolant assumption):

      Spin-up  (t > sb): c_skip = (t'-sb)/(t-sb)      (boundary-relative)
      Physical (t ≤ sb): c_skip = t'/t                 (self-consistent)

    The spin-up formula telescopes: ∏(t'_k-sb)/(t_k-sb) → 0 at t=sb,
    guaranteeing full noise elimination at the boundary regardless of
    the number of sampling steps.

      conditioning_mode:
        "none"     — no extra conditioning                            (grad_j_channels=0)
        "grad"     — ∇J(x,y,t)/std  in physical regime               (grad_j_channels=1)
        "obs"      — cat(y_local, mask_local)  in physical regime     (grad_j_channels=2)
        "obs+grad" — cat(y_local, mask_local, ∇J)  in physical regime (grad_j_channels=3)
    """
    grad_cond: Optional[Tensor] = None

    if conditioning_mode in ("grad", "obs", "obs+grad"):
        B = x.shape[0]
        n_ch = {"grad": 1, "obs": 2, "obs+grad": 3}[conditioning_mode]
        grad_cond = torch.zeros(B, n_ch, *x.shape[2:], device=x.device, dtype=x.dtype)

        is_phys = (t <= spinup_boundary)
        if is_phys.any():
            b_p = torch.where(is_phys)[0]

            if conditioning_mode in ("obs", "obs+grad"):
                loc = _get_local_obs(y[b_p], t[b_p], spinup_boundary)  # (B_p, 2, H, W)
                grad_cond[b_p, :2] = loc

            if conditioning_mode in ("grad", "obs+grad") and obs_cost is not None:
                g_J = _compute_physical_grad_J(
                    x[b_p], y[b_p], t[b_p],
                    obs_cost, prior_cost, lambda_reg, spinup_boundary,
                )
                ch = 2 if conditioning_mode == "obs+grad" else 0
                grad_cond[b_p, ch:ch + 1] = g_J

    sb = spinup_boundary
    is_spinup = (t > sb)
    is_sp = pad_dims_like(is_spinup.to(x.dtype), x)

    # Spin-up: boundary-relative (no SDE assumption, telescopes to 0 at t=sb)
    c_skip_sp = pad_dims_like(
        (t_prime - sb).clamp(min=0) / (t - sb).clamp(min=1e-8), x
    )
    c_out_sp  = 1.0 - c_skip_sp

    # Physical: self-consistent (t'/t)
    c_skip_ph = pad_dims_like(t_prime / t.clamp(min=1e-8), x)
    c_out_ph  = 1.0 - c_skip_ph

    c_skip = is_sp * c_skip_sp + (1 - is_sp) * c_skip_ph
    c_out  = is_sp * c_out_sp  + (1 - is_sp) * c_out_ph

    model_out = model(x, y, t, t_prime, grad_cond=grad_cond, **kw)
    return c_skip * x + c_out * model_out


# ── Training ──────────────────────────────────────────────────────────────────

class VarDynCMTraining:
    """
    Pairwise self-consistency training for VarDynCM.

    Parameters
    ----------
    conditioning_mode : "none" | "grad"
        "none" — no gradient conditioning (same as DynCM)
        "grad" — ∇J passed to model in physical regime, zeros in spin-up
    obs_cost / prior_cost / lambda_reg_init
        Required when conditioning_mode = "grad".
        lambda_reg is a learnable scalar weight; pass the nn.Parameter from the
        Lightning module so that it is optimised jointly.
    """

    def __init__(
        self,
        spinup_boundary:    float = 0.7,
        sigma_max:          float = 3.0,
        initial_timesteps:  int   = 5,
        final_timesteps:    int   = 50,
        lambda_IC:          float = 1.0,
        lambda_phys:        float = 1.0,
        lambda_obs:         float = 0.0,
        lambda_ae:          float = 1.0,
        conditioning_mode:  str   = "none",
    ):
        self.spinup_boundary   = spinup_boundary
        self.sigma_max         = sigma_max
        self.initial_timesteps = initial_timesteps
        self.final_timesteps   = final_timesteps
        self.lambda_IC         = lambda_IC
        self.lambda_phys       = lambda_phys
        self.lambda_obs        = lambda_obs
        self.lambda_ae         = lambda_ae
        self.conditioning_mode = conditioning_mode

    def __call__(
        self,
        student:    nn.Module,
        teacher:    nn.Module,
        x:          Tensor,               # (B, C, H, W)  GT physical frames
        y:          Tensor,               # (B, C, H, W)  observations (NaN = unobs.)
        current_step:  int,
        total_steps:   int,
        obs_cost:      Optional[nn.Module]    = None,
        prior_cost:    Optional[nn.Module]    = None,
        lambda_reg:    Optional[nn.Parameter] = None,
        **kwargs,
    ) -> VarDynCMOutput:

        B, C, H, W = x.shape
        sb  = self.spinup_boundary
        dev = x.device

        N = max(timesteps_schedule(
            current_step, total_steps,
            self.initial_timesteps, self.final_timesteps), 3)
        times = torch.linspace(1.0, 1e-8, N, device=dev)

        # ── Spinup / physical partition ───────────────────────────────────────
        k_IC         = int((times > sb).sum().item()) - 1
        valid_spinup = list(range(0, max(0, k_IC - 1)))
        valid_phys   = list(range(k_IC + 1, N - 2))
        valid_all    = valid_spinup + valid_phys or list(range(N - 2))

        vt       = torch.tensor(valid_all, device=dev)
        rand_pos = torch.randint(0, len(valid_all), (B,), device=dev)
        tidx     = vt[rand_pos]

        t_curr = times[tidx]      # t_i  (largest)
        t_int  = times[tidx + 1]  # t_j  (middle)
        t_next = times[tidx + 2]  # t_k  (smallest)

        IC    = x[:, [0], :, :]
        noise = torch.randn(B, 1, H, W, device=dev, dtype=x.dtype)
        x_start = _get_start(x, noise, t_curr, sb, self.sigma_max)

        def fwd(m, inp, tc, tn):
            return _fwd(
                m, inp, y, tc, tn, sb,
                obs_cost=obs_cost, prior_cost=prior_cost, lambda_reg=lambda_reg,
                conditioning_mode=self.conditioning_mode,
                **kwargs,
            )

        # ── L_comp: pairwise consistency (v8 fix: student=big step, teacher=small step)
        student_pred = fwd(student, x_start, t_curr, t_next)

        with torch.no_grad():
            x_int_gt     = _get_start(x, noise, t_int,  sb, self.sigma_max)
            x_next_gt    = _get_start(x, noise, t_next, sb, self.sigma_max)
            teacher_pred = fwd(teacher, x_int_gt, t_int, t_next)
            is_phys      = (t_curr <= sb).view(B, 1, 1, 1)
            target_comp  = torch.where(is_phys, x_next_gt.to(student_pred.dtype), teacher_pred)

        loss_comp = pseudo_huber_loss(student_pred, target_comp.detach()).mean()

        # ── L_IC: spin-up anchoring  g(noise, t_spin, t_IC=sb) = IC ─────────
        if k_IC >= 1:
            t_sp_pool = times[:k_IC + 1]
            sp_idx    = torch.randint(0, k_IC + 1, (B,), device=dev)
            t_sp      = t_sp_pool[sp_idx]
            t_IC_vec  = torch.full((B,), sb, device=dev, dtype=x.dtype)
            noise_sp  = torch.randn(B, 1, H, W, device=dev, dtype=x.dtype)
            x_sp      = _get_start(x, noise_sp, t_sp, sb, self.sigma_max)
            pred_IC   = _fwd(
                student, x_sp, y, t_sp, t_IC_vec, sb,
                conditioning_mode="none",
                **kwargs,
            )
            loss_IC   = pseudo_huber_loss(pred_IC, IC).mean()
        else:
            loss_IC = x.new_zeros(())

        # ── L_phys: physical anchoring  g(x_k, t_k, t_{k+1}) = x_{k+1} ──────
        # ── L_obs:  same but MSE restricted to observed pixels of x_{k+1} ───
        loss_phys = x.new_zeros(())
        loss_obs  = x.new_zeros(())
        for k in range(C - 1):
            t_k   = sb * (C - 1 - k) / max(C - 1, 1)
            t_k1  = sb * (C - 2 - k) / max(C - 1, 1)
            x_k   = x[:, [k],     :, :]
            x_k1  = x[:, [k + 1], :, :]
            t_k_v  = torch.full((B,), t_k,  device=dev, dtype=x.dtype)
            t_k1_v = torch.full((B,), t_k1, device=dev, dtype=x.dtype)
            pred   = fwd(student, x_k, t_k_v, t_k1_v)
            loss_phys = loss_phys + pseudo_huber_loss(pred, x_k1).mean()

            mask_k1 = (~torch.isnan(y[:, [k + 1], :, :])).to(dtype=x.dtype)
            n_obs   = mask_k1.sum().clamp(min=1.0)
            loss_obs = loss_obs + (pseudo_huber_loss(pred, x_k1) * mask_k1).sum() / n_obs

        loss_phys = loss_phys / max(C - 1, 1)
        loss_obs  = loss_obs  / max(C - 1, 1)

        # ── L_ae: AE reconstruction — trains prior_cost directly ─────────────
        # prior_cost.parameters() receive zero gradient from all other terms
        # (autograd is taken w.r.t. a detached x_leaf only).  This explicit
        # reconstruction term is required for prior_cost to learn a meaningful
        # state manifold; without it the gradient ∇J_b stays random, causing
        # grad_j_projection to amplify noise → variance explosion.
        if prior_cost is not None and self.lambda_ae > 0.0:
            x_flat = x.reshape(B * C, 1, H, W)
            ae_dtype = next(prior_cost.parameters()).dtype
            l_ae = F.mse_loss(
                prior_cost.forward_ae(x_flat.to(dtype=ae_dtype, non_blocking=True)),
                x_flat.to(dtype=ae_dtype, non_blocking=True),
            ).float()
        else:
            l_ae = x.new_zeros(())

        loss = (loss_comp
                + self.lambda_IC   * loss_IC
                + self.lambda_phys * loss_phys
                + self.lambda_obs  * loss_obs
                + self.lambda_ae   * l_ae)
        return VarDynCMOutput(loss, loss_comp, loss_IC, loss_phys, loss_obs, l_ae, N, times)


# ── Sampling ─────────────────────────────────────────────────────────────────

class VarDynCMSamplingAndEditing:
    """
    Consistency sampling for VarDynCM.

    Replaces ConsistencySamplingAndEditingDynamicalSystems for VarDynCM:
    uses _fwd at each step so that variational gradient / obs conditioning
    (conditioning_mode = "grad" | "obs" | "obs+grad") is applied during
    inference — not just during training.

    Without this, the model trains WITH grad_cond but samples WITHOUT it,
    causing a train/inference mismatch that produces smooth / unconditioned
    outputs regardless of the training conditioning mode.

    Same output format as ConsistencySamplingAndEditingDynamicalSystems:
      returns (final_x, all_xs_stacked, phys_frame_indices)
    """

    def __init__(
        self,
        sigma_max:         float = 3.0,
        spinup_boundary:   float = 0.7,
        conditioning_mode: str   = "none",
    ):
        self.sigma_max         = sigma_max
        self.spinup_boundary   = spinup_boundary
        self.conditioning_mode = conditioning_mode

    def __call__(
        self,
        model:         nn.Module,
        noise:         Tensor,                       # (B, 1, H, W)  unit normal
        y:             Tensor,                       # (B, C, H, W)  observations
        nsteps:        int,
        obs_cost:      Optional[nn.Module]    = None,
        prior_cost:    Optional[nn.Module]    = None,
        lambda_reg:    Optional[nn.Parameter] = None,
        clip_denoised: bool                   = False,
        **kwargs,
    ):
        sb    = self.spinup_boundary
        dev   = noise.device
        dtype = noise.dtype
        times = torch.linspace(1.0, 1e-8, nsteps, device=dev, dtype=dtype)

        x      = noise * self.sigma_max
        all_xs = [x]

        for i in range(nsteps - 1):
            tc = torch.full((noise.shape[0],), times[i].item(),     device=dev, dtype=dtype)
            tn = torch.full((noise.shape[0],), times[i + 1].item(), device=dev, dtype=dtype)
            x = _fwd(
                model, x, y, tc, tn, sb,
                obs_cost=obs_cost, prior_cost=prior_cost, lambda_reg=lambda_reg,
                conditioning_mode=self.conditioning_mode,
                **kwargs,
            )
            if clip_denoised:
                x = x.clamp(-1.0, 1.0)
            all_xs.append(x)

        # Physical frame indices — IC first (same convention as DynCM sampler)
        phys_steps = torch.linspace(0.0, sb, y.shape[1]).tolist()
        times_cpu  = times.cpu()
        phys_frame_indices = [
            max(1, torch.argmin(torch.abs(times_cpu - t_k)).item())
            for t_k in phys_steps
        ]
        phys_frame_indices = phys_frame_indices[::-1]

        return x, torch.stack(all_xs, dim=0), phys_frame_indices


# ── Starting-state helper ─────────────────────────────────────────────────────

def _get_start(x: Tensor, noise: Tensor, t_curr: Tensor, sb: float, sigma_max: float) -> Tensor:
    """
    Spin-up (t > sb)  : τ · σ_max · noise  +  (1-τ) · IC     where τ = (t-sb)/(1-sb)
    Physical (t ≤ sb) : nearest GT frame at physical anchor t_k = sb·(C-1-k)/(C-1)

    τ(t) is the cumulative c_skip product from t=1 to t under the boundary-
    relative preconditioning c_skip=(t'-sb)/(t-sb).  The interpolation between
    noise and IC matches what the sampling trajectory produces when the model
    is correct, ensuring training and inference see the same distribution.
    """
    B, C, H, W = x.shape
    device, dtype = x.device, t_curr.dtype

    frame_times = torch.tensor(
        [sb * (C - 1 - k) / max(C - 1, 1) for k in range(C)],
        device=device, dtype=dtype,
    )  # [sb, ..., 0]

    IC  = x[:, [0], :, :]
    tau = ((t_curr - sb) / (1.0 - sb)).clamp(min=0).view(B, 1, 1, 1)
    x_start = tau * noise * sigma_max + (1.0 - tau) * IC

    is_phys = t_curr <= sb
    if is_phys.any():
        t_phys  = t_curr[is_phys].unsqueeze(1)
        nearest = (frame_times.unsqueeze(0) - t_phys).abs().argmin(dim=1)
        b_phys  = torch.where(is_phys)[0]
        x_start = x_start.clone()
        x_start[b_phys] = x[b_phys, nearest, :, :].unsqueeze(1)

    return x_start
