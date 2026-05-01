"""
Pairwise Self-Consistency in Variational Dynamics (without explicit forward SDE)
=================================================================================

Theory -- Interpolant Framework
-------------------------------
Instead of a prescribed forward diffusion, we use a deterministic interpolant

    x_t = alpha(t) * x_1 + beta(t) * x_0,    t in [0, 1]

with boundary conditions:

    alpha(0) = 0,  alpha(1) = 1,
    beta(0)  = 1,  beta(1)  = 0

Common choices:
  Linear (default):       alpha(t) = t,              beta(t) = 1 - t
  Trigonometric (VP):     alpha(t) = sin(pi/2 * t),  beta(t) = cos(pi/2 * t)

The induced transport from t to t' is:

    x_{t'} = (alpha(t')/alpha(t)) * x_t
            + (beta(t') - alpha(t')/alpha(t) * beta(t)) * x_0

which defines the "ideal" g_phi when x_0 is known.

Network Preconditioning (c_in / c_out / c_skip)
-------------------------------------------------
Inspired by Karras et al. (EDM) but adapted to the interpolant setting:

  c_skip(t, t') = alpha(t') / alpha(t)
      The deterministic forward-transport ratio.
      At t' = 0:  c_skip = 0  (no direct x carry-over).
      At t' = t:  c_skip = 1  (identity).

  c_out(t, t') = beta(t') - (alpha(t') / alpha(t)) * beta(t)
      Scale of the network's learned residual (= coefficient of x_0
      in the ideal transport).
      At t' = 0:  c_out = beta(0) = 1.
      At t' = t:  c_out = 0  (identity, network predicts nothing).

  c_in(t) = 1 / sqrt(alpha(t)^2 + beta(t)^2)
      Normalises the input variance (assuming unit-std x_0 and x_1).
      For the trigonometric schedule this equals 1 identically.
      For the linear schedule it rescales by 1/sqrt(t^2 + (1-t)^2).

The operator is parametrised as:

    g_phi(x, grad_J, t, t') = c_skip(t,t') * x
                             + c_out(t,t')  * F_theta(c_in(t)*x, grad_J, t, t')

where F_theta is the UNet receiving c_in(t)*x and grad_J as *separate* arguments
(both of shape (B, C, H, W)), plus two solver-time embeddings.  Inside the UNet,
c_in(t)*x goes through input_projection and grad_J through grad_j_projection;
their outputs are added before the first encoder block.

Training Loss -- Composition-Based Supervision
----------------------------------------------
"Times" t in [0, 1] are abstract solver indices (not diffusion noise levels).

The key insight: at inference the model runs a chain

    x_T  ->[g(x,T,t_1)]->  x_1  ->[g(x,t_1,t_2)]->  x_2  -> ... -> x_0

We always have access to x_T (pure noise) and x_0 (ground truth), so every
composed state  h_t = g(x_T, T, t)  can be analytically cross-checked against
the interpolant  x_{interp,t} = alpha(t)*x_T + beta(t)*x_0.

Three terms, teacher/student strategy:

  Term 1 - Pairwise consistency (anti-identity, via shared x_T):
    Sample t < t' < t'' from the schedule.
    h_t   = g_teacher(x_T, T, t)   [no grad, teacher EMA]
    h_t'  = g_teacher(x_T, T, t')  [no grad, teacher EMA]
    out_t  = g_student(h_t,  t,  t'')  [GRAD]
    out_t' = g_teacher(h_t', t', t'') [stop-grad, teacher EMA]
    L1 = ||out_t - sg(out_t')||^2

    Using the teacher (EMA of student) for the stop-grad branch provides
    a STABLE target: the teacher evolves slowly (EMA decay ~0.99+), so the
    target barely changes between consecutive training steps.  Without this,
    the stop-grad target is computed from the current student weights and
    oscillates as strongly as the student -- an unstable "moving target".
    Both branches start from the SAME x_T and target the SAME t'', so they
    cannot collapse to identity.  The unique minimum is the true x_{t''}.

  Term 2 - Composed-path interpolant supervision (progressive denoising):
    Supervise the full composition h_t = g(x_T, T, t) against x_{interp,t}.
    L2 = ||g(x_T, T, t)  - x_{interp,t} ||^2
       + ||g(x_T, T, t') - x_{interp,t'}||^2

    This forces g to predict states with the *right variance level* for each t:
    g(x_T, T, large_t) must stay close to x_T,
    g(x_T, T, small_t) must approach x_0.
    Without this term, Term 1 alone can be satisfied by any fixed denoising
    level (e.g. always outputting x_0 for all t).

  Term 3 - Prior regularisation:
    L3 = ||x_0 - Phi(x_0)||^2

Time discretisation:
  Times t in [0, 1] follow the Karras schedule (karras_schedule(..., as_time=True)),
  recomputed at every training step from timesteps_schedule.
"""

from .utils import *
from torch.nn import functional as F
import math


# ---------------------------------------------------------------------------
# Interpolant schedules  alpha(t) / beta(t)
# ---------------------------------------------------------------------------

class LinearSchedule:
    """
    Linear interpolant:  alpha(t) = t,  beta(t) = 1 - t.

    c_in(t)  = 1 / sqrt(t^2 + (1-t)^2)    (input normalisation)
    c_skip(t, t') = t' / t                  (forward-transport ratio)
    c_out(t, t')  = (1-t') - (t'/t)*(1-t)  (residual scale = t'/t - 1 + 1 - t')
    """
    @staticmethod
    def alpha(t: Tensor) -> Tensor:
        return t

    @staticmethod
    def beta(t: Tensor) -> Tensor:
        return 1.0 - t

    @staticmethod
    def c_in(t: Tensor, sigma_noise: float = 1.0) -> Tensor:
        """Normalise input: 1 / sqrt(sigma_data^2*(1-t)^2 + sigma_noise^2*t^2).

        With sigma_data=1 (normalized data). For sigma_noise=1 this reduces to
        1/sqrt((1-t)^2 + t^2), identical to the old formula (backward compat).
        For sigma_noise>>1 this properly whitens x_t = x_0 + t*sigma_noise*z.
        """
        sigma_data = 1.0
        return 1.0 / ((sigma_data * (1.0 - t)) ** 2 + (sigma_noise * t) ** 2).sqrt().clamp(min=1e-8)

    @staticmethod
    def c_skip(t: Tensor, t_prime: Tensor) -> Tensor:
        """alpha(t') / alpha(t) -- carries x forward deterministically."""
        return t_prime / t.clamp(min=1e-8)

    @staticmethod
    def c_out(t: Tensor, t_prime: Tensor) -> Tensor:
        """beta(t') - c_skip * beta(t) -- scale of the learned residual."""
        c_s = LinearSchedule.c_skip(t, t_prime)
        return (1.0 - t_prime) - c_s * (1.0 - t)


class TrigSchedule:
    """
    Trigonometric (variance-preserving) interpolant:
        alpha(t) = sin(pi/2 * t),  beta(t) = cos(pi/2 * t).

    alpha^2 + beta^2 = 1  =>  c_in(t) = 1 identically.
    """
    @staticmethod
    def alpha(t: Tensor) -> Tensor:
        return torch.sin(0.5 * torch.pi * t)

    @staticmethod
    def beta(t: Tensor) -> Tensor:
        return torch.cos(0.5 * torch.pi * t)

    @staticmethod
    def c_in(t: Tensor, sigma_noise: float = 1.0) -> Tensor:
        """1 / sqrt(sigma_data^2*cos^2 + sigma_noise^2*sin^2). For sigma_noise=1: =1."""
        sigma_data = 1.0
        a = TrigSchedule.alpha(t)
        b = TrigSchedule.beta(t)
        return 1.0 / ((sigma_data * b) ** 2 + (sigma_noise * a) ** 2).sqrt().clamp(min=1e-8)

    @staticmethod
    def c_skip(t: Tensor, t_prime: Tensor) -> Tensor:
        alpha_t  = TrigSchedule.alpha(t)
        alpha_tp = TrigSchedule.alpha(t_prime)
        return alpha_tp / alpha_t.clamp(min=1e-8)

    @staticmethod
    def c_out(t: Tensor, t_prime: Tensor) -> Tensor:
        c_s    = TrigSchedule.c_skip(t, t_prime)
        beta_t  = TrigSchedule.beta(t)
        beta_tp = TrigSchedule.beta(t_prime)
        return beta_tp - c_s * beta_t


# Default schedule used throughout the module
DEFAULT_SCHEDULE = LinearSchedule


# ---------------------------------------------------------------------------
# Forward wrapper  (EDM-style c_in / c_skip / c_out preconditioning)
# ---------------------------------------------------------------------------

def model_variational_forward_wrapper(
    model: nn.Module,
    obs_cost: nn.Module,
    prior_cost: nn.Module,
    lambda_reg: nn.Parameter,
    x: Tensor,
    y: Tensor,
    t: Tensor,
    t_prime: Tensor,
    schedule=None,
    sigma_noise: float = 1.0,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    """
    Apply the preconditioned transport operator:

        g_phi(x, grad_J, t, t') = c_skip(t,t') * x
                                 + c_out(t,t')  * F_theta(c_in(t)*x, grad_J, t, t')

    where grad_J = nabla_x J(x)  is the spatial gradient of the variational cost

        J(x) = J_o(x, y) + lambda_reg * J_b(x)

    and the schedule functions (c_in, c_skip, c_out) are derived from the
    interpolant alpha(t), beta(t):

        c_skip(t, t') = alpha(t') / alpha(t)
        c_out(t, t')  = beta(t')  - c_skip * beta(t)
        c_in(t)       = 1 / ||[alpha(t), beta(t)]||_2

    grad_J (shape (B, C, H, W), same as x) is passed as a *separate* argument
    to F_theta -- no channel concatenation.  Inside the UNet, it goes through
    its own learned projection (grad_j_projection) which is added to the
    output of input_projection before the first encoder block.  This keeps
    the two signals cleanly decoupled while avoiding any increase in the
    number of input channels.

    Boundary conditions guaranteed by the preconditioning:
      * g_phi(x, grad_J, t, t)  = x          (identity when t' = t)
      * g_phi(x, grad_J, t, 0)  = F_theta(.) (pure prediction at t' = 0)

    Parameters
    ----------
    model      : nn.Module       Student or teacher UNet.
    obs_cost   : nn.Module       Observation cost J_o(x, y).
    prior_cost : nn.Module       Prior cost J_b(x) (contains the AE Phi).
    lambda_reg : nn.Parameter    Scalar weight: J = J_o + lambda_reg * J_b.
    x          : (B, C, H, W)   Current state.
    y          : (B, C, H, W)   Observations.
    t          : (B,)            Source solver time in [0, 1].
    t_prime    : (B,)            Target solver time in [0, 1].
    schedule   : class           LinearSchedule (default) or TrigSchedule.

    Returns
    -------
    x_out  : Tensor  (B, C, H, W)  transported state g_phi(x, grad_J, t, t')
    grad_J : Tensor  (B, C, H, W)  gradient of J at the *input* x (detached)
    """
    if schedule is None:
        schedule = DEFAULT_SCHEDULE

    # conditioning_mode: "grad" (default) -> pass grad_J to model
    #                    "obs"            -> pass cat([y_filled, mask], dim=1) to model
    conditioning_mode = kwargs.pop("conditioning_mode", "grad")

    B, C, H, W = x.shape

    if conditioning_mode == "obs":
        # Build (y_filled, mask) -> (B, 2C, H, W), no autograd needed
        mask   = (~torch.isnan(y)).to(dtype=x.dtype)
        y_fill = torch.nan_to_num(y, nan=0.0).to(dtype=x.dtype)
        cond   = torch.cat([y_fill, mask], dim=1)   # (B, 2C, H, W)
        grad_J = cond  # returned for API compatibility (not a true gradient)
    else:
        # ---- Compute grad_J = nabla_x J(x)  -----------------------------------
        # x_leaf is kept in float32 so autograd.grad always returns float32.
        # The costs may live in any dtype (fp16, bf16 ...), so we cast the inputs
        # to the cost's dtype before the forward pass.  .to(dtype) is differentiable,
        # so the gradient still flows back to x_leaf correctly.
        x_leaf = x.detach().float().requires_grad_(True)
        y_c    = y.detach().float()
        lr_f   = lambda_reg.detach().float().clamp(min=0.0)  # keep lambda >= 0
        try:
            cost_dtype = next(prior_cost.parameters()).dtype
        except StopIteration:
            cost_dtype = torch.float32
        x_cost = x_leaf.to(dtype=cost_dtype)
        y_cost = y_c.to(dtype=cost_dtype)
        with torch.enable_grad():
            Jo  = obs_cost(x_cost, y_cost)
            Jb  = prior_cost(x_cost)
            J   = (Jo + lr_f.to(cost_dtype) * Jb).float()
            grad_J_c = torch.autograd.grad(
                J, x_leaf, create_graph=False, allow_unused=True
            )[0]
        if grad_J_c is None:
            grad_J_c = torch.zeros_like(x_leaf)
        # Normalise grad_J to unit std to keep the network input well-conditioned
        # regardless of the absolute scale of J (which can be O(sigma_noise^2)).
        g = grad_J_c.detach()
        g_std = g.std().clamp(min=1e-8)
        grad_J = cond = (g / g_std).to(dtype=x.dtype)

    # Work with detached x from here on (no gradient through transport steps)
    x = x.detach()

    # ---- Preconditioning scalars (broadcast to (B, 1, 1, 1)) --------------
    def _expand(s: Tensor) -> Tensor:
        return s.view(B, 1, 1, 1)

    c_in   = _expand(schedule.c_in(t, sigma_noise))  # (B, 1, 1, 1)
    c_skip = _expand(schedule.c_skip(t, t_prime))  # (B, 1, 1, 1)
    c_out  = _expand(schedule.c_out(t, t_prime))   # (B, 1, 1, 1)

    # ---- Build network inputs -----------------------------------------------
    # x and grad_J are passed as *separate* arguments so the UNet can project
    # them independently (each through its own Conv2d) before adding.
    net_x     = c_in * x        # (B, C, H, W)  -- whitened state
    # grad_J is NOT scaled by c_in: it already carries units of 1/state
    # and its own learned projection handles the scaling.

    # ---- Network forward ---------------------------------------------------
    # F_theta output: (B, C, H, W) -- direct prediction of x_0
    F_out = model(net_x, cond, t, t_prime, **kwargs)  # (B, C, H, W)

    # ---- Preconditioned output: c_skip * x + c_out * F_theta --------------
    x_out = c_skip * x + c_out * F_out             # (B, C, H, W)

    return x_out, grad_J


# ---------------------------------------------------------------------------
# Stabilised VarCM training (three-term loss, all targets at t''=0)
# ---------------------------------------------------------------------------

def _g_fwd(
    model, obs_cost, prior_cost, lambda_reg,
    x, y, t, t_prime, sigma_noise, conditioning_mode,
):
    """Single g_φ step — returns transported state (B,C,H,W)."""
    x_out, _ = model_variational_forward_wrapper(
        model, obs_cost, prior_cost, lambda_reg,
        x, y, t, t_prime,
        sigma_noise=sigma_noise,
        conditioning_mode=conditioning_mode,
    )
    return x_out


@dataclass
class VarCMOutput:
    """Output of VarCMTraining.forward()."""
    loss:          Tensor
    l_pair:        Tensor   # Term 1: pairwise consistency (at t''=0)
    l_long:        Tensor   # Term 2: long-chain anchor  x_T -> t -> 0
    l_short:       Tensor   # Term 3: short-chain anchor x_t -> t' -> 0
    num_timesteps: int
    times:         Tensor   # full time grid (for logging)
    t_sampled:     Tensor   # t  (B,)
    t_p_sampled:   Tensor   # t' (B,)


class VarCMTraining(nn.Module):
    """
    Deterministic pairwise VarCM training — stabilised three-term loss.

    All three terms produce x_0-scale predictions (t''=0), preventing the
    scale contradiction that causes explosion when t''≠0.

        L = λ_pair  * || g_s(g_t(x_T,T,t), t, 0) - sg[g_t(g_t(x_T,T,t'),t',0)] ||²
          + λ_long  * || g_s(g_t(x_T,T,t), t, 0) - x_0 ||²
          + λ_short * (see below)

    L_pair and L_long share the same student forward pass.

    Two modes for L_short, controlled by ``pure_short``:

    pure_short=False (default — interpolant mode):
        x_t = alpha(t)*x_T + beta(t)*x_0   (requires knowing x_0)
        L_short = || g_s(sg[g_s(x_t, t, t')], t', 0) - x_0 ||²

    pure_short=True (pure self-consistency, *no interpolant hypothesis*):
        Starting point: h_t = sg[g_t(x_T, T, t)]  (teacher rollout, no x_0 needed)
        L_short = || g_s(sg[g_s(h_t, t, t')], t', 0) - sg[g_t(h_t', t', 0)] ||²
        where h_t' = sg[g_t(x_T, T, t')]  (already computed for L_pair).
        This is a pure two-hop semigroup constraint: the student two-step
        x_T->t->t'->0 must match the teacher direct step x_T->t'->0,
        with no reference to x_0 or the interpolant.
    """

    def __init__(
        self,
        initial_timesteps:  int   = 5,
        final_timesteps:    int   = 17,
        sigma_noise:        float = 80.0,
        lambda_pair:        float = 1.0,
        lambda_long:        float = 1.0,
        lambda_short:       float = 1.0,
        schedule_power:     float = 1.0,
        conditioning_mode:  str   = "obs",
        pure_short:         bool  = True,
    ) -> None:
        super().__init__()
        self.initial_timesteps = initial_timesteps
        self.final_timesteps   = final_timesteps
        self.sigma_noise       = sigma_noise
        self.lambda_pair       = lambda_pair
        self.lambda_long       = lambda_long
        self.lambda_short      = lambda_short
        self.schedule_power    = schedule_power
        self.conditioning_mode = conditioning_mode
        self.pure_short        = pure_short

    def _fwd(self, model, obs_cost, prior_cost, lambda_reg, x, y, t, t_prime):
        return _g_fwd(model, obs_cost, prior_cost, lambda_reg, x, y, t, t_prime,
                      self.sigma_noise, self.conditioning_mode)

    def forward(
        self,
        student:     nn.Module,
        teacher:     nn.Module,
        obs_cost:    nn.Module,
        prior_cost:  nn.Module,
        lambda_reg:  nn.Parameter,
        x0:          Tensor,
        y:           Tensor,
        global_step: int,
        total_steps: int,
    ) -> "VarCMOutput":
        device = x0.device
        B = x0.shape[0]

        # ── Time grid ─────────────────────────────────────────────────────────
        # karras_schedule returns ascending values (sigma_min → sigma_max).
        # We flip to get a descending grid [1.0, ..., ~0] so that:
        #   times[0]  = T = 1.0   (starting noise level)
        #   times[-1] ≈ 0         (target x_0 scale)
        # and sampling j_idx < i_idx correctly gives t_s > t_p.
        N = timesteps_schedule(global_step, total_steps,
                               self.initial_timesteps, self.final_timesteps)
        times = karras_schedule(N, sigma_min=0.002 / self.sigma_noise,
                                sigma_max=1.0, rho=7.0, device=device)
        times = times.flip(0).pow(self.schedule_power).clamp(0.0, 1.0)
        # Ensure exactly t=0 at the end (in case sigma_min rounds above zero)
        if times[-1].item() > 1e-3:
            times = torch.cat([times, torch.zeros(1, device=device)])

        T_val  = times[0].item()   # = 1.0
        eps    = 1e-4
        n      = len(times)

        # ── Sample t > t' from interior of the grid ───────────────────────────
        interior = max(n - 2, 2)
        pair_idx = torch.randperm(interior, device=device)[:2].sort().values
        j_idx, i_idx = int(pair_idx[0]), int(pair_idx[1])
        # times is descending: larger index → smaller value → t_s > t_p
        t_s    = times[j_idx + 1].expand(B)
        t_p    = times[i_idx + 1].expand(B)
        T_vec  = torch.full((B,), T_val, device=device, dtype=x0.dtype)
        t0_vec = torch.full((B,), eps,   device=device, dtype=x0.dtype)

        schedule = DEFAULT_SCHEDULE

        # ── Noisy x_T (always needed) ─────────────────────────────────────────
        x_T = torch.randn_like(x0) * self.sigma_noise

        # Interpolant x_t: only needed when pure_short=False
        if not self.pure_short:
            alpha_t    = schedule.alpha(t_s).view(B, 1, 1, 1)
            beta_t     = schedule.beta(t_s).view(B, 1, 1, 1)
            x_t_interp = alpha_t * x_T + beta_t * x0

        # ── Teacher intermediates (all stop-grad) ─────────────────────────────
        with torch.no_grad():
            h_t  = self._fwd(teacher, obs_cost, prior_cost, lambda_reg,
                             x_T, y, T_vec, t_s)
            h_tp = self._fwd(teacher, obs_cost, prior_cost, lambda_reg,
                             x_T, y, T_vec, t_p)
            target_pair = self._fwd(teacher, obs_cost, prior_cost, lambda_reg,
                                    h_tp, y, t_p, t0_vec)

        # ── Terms 1 & 2: shared student forward g_s(h_t, t, 0) ───────────────
        pred_long = self._fwd(student, obs_cost, prior_cost, lambda_reg,
                              h_t, y, t_s, t0_vec)
        l_long = F.mse_loss(pred_long, x0)
        l_pair = F.mse_loss(pred_long, target_pair)

        # ── Term 3: L_short ───────────────────────────────────────────────────
        if self.pure_short:
            # Pure self-consistency: no interpolant, no x_0.
            # Short path: x_T --(teacher,sg)--> h_t --(student,sg)--> x_mid
            #                                        --(student)-----> t'->0
            # Target: sg[g_t(h_tp, t', 0)] = target_pair  (already computed)
            with torch.no_grad():
                x_mid = self._fwd(student, obs_cost, prior_cost, lambda_reg,
                                  h_t, y, t_s, t_p)
            pred_short = self._fwd(student, obs_cost, prior_cost, lambda_reg,
                                   x_mid, y, t_p, t0_vec)
            l_short = F.mse_loss(pred_short, target_pair)
        else:
            # Interpolant mode: x_t built from x_0, target is x_0.
            with torch.no_grad():
                x_mid = self._fwd(student, obs_cost, prior_cost, lambda_reg,
                                  x_t_interp, y, t_s, t_p)
            pred_short = self._fwd(student, obs_cost, prior_cost, lambda_reg,
                                   x_mid, y, t_p, t0_vec)
            l_short = F.mse_loss(pred_short, x0)

        loss = (self.lambda_pair  * l_pair
              + self.lambda_long  * l_long
              + self.lambda_short * l_short)

        return VarCMOutput(
            loss=loss, l_pair=l_pair, l_long=l_long, l_short=l_short,
            num_timesteps=N, times=times,
            t_sampled=t_s, t_p_sampled=t_p,
        )


@dataclass
class LitVarCMConfig:
    initial_ema_decay_rate:       float = 0.95
    student_model_ema_decay_rate: float = 0.99993
    lr:                           float = 1e-4
    betas:              Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor:    float = 1e-5
    lr_scheduler_iters:           int   = 10_000
    lambda_reg_init:              float = 1.0
    total_training_steps:         int   = 5_000


# LitVarCM is defined here but imports pytorch_lightning lazily to avoid a hard
# dependency when importing the module without a Lightning environment.
try:
    from pytorch_lightning import LightningModule
    from pytorch_lightning.utilities import rank_zero_info

    class LitVarCM(LightningModule):
        """
        Lightning module for deterministic VarCM.

            L = λ_pair  * L_pair   (pairwise consistency — both at t''=0)
              + λ_long  * L_long   (long-chain anchor  x_T -> t -> 0 -> x_0)
              + λ_short * L_short  (short-chain anchor x_t -> t'-> 0 -> x_0)
        """

        def __init__(
            self,
            var_cm_training:   VarCMTraining,
            student_model:     nn.Module,
            teacher_model:     nn.Module,
            ema_student_model: nn.Module,
            obs_cost:          nn.Module,
            prior_cost:        nn.Module,
            config:            LitVarCMConfig,
        ) -> None:
            super().__init__()
            self.var_cm_training   = var_cm_training
            self.teacher_model     = teacher_model
            self.student_model     = student_model
            self.ema_student_model = ema_student_model
            self.obs_cost          = obs_cost
            self.prior_cost        = prior_cost
            self.config            = config
            self.num_timesteps     = var_cm_training.initial_timesteps

            self.lambda_reg = nn.Parameter(
                torch.tensor(config.lambda_reg_init, dtype=torch.float32)
            )
            for p in self.teacher_model.parameters():
                p.requires_grad = False
            for p in self.ema_student_model.parameters():
                p.requires_grad = False
            self.teacher_model.eval()
            self.ema_student_model.eval()

        def on_train_epoch_start(self) -> None:
            print(f"[Epoch {self.current_epoch}] N={self.num_timesteps}  "
                  f"λ_reg={self.lambda_reg.item():.4f}")

        def training_step(self, batch, batch_idx: int):
            if isinstance(batch, list):
                batch = batch[0]
            self.lambda_reg.data.clamp_(min=0.0)

            out = self.var_cm_training(
                self.student_model,
                self.teacher_model,
                self.obs_cost,
                self.prior_cost,
                self.lambda_reg,
                batch.tgt,
                batch.input,
                self.global_step,
                self.config.total_training_steps,
            )
            self.num_timesteps = out.num_timesteps

            if batch_idx % 10 == 0:
                times_str = "  ".join(f"{v:.3f}" for v in out.times.tolist())
                print(
                    f"[Ep {self.current_epoch} | step {self.global_step}]  "
                    f"N={out.num_timesteps}\n"
                    f"  grid    : [{times_str}]\n"
                    f"  sampled : t={out.t_sampled[0].item():.3f}  "
                    f"t'={out.t_p_sampled[0].item():.3f}\n"
                    f"  L={out.loss.item():.4f}  "
                    f"L_pair={out.l_pair.item():.4f}  "
                    f"L_long={out.l_long.item():.4f}  "
                    f"L_short={out.l_short.item():.4f}"
                )

            self.log_dict({
                "train_loss":    out.loss,
                "L_pair":        out.l_pair,
                "L_long":        out.l_long,
                "L_short":       out.l_short,
                "lambda_reg":    self.lambda_reg.detach(),
                "num_timesteps": float(out.num_timesteps),
            }, prog_bar=False)
            return out.loss

        def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
            ema_decay = ema_decay_rate_schedule(
                self.num_timesteps,
                self.config.initial_ema_decay_rate,
                self.var_cm_training.initial_timesteps,
            )
            update_ema_model_(self.teacher_model,     self.student_model, ema_decay)
            update_ema_model_(self.ema_student_model, self.student_model,
                              self.config.student_model_ema_decay_rate)
            self.log("ema_decay_rate", ema_decay)

        def configure_optimizers(self):
            params = (
                list(self.student_model.parameters())
                + list(self.obs_cost.parameters())
                + list(self.prior_cost.parameters())
                + [self.lambda_reg]
            )
            opt = torch.optim.Adam(params,
                                   lr=self.config.lr,
                                   betas=self.config.betas)
            sched = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=self.config.lr_scheduler_start_factor,
                total_iters=self.config.lr_scheduler_iters,
            )
            return [opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]

except ImportError:
    pass   # pytorch_lightning not available — VarCMTraining / VarCMOutput still usable


# ---------------------------------------------------------------------------
# Sampling / inference
# ---------------------------------------------------------------------------

class VariationalConsistencySampling:
    """
    Iterative sampling using the learned transport operator g_phi.

    Starting from x_T = sigma_noise * z, applies g_phi(x, t_i, t_{i-1})
    stepping backwards through the Karras schedule from T=1 to t=0.

    The number of steps `nsteps` is passed at call time (not at construction),
    exactly as in ConsistencySamplingAndEditingFewSteps_TimeEmbedding.

    Parameters
    ----------
    sigma_noise : float
        Std of the initial noise x_T = sigma_noise * z (same as training).
    rho : float
        Karras schedule exponent (used only when noise_schedule="edm").
    noise_schedule : str
        Controls the sequence of *noise levels* sigma_k at each sampling step,
        i.e. how fast variance collapses from sigma_noise down to ~0.

        The time t_k is derived as t_k = sigma_k / sigma_noise, so the shape
        of the std-vs-step curve is directly controlled -- independently of the
        training time grid.

        "linear"  : uniform steps in t (sigma ∝ k).  Flat std collapse.
        "cosine"  : sigma_k = sigma_noise * cos(pi/2 * k/(N-1))
                    Fast initial drop, slow refinement near the end.
                    (same spirit as the cosine beta-schedule in DDPM)
        "power"   : sigma_k = sigma_noise * (1 - k/(N-1))^p  (p=schedule_power)
                    p>1 → slow initial drop, fast final refinement.
                    p<1 → fast initial drop, slow final refinement.
        "edm"     : geometric (Karras-style): sigma_k spaced log-uniformly
                    between sigma_noise and sigma_min.  Good for large sigma_noise.

    schedule_power : float
        Exponent used when noise_schedule="power".  p<1 for fast-then-slow,
        p>1 for slow-then-fast.  (Default 0.5 → fast initial denoising.)
    schedule : class
        LinearSchedule (default) or TrigSchedule.
    """

    def __init__(
        self,
        sigma_noise: float = 1.0,
        rho: float = 7.0,
        noise_schedule: str = "cosine",
        schedule_power: float = 0.5,
        schedule=None,
        conditioning_mode: str = "grad",  # "grad" or "obs"
    ) -> None:
        self.sigma_noise = sigma_noise
        self.rho = rho
        self.noise_schedule = noise_schedule
        self.schedule_power = schedule_power
        self.schedule = schedule if schedule is not None else DEFAULT_SCHEDULE
        self.conditioning_mode = conditioning_mode

    def __call__(
        self,
        model: nn.Module,
        obs_cost: nn.Module,
        prior_cost: nn.Module,
        lambda_reg: nn.Parameter,
        noise: Tensor,
        y: Tensor,
        nsteps: int = 10,
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run the reverse transport from T=1 to t=0.

        Parameters
        ----------
        model : nn.Module
            Trained student / EMA model.
        noise : Tensor  (B, C, H, W)
            Pure Gaussian noise; scaled internally by sigma_noise.
        y : Tensor  (B, C, H, W)
            Observations.
        nsteps : int
            Number of denoising steps (analogous to nsteps in the reference sampler).
            More steps = finer trajectory, at the cost of compute.
        clip_denoised : bool
            Clip output to [-1, 1].
        verbose : bool
            Show progress bar.

        Returns
        -------
        x : Tensor          final reconstructed state  (B, C, H, W)
        all_xs : Tensor     (nsteps, B, C, H, W) full trajectory
        """
        device = noise.device
        dtype = noise.dtype
        B = noise.shape[0]

        # ----------------------------------------------------------------
        # Build the sampling time sequence t_k, descending from 1 to ~0.
        # Derived from a noise-level schedule sigma_k in [sigma_noise, ~0],
        # with t_k = sigma_k / sigma_noise.
        # The shape of sigma(k) controls the std collapse curve -- this is
        # independent of the training time grid.
        # ----------------------------------------------------------------
        N = max(nsteps, 3)
        k = torch.linspace(0.0, 1.0, N, device=device)   # k in [0,1], k=0 at T=1

        ns = self.noise_schedule
        if ns == "linear":
            # uniform: sigma decreases linearly
            sigma_k = self.sigma_noise * (1.0 - k)
        elif ns == "cosine":
            # cosine: fast initial drop, slow final refinement
            sigma_k = self.sigma_noise * torch.cos(math.pi / 2.0 * k)
        elif ns == "power":
            # power: (1-k)^p  -- p<1 for fast-then-slow, p>1 for slow-then-fast
            sigma_k = self.sigma_noise * (1.0 - k) ** self.schedule_power
        elif ns == "edm":
            # geometric (Karras-style): log-uniform in sigma
            sigma_min = self.sigma_noise * 1e-3
            log_min = math.log(sigma_min)
            log_max = math.log(self.sigma_noise)
            sigma_k = torch.exp(torch.linspace(log_max, log_min, N, device=device))
        else:
            raise ValueError(f"Unknown noise_schedule='{ns}'. "
                             "Choose from: 'linear', 'cosine', 'power', 'edm'.")

        # Build times: N-1 intermediate steps + final step to exactly 0.
        # times[0]=1 (start), times[1..N-2]=intermediate, times[N-1]=eps (near 0).
        # We then append a hard 0 so the last call is always g(x, t_eps, 0),
        # ensuring c_skip→0, c_out→1 and the network outputs F_theta directly.
        sigma_k = sigma_k.clamp(min=self.sigma_noise * 1e-4,
                                 max=self.sigma_noise).to(dtype=dtype)
        times_inner = sigma_k / self.sigma_noise       # (N,) descending: [1, ..., ~1e-4]
        t_zero = torch.zeros(1, device=device, dtype=dtype)
        times = torch.cat([times_inner, t_zero], dim=0)  # (N+1,), last=0

        # x_T = sigma_noise * z
        x = noise * self.sigma_noise
        all_xs = [x]

        step_iter = range(len(times) - 1)
        if verbose:
            step_iter = tqdm(step_iter, desc="Variational sampling")

        for i in step_iter:
            t_cur  = torch.full((B,), times[i].item(),     dtype=dtype, device=device)
            t_prev = torch.full((B,), times[i + 1].item(), dtype=dtype, device=device)
            # At t_prev=0: c_skip=0, c_out=1 -> output = F_theta directly (pure prediction)
            t_prev = t_prev.clamp(min=0.0)

            with torch.no_grad():
                x, _ = model_variational_forward_wrapper(
                    model, obs_cost, prior_cost, lambda_reg,
                    x, y, t_cur, t_prev, schedule=self.schedule,
                    sigma_noise=self.sigma_noise,
                    conditioning_mode=self.conditioning_mode, **kwargs
                )

            if clip_denoised:
                x = x.clamp(-1.0, 1.0)
            all_xs.append(x)

        return x, torch.stack(all_xs, dim=0)
