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

from ..utils import *
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
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class VariationalConsistencyTrainingOutput:
    # Term 1: pairwise consistency via shared x_T  (anti-identity guarantee)
    pairwise_grad:     Tensor   # out_t   -- student, gradient flows here
    pairwise_stopgrad: Tensor   # out_t'  -- teacher stop-grad target

    # Term 2: x0 anchoring via double composition to t=0 (no assumption on g)
    #   gradient flows only through the outer call (inner is detached)
    anchor_at_t:   Tensor   # g(sg[g(x_T,T,t)],  t,  0) -- should recover x0
    anchor_at_tp:  Tensor   # g(sg[g(x_T,T,t')], t', 0) -- should recover x0
    x0:            Tensor   # ground truth x0

    # Term 4 (bootstrap): direct g(x_T, T, 0) ~ x0 -- single call, known target
    direct_x0:     Tensor

    # Term 3: prior regularisation  ||x_0 - Phi(x_0)||^2
    prior_reg: Tensor

    # Warmup support: raw states needed to compute linear-interpolant targets
    x_T:    Tensor   # noisy initial state  x_0 + sigma_noise * z
    x_at_t: Tensor   # g_student(x_T, T, t)  -- 1st hop at t
    x_at_tp: Tensor  # g_student(x_T, T, t') -- 1st hop at t'

    num_timesteps:   int
    times:           Tensor   # (N,) solver times in [0,1]
    t_sampled:       Tensor   # (B,) t
    t_prime_sampled: Tensor   # (B,) t'
    t_pp_sampled:    Tensor   # (B,) t''


# ---------------------------------------------------------------------------
# Training class
# ---------------------------------------------------------------------------

class VariationalConsistencyTraining:
    """
    Composition-based pairwise consistency training for variational dynamics.

    Teacher/student strategy:
      - student_model : the model being optimised (gradients flow through it)
      - teacher_model : EMA of student (fast decay ~0.99); provides stable
                        stop-grad targets for Term 1.  Without the teacher,
                        the stop-grad target is computed from current student
                        weights and oscillates as much as the student itself
                        -- a classic "moving target" instability.

    Parameters
    ----------
    initial_timesteps : int
        Number of Karras discretisation steps at the start of training.
    final_timesteps : int
        Number of Karras discretisation steps at the end of training.
    sigma_noise : float
        Std of the initial Gaussian noise x_T = x_0 + sigma_noise * z.
    lambda_pair : float
        Weight of T1 (pairwise consistency).
    lambda_interp : float
        Weight of T2 (composed-path interpolant supervision).
    lambda_prior : float
        Weight of T3 (prior regularisation ||x - Phi(x)||^2).
    rho : float
        Karras schedule exponent (default 7.0).
    schedule_power : float
        Exponent of the power-law time grid.  p=1 gives uniform spacing.
        p>1 concentrates steps near t=0 (refinement) and spreads them at
        high t (fast initial denoising), producing a non-linear std collapse.
        Typical values: 1.5 (mild), 2.0 (moderate), 3.0 (aggressive).
    schedule : class
        LinearSchedule (default) or TrigSchedule.
    """

    def __init__(
        self,
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
        sigma_noise: float = 1.0,
        lambda_pair: float = 1.0,
        lambda_interp: float = 1.0,
        lambda_prior: float = 0.1,
        lambda_direct: float = 1.0,   # bootstrap: g(x_T, T, 0) ~ x0 (single call)
        rho: float = 7.0,
        schedule_power: float = 1.0,
        schedule=None,
        warmup_epochs: int = 50,
        conditioning_mode: str = "grad",  # "grad": use nabla_x J(x), "obs": use cat([y, mask])
    ) -> None:
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
        self.sigma_noise = sigma_noise
        self.lambda_pair = lambda_pair
        self.lambda_interp = lambda_interp
        self.lambda_prior = lambda_prior
        self.lambda_direct = lambda_direct
        self.rho = rho
        self.schedule_power = schedule_power
        self.schedule = schedule if schedule is not None else DEFAULT_SCHEDULE
        self.warmup_epochs = warmup_epochs  # epochs during which we blend toward the linear interpolant
        self.conditioning_mode = conditioning_mode

    def _times(self, num_timesteps: int, device) -> Tensor:
        """Returns `num_timesteps` solver times in (0, 1], ascending.

        The grid is shifted so that  times[0] = 1/(N-1)  instead of 0.
        This avoids the  t = 0  singularity where
            c_skip(t, t') = alpha(t') / alpha(t)  ->  infinity
        and the UNet output is completely masked by the skip connection,
        killing the gradient signal through L_pair.

        schedule_power > 1: power-law grid, dense near t=0 and sparse at
        high t. This forces the model to learn large initial denoising jumps
        and fine refinement steps near t=0, producing a non-linear std collapse
        during sampling (fast denoising at start, slow refinement at the end).

        num_timesteps is recomputed at every training step via timesteps_schedule.
        """
        N = max(num_timesteps, 5)
        # Uniform base grid on (0, 1]: avoids the t=0 singularity.
        raw = torch.linspace(1.0 / (N - 1), 1.0, N, device=device)
        # Apply power law: p>1 squeezes points toward t=0.
        times = raw ** self.schedule_power
        return times

    def __call__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        obs_cost: nn.Module,
        prior_cost: nn.Module,
        lambda_reg: nn.Parameter,
        x: Tensor,
        y: Tensor,
        current_training_step: int,
        total_training_steps: int,
        **kwargs: Any,
    ) -> VariationalConsistencyTrainingOutput:
        """
        One training step.  Three terms:

        Term 1 - Pairwise consistency (teacher/student, via shared x_T)
          Sample t < t' < t'' on the schedule.
          h_t  = g_teacher(x_T, T, t)   [no grad, teacher EMA]
          h_t' = g_teacher(x_T, T, t')  [no grad, teacher EMA]
          out_t  = g_student(h_t,  t,  t'')  [GRAD]
          out_t' = g_teacher(h_t', t', t'') [stop-grad, teacher EMA]
          L1 = ||out_t - sg(out_t')||^2

          The teacher (EMA of student, fast decay) provides a stable target:
          its weights change slowly, so the stop-grad target is nearly
          constant between steps -- avoids the moving-target instability that
          would occur if the same student computed both branches.

        Term 2 - Composed-path interpolant supervision
          L2 = ||g_student(x_T, T, t)  - x_{interp,t} ||^2
             + ||g_student(x_T, T, t') - x_{interp,t'}||^2
          where x_{interp,s} = alpha(s)*x_T + beta(s)*x_0.

          Forces the output variance to match the schedule:
          large t -> output stays close to x_T,  small t -> approaches x_0.

        Term 3 - Prior regularisation
          L3 = ||x_0 - Phi(x_0)||^2
        """
        B, C, H, W = x.shape
        device = x.device
        dtype  = x.dtype
        sched  = self.schedule

        num_timesteps = timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )

        times = self._times(num_timesteps, device)  # (N,) ascending in [0,1]
        N = len(times)

        # x_T = x_0 + sigma_noise * z
        noise = torch.randn_like(x)
        x_T   = x + self.sigma_noise * noise

        T_val = times[-1]   # 1.0
        T_vec = torch.full((B,), T_val, dtype=dtype, device=device)

        # Sample three ordered indices i < j < k => t < t' < t''
        # Start from i=0 (safe now that times[0] > 0 after _times shift).
        i_idx = torch.randint(0, max(N - 2, 1), (B,), device=device).clamp(max=N - 3)
        j_idx = (i_idx + 1 + (torch.rand(B, device=device) *
                 (N - 1 - i_idx - 1).float().clamp(min=1)).long()).clamp(max=N - 2)
        k_idx = (j_idx + 1).clamp(max=N - 1)

        # Safety: ensure t_vals > 0 to avoid c_skip singularity
        t_vals   = times[i_idx].clamp(min=1e-3)    # (B,)  t
        tp_vals  = times[j_idx]                     # (B,)  t'
        tpp_vals = times[k_idx]                     # (B,)  t''

        # --- T1: pairwise consistency (teacher EMA -> stable stop-grad) ---
        _cm = self.conditioning_mode
        _sn = self.sigma_noise
        with torch.no_grad():
            h_t,  _ = model_variational_forward_wrapper(
                teacher_model, obs_cost, prior_cost, lambda_reg,
                x_T, y, T_vec, t_vals, schedule=sched, sigma_noise=_sn, conditioning_mode=_cm, **kwargs
            )
            h_tp, _ = model_variational_forward_wrapper(
                teacher_model, obs_cost, prior_cost, lambda_reg,
                x_T, y, T_vec, tp_vals, schedule=sched, sigma_noise=_sn, conditioning_mode=_cm, **kwargs
            )

        # out_t  = g_student(h_t, t, t'')  [GRAD]
        out_t, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            h_t.detach(), y, t_vals, tpp_vals, schedule=sched, sigma_noise=_sn, conditioning_mode=_cm, **kwargs
        )
        # out_t' = g_teacher(h_t', t', t'')  [stop-grad]
        with torch.no_grad():
            out_tp, _ = model_variational_forward_wrapper(
                teacher_model, obs_cost, prior_cost, lambda_reg,
                h_tp.detach(), y, tp_vals, tpp_vals, schedule=sched, sigma_noise=_sn, conditioning_mode=_cm, **kwargs
            )

        # --- T2: x0 anchoring -- g(g(x_T, T, t), t, 0) should recover x0 ---
        # Full backprop through both calls: the T2 gradient flows back to the
        # first hop, forcing g(x_T,T,t) to produce an informative x_at_t
        # on which grad_J(x_at_t) points toward the observations.
        # Without this gradient, x_at_t stays noisy and T2 cannot converge.
        # (bf16 handles the dynamic range -- no stop-grad needed here)
        t_zero = torch.zeros(B, dtype=dtype, device=device)
        x_at_t, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            x_T, y, T_vec, t_vals, schedule=sched, sigma_noise=_sn, conditioning_mode=_cm, **kwargs
        )
        x_at_tp, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            x_T, y, T_vec, tp_vals, schedule=sched, sigma_noise=_sn, conditioning_mode=_cm, **kwargs
        )
        anchor_t, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            x_at_t, y, t_vals, t_zero, schedule=sched, sigma_noise=_sn, conditioning_mode=_cm, **kwargs
        )
        anchor_tp, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            x_at_tp, y, tp_vals, t_zero, schedule=sched, sigma_noise=_sn, conditioning_mode=_cm, **kwargs
        )

        # --- T4 (bootstrap): direct supervision g(x_T, T, 0) ~ x0 ---
        # Single call, known target x0. Bootstraps denoising capacity
        # from the start, speeds up T2 convergence.
        direct_x0, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            x_T, y, T_vec, t_zero, schedule=sched, sigma_noise=_sn, conditioning_mode=_cm, **kwargs
        )

        # --- T3: prior regularisation ---
        prior_reg = F.mse_loss(x, prior_cost.forward_ae(x))

        return VariationalConsistencyTrainingOutput(
            pairwise_grad=out_t,
            pairwise_stopgrad=out_tp.detach(),
            anchor_at_t=anchor_t,
            anchor_at_tp=anchor_tp,
            x0=x.detach(),
            direct_x0=direct_x0,
            prior_reg=prior_reg,
            num_timesteps=num_timesteps,
            times=times,
            t_sampled=t_vals,
            t_prime_sampled=tp_vals,
            t_pp_sampled=tpp_vals,
            x_T=x_T.detach(),
            x_at_t=x_at_t.detach(),
            x_at_tp=x_at_tp.detach(),
        )

    @staticmethod
    def _normalized_mse(
        pred: Tensor,
        target: Tensor,
        eps: float = 1e-4,
        ref_var: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Scale-invariant MSE: normalise by ref_var (defaults to var(target)) before
        computing the mean squared error.

            L = mean( (pred - target)^2 ) / (ref_var + eps)

        ref_var defaults to var(target), but for L_pair the target variance is
        dominated by the shared c_skip*x_T term (~sigma_noise^2 ~ 6400) which
        cancels in the difference pred-target.  Passing ref_var=var(x0) (~1)
        gives a properly scaled loss in that case.
        eps is set relative to the expected signal variance (~1 after T2 kicks
        in) so it only activates when both pred and target are near-zero.
        """
        if ref_var is None:
            ref_var = target.detach().var()
        scale = ref_var + eps
        return F.mse_loss(pred, target) / scale

    def compute_loss(
        self,
        output: VariationalConsistencyTrainingOutput,
        current_epoch: int = 0,
    ) -> Tensor:
        """
        Four-term loss with linear-interpolant warmup.

        During the first `warmup_epochs` epochs, a blending factor
            beta_w = max(0, 1 - epoch / warmup_epochs)
        gradually shifts targets toward the linear interpolant
            x_interp(t) = alpha(t)*x_T + beta(t)*x0
        giving the network a well-defined regression target from the start.
        beta_w = 1 at epoch 0, beta_w = 0 at epoch >= warmup_epochs.

        T1 - L_pair   : nMSE(student_out, blended_target)
          target = (1-beta_w)*sg(teacher_out) + beta_w*x_interp(t'')
        T2 - L_interp : blended anchor supervision
          std  term: 0.5*(nMSE(anchor_t, x0) + nMSE(anchor_tp, x0))
          warm term: 0.5*(nMSE(x_at_t, x_interp_t) + nMSE(x_at_tp, x_interp_tp))
          loss = (1-beta_w)*std + beta_w*warm
        T3 - L_prior  : ||x0 - Phi(x0)||^2
        T4 - L_direct : nMSE(g(x_T, T, 0), x0)  (bootstrap)

        Returns (total, loss_pair, loss_interp, loss_prior, loss_direct, beta_w).
        """
        sched  = self.schedule
        beta_w = max(0.0, 1.0 - current_epoch / max(1, self.warmup_epochs))

        def _expand(t: Tensor) -> Tensor:
            return t.view(-1, 1, 1, 1).to(dtype=output.x0.dtype)

        t   = _expand(output.t_sampled)
        tp  = _expand(output.t_prime_sampled)
        tpp = _expand(output.t_pp_sampled)
        x0  = output.x0
        x_T = output.x_T

        x_interp_t   = sched.alpha(t)   * x_T + sched.beta(t)   * x0
        x_interp_tp  = sched.alpha(tp)  * x_T + sched.beta(tp)  * x0
        x_interp_tpp = sched.alpha(tpp) * x_T + sched.beta(tpp) * x0

        # T1: blend pairwise target toward linear interpolant during warmup
        if beta_w > 0.0:
            target_pair = (
                (1.0 - beta_w) * output.pairwise_stopgrad
                + beta_w * x_interp_tpp.detach()
            )
        else:
            target_pair = output.pairwise_stopgrad
        x0_var = output.x0.detach().var()
        loss_pair = self._normalized_mse(output.pairwise_grad, target_pair, ref_var=x0_var)

        # T2: blend anchor supervision toward direct interpolant regression
        loss_interp_std = 0.5 * (
            self._normalized_mse(output.anchor_at_t,  x0) +
            self._normalized_mse(output.anchor_at_tp, x0)
        )
        # x_interp_t/tp are dominated by alpha(t)*x_T whose variance is
        # O(t^2 * sigma_noise^2) >> var(x0).  Normalise by var(x0) so the
        # warmup term lives on the same scale as loss_interp_std.
        loss_interp_warm = 0.5 * (
            self._normalized_mse(output.x_at_t,  x_interp_t.detach(),  ref_var=x0_var) +
            self._normalized_mse(output.x_at_tp, x_interp_tp.detach(), ref_var=x0_var)
        )
        loss_interp = (1.0 - beta_w) * loss_interp_std + beta_w * loss_interp_warm

        loss_prior  = output.prior_reg
        # T4 bootstrap: direct supervision g(x_T, T, 0) ~ x0
        loss_direct = self._normalized_mse(output.direct_x0, x0)

        total = (
            self.lambda_pair   * loss_pair
            + self.lambda_interp * loss_interp
            + self.lambda_prior  * loss_prior
            + self.lambda_direct * loss_direct
        )
        return total, loss_pair, loss_interp, loss_prior, loss_direct, beta_w


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
