"""
Pairwise Self-Consistency in Variational Dynamics (without explicit forward SDE)
=================================================================================

Theory — Interpolant Framework
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

Training Loss — Composition-Based Supervision
----------------------------------------------
"Times" t in [0, 1] are abstract solver indices (not diffusion noise levels).

The key insight: at inference the model runs a chain

    x_T  →[g(·,T,t_1)]→  x_1  →[g(·,t_1,t_2)]→  x_2  → … → x_0

We always have access to x_T (pure noise) and x_0 (ground truth), so every
composed state  h_t = g(x_T, T, t)  can be analytically cross-checked against
the interpolant  x_{interp,t} = alpha(t)*x_T + beta(t)*x_0.

Three terms, teacher/student strategy:

  Term 1 – Pairwise consistency (anti-identity, via shared x_T):
    Sample t < t' < t'' from the schedule.
    h_t   = g_teacher(x_T, T, t)   [no grad, teacher EMA]
    h_t'  = g_teacher(x_T, T, t')  [no grad, teacher EMA]
    out_t  = g_student(h_t,  t,  t'')  [GRAD]
    out_t' = g_teacher(h_t', t', t'') [stop-grad, teacher EMA]
    L1 = ||out_t - sg(out_t')||²

    Using the teacher (EMA of student) for the stop-grad branch provides
    a STABLE target: the teacher evolves slowly (EMA decay ~0.99+), so the
    target barely changes between consecutive training steps.  Without this,
    the stop-grad target is computed from the current student weights and
    oscillates as strongly as the student — an unstable "moving target".
    Both branches start from the SAME x_T and target the SAME t'', so they
    cannot collapse to identity.  The unique minimum is the true x_{t''}.

  Term 2 – Composed-path interpolant supervision (progressive denoising):
    Supervise the full composition h_t = g(x_T, T, t) against x_{interp,t}.
    L2 = ||g(x_T, T, t)  - x_{interp,t} ||²
       + ||g(x_T, T, t') - x_{interp,t'}||²

    This forces g to predict states with the *right variance level* for each t:
    g(x_T, T, large_t) must stay close to x_T,
    g(x_T, T, small_t) must approach x_0.
    Without this term, Term 1 alone can be satisfied by any fixed denoising
    level (e.g. always outputting x_0 for all t).

  Term 3 – Prior regularisation:
    L3 = ||x_0 - Phi(x_0)||²

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
    def c_in(t: Tensor) -> Tensor:
        """Normalise input: 1 / ||[alpha(t), beta(t)]||_2."""
        return 1.0 / (t ** 2 + (1.0 - t) ** 2).sqrt().clamp(min=1e-8)

    @staticmethod
    def c_skip(t: Tensor, t_prime: Tensor) -> Tensor:
        """alpha(t') / alpha(t) — carries x forward deterministically."""
        return t_prime / t.clamp(min=1e-8)

    @staticmethod
    def c_out(t: Tensor, t_prime: Tensor) -> Tensor:
        """beta(t') - c_skip * beta(t) — scale of the learned residual."""
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
    def c_in(t: Tensor) -> Tensor:
        return torch.ones_like(t)

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
    to F_theta — no channel concatenation.  Inside the UNet, it goes through
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

    B, C, H, W = x.shape

    # ---- Compute grad_J = nabla_x J(x)  -----------------------------------
    # Match x_leaf dtype to the cost modules to avoid dtype mismatches under
    # AMP (e.g. bfloat16 inference) or float32 training.
    try:
        cost_dtype = next(prior_cost.parameters()).dtype
    except StopIteration:
        cost_dtype = x.dtype  # obs_cost / prior_cost have no parameters → use x dtype
    x_leaf = x.detach().to(dtype=cost_dtype).requires_grad_(True)
    y_c    = y.to(dtype=cost_dtype)
    with torch.enable_grad():
        Jo  = obs_cost(x_leaf, y_c)
        Jb  = prior_cost(x_leaf)
        J   = Jo + lambda_reg.to(dtype=cost_dtype) * Jb
        grad_J_c = torch.autograd.grad(
            J, x_leaf, create_graph=False, allow_unused=True
        )[0]
    if grad_J_c is None:
        grad_J_c = torch.zeros_like(x_leaf)
    grad_J = grad_J_c.detach().to(dtype=x.dtype)  # cast back to network dtype

    # Work with detached x from here on (no gradient through transport steps)
    x = x.detach()

    # ---- Preconditioning scalars (broadcast to (B, 1, 1, 1)) --------------
    def _expand(s: Tensor) -> Tensor:
        return s.view(B, 1, 1, 1)

    c_in   = _expand(schedule.c_in(t))             # (B, 1, 1, 1)
    c_skip = _expand(schedule.c_skip(t, t_prime))  # (B, 1, 1, 1)
    c_out  = _expand(schedule.c_out(t, t_prime))   # (B, 1, 1, 1)

    # ---- Build network inputs -----------------------------------------------
    # x and grad_J are passed as *separate* arguments so the UNet can project
    # them independently (each through its own Conv2d) before adding.
    net_x     = c_in * x        # (B, C, H, W)  — whitened state
    # grad_J is NOT scaled by c_in: it already carries units of 1/state
    # and its own learned projection handles the scaling.

    # ---- Network forward ---------------------------------------------------
    # F_theta output: (B, C, H, W) — direct prediction of x_0
    F_out = model(net_x, grad_J, t, t_prime, **kwargs)  # (B, C, H, W)

    # ---- Preconditioned output: c_skip * x + c_out * F_theta --------------
    x_out = c_skip * x + c_out * F_out             # (B, C, H, W)

    return x_out, grad_J


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class VariationalConsistencyTrainingOutput:
    # Term 1: pairwise consistency via shared x_T  (anti-identity guarantee)
    # h_t  = g(x_T, T, t)  [no grad];  out_t  = g(h_t,  t,  t'')  [GRAD]
    # h_t' = g(x_T, T, t') [no grad];  out_t' = g(h_t', t', t'') [stop-grad]
    pairwise_grad:     Tensor   # out_t   — gradient flows here
    pairwise_stopgrad: Tensor   # out_t'  — stop-grad target

    # Term 2: x₀ anchoring via full composition to t=0
    # g(g(x_T, T, t),  t,  0) should recover x₀
    # g(g(x_T, T, t'), t', 0) should recover x₀
    anchor_at_t:        Tensor   # g(g(x_T, T, t),  t,  0) — composed to t=0 via t
    anchor_at_tp:       Tensor   # g(g(x_T, T, t'), t', 0) — composed to t=0 via t'
    x0:                 Tensor   # ground truth x₀ (target for both anchors)

    # Term 3: prior regularisation  ||x_0 - Phi(x_0)||^2
    prior_reg: Tensor   # scalar

    # Term 4: gradient conjugacy — grad_J at three successive iterates
    # g_t = nabla J(h_t),  g_tp = nabla J(h_tp),  g_tpp = nabla J(out_t)
    grad_J_t:   Tensor   # (B, C, H, W)  nabla J at iterate h_t
    grad_J_tp:  Tensor   # (B, C, H, W)  nabla J at iterate h_tp
    grad_J_tpp: Tensor   # (B, C, H, W)  nabla J at iterate out_t

    # Term 5: stochastic trajectory warmup
    # Student iterates at the three times (same as pairwise path) vs stochastic targets
    composed_t:  Tensor   # (B, C, H, W) g_student(x_T, T, t)   — student iterate at t
    composed_tp: Tensor   # (B, C, H, W) g_student(x_T, T, t')  — student iterate at t'
    out_t:       Tensor   # (B, C, H, W) g_student(h_t,  t, t'') — student iterate at t''
    h_t:     Tensor   # (B, C, H, W) teacher iterate at t  (for J eval / conjugacy)
    h_tp:    Tensor   # (B, C, H, W) teacher iterate at t' (for J eval / conjugacy)
    traj_t:  Tensor   # (B, C, H, W) stochastic trajectory target at t
    traj_tp: Tensor   # (B, C, H, W) stochastic trajectory target at t'
    traj_tpp: Tensor  # (B, C, H, W) stochastic trajectory target at t''

    # J values at the three iterates (for descent constraint)
    J_t:   Tensor     # (B,) J(h_t)
    J_tp:  Tensor     # (B,) J(h_tp)
    J_tpp: Tensor     # (B,) J(out_t)

    num_timesteps: int
    times: Tensor            # 1-D tensor of N solver times in [0,1]
    t_sampled:   Tensor      # (B,) t  values used this step
    t_prime_sampled: Tensor  # (B,) t' values used this step
    t_pp_sampled:    Tensor  # (B,) t'' values used this step


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
                        — a classic "moving target" instability.

    Parameters
    ----------
    initial_timesteps : int
        Number of Karras discretisation steps at the start of training.
    final_timesteps : int
        Number of Karras discretisation steps at the end of training.
    sigma_noise : float
        Std of the initial Gaussian noise x_T = x_0 + sigma_noise * z.
    lambda_pair : float
        Weight of Term 1 (pairwise consistency via x_T composition).
    lambda_interp : float
        Weight of Term 2 (composed-path interpolant supervision).
    lambda_prior : float
        Weight of Term 3 (prior regularisation).
    rho : float
        Karras schedule exponent (default 7.0).
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
        lambda_conj: float = 0.1,
        lambda_traj: float = 1.0,
        warmup_epochs: int = 50,
        warmup_gamma: float = 0.05,
        rho: float = 7.0,
        schedule=None,
    ) -> None:
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
        self.sigma_noise = sigma_noise
        self.lambda_pair = lambda_pair
        self.lambda_interp = lambda_interp
        self.lambda_prior = lambda_prior
        self.lambda_conj = lambda_conj
        self.lambda_traj = lambda_traj
        self.warmup_epochs = warmup_epochs
        self.warmup_gamma = warmup_gamma
        self.rho = rho
        self.schedule = schedule if schedule is not None else DEFAULT_SCHEDULE

    def _times(self, num_timesteps: int, device) -> Tensor:
        """Returns `num_timesteps` solver times in (0, 1], ascending.

        The grid is shifted so that  times[0] = 1/(N-1)  instead of 0.
        This avoids the  t = 0  singularity where
            c_skip(t, t') = alpha(t') / alpha(t)  ->  infinity
        and the UNet output is completely masked by the skip connection,
        killing the gradient signal through L_pair.

        num_timesteps is recomputed at every training step via timesteps_schedule.
        """
        N = max(num_timesteps, 5)
        # Original Karras grid: [0, 1/(N-1), 2/(N-1), ..., 1]
        # Shifted grid:         [1/(N-1), 2/(N-1), ..., 1]   (drop t=0, keep N-1 pts)
        # Then we re-expand to N pts: linspace from 1/(N-1) to 1, N pts.
        times = torch.linspace(1.0 / (N - 1), 1.0, N, device=device)
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
          constant between steps — avoids the moving-target instability that
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

        # --- Stochastic trajectory prior (warmup) ---
        # Sample a random trajectory family per batch element:
        #   alpha_phi(tau) = (tau / T)^p  with p ~ Uniform[1.5, 3.0]
        #   beta_phi(tau)  = 1 - alpha_phi(tau)
        # This covers parabolic (p=2, fast at beginning) to cubic shapes.
        # tau is the solver time in (0, 1], T = times[-1] = 1.
        p_phi = 1.5 + 1.5 * torch.rand(B, device=device, dtype=dtype)  # (B,) in [1.5, 3.0]
        def stochastic_interp(tau):
            """x_t^(phi) = alpha_phi(t) * x_T + (1 - alpha_phi(t)) * x_0"""
            alpha_phi = tau.pow(p_phi)                      # (B,)
            a = alpha_phi.view(B, 1, 1, 1)
            return a * x_T.detach() + (1.0 - a) * x.detach()

        traj_t   = stochastic_interp(t_vals)    # (B, C, H, W)
        traj_tp  = stochastic_interp(tp_vals)   # (B, C, H, W)
        traj_tpp = stochastic_interp(tpp_vals)  # (B, C, H, W)

        # --- Term 1: pairwise consistency (teacher provides stable stop-grad) ---
        # Compute h_t = g_teacher(x_T, T, t) and h_t' = g_teacher(x_T, T, t').
        with torch.no_grad():
            h_t, grad_J_t = model_variational_forward_wrapper(
                teacher_model, obs_cost, prior_cost, lambda_reg,
                x_T, y, T_vec, t_vals, schedule=sched, **kwargs
            )
            h_tp, grad_J_tp = model_variational_forward_wrapper(
                teacher_model, obs_cost, prior_cost, lambda_reg,
                x_T, y, T_vec, tp_vals, schedule=sched, **kwargs
            )

        # out_t  = g_student(h_t, t, t'')  [GRAD — gradient flows here]
        out_t, grad_J_tpp = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            h_t.detach(), y, t_vals, tpp_vals, schedule=sched, **kwargs
        )
        # out_t' = g_teacher(h_t', t', t'')  [stop-grad, stable EMA target]
        with torch.no_grad():
            out_tp, _ = model_variational_forward_wrapper(
                teacher_model, obs_cost, prior_cost, lambda_reg,
                h_tp.detach(), y, tp_vals, tpp_vals, schedule=sched, **kwargs
            )

        # --- Term 2: x₀ anchoring via full composition to t=0 ---
        # anchor_t  = g(g(x_T, T, t),  t,  0)  should recover x₀
        # anchor_tp = g(g(x_T, T, t'), t', 0)  should recover x₀
        # Step 1: g(x_T, T, t) and g(x_T, T, t') — recomputed with student grad
        t_zero = torch.zeros(B, dtype=dtype, device=device)  # target time = 0
        composed_t, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            x_T, y, T_vec, t_vals, schedule=sched, **kwargs
        )
        composed_tp, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            x_T, y, T_vec, tp_vals, schedule=sched, **kwargs
        )
        # Step 2: compose to t=0 — anchor to x₀
        anchor_t, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            composed_t, y, t_vals, t_zero, schedule=sched, **kwargs
        )
        anchor_tp, _ = model_variational_forward_wrapper(
            student_model, obs_cost, prior_cost, lambda_reg,
            composed_tp, y, tp_vals, t_zero, schedule=sched, **kwargs
        )

        # --- Term 3: prior regularisation ---
        x0_recon  = prior_cost.forward_ae(x)
        prior_reg = F.mse_loss(x, x0_recon)

        # --- Compute J values at the three iterates (for descent constraint) ---
        def _eval_J(state):
            """Evaluate J(state) = J_o(state, y) + lambda_reg * J_b(state), per sample."""
            with torch.no_grad():
                try:
                    cost_dtype = next(prior_cost.parameters()).dtype
                except StopIteration:
                    cost_dtype = state.dtype
                s = state.detach().to(dtype=cost_dtype)
                yc = y.to(dtype=cost_dtype)
                lr = lambda_reg.detach().to(dtype=cost_dtype)
                # Per-sample J: compute individually to get (B,) vector
                B_ = s.shape[0]
                J_vals = torch.empty(B_, device=state.device, dtype=state.dtype)
                for b in range(B_):
                    Jo_b = obs_cost(s[b:b+1], yc[b:b+1])
                    Jb_b = prior_cost(s[b:b+1])
                    J_vals[b] = (Jo_b + lr * Jb_b).to(dtype=state.dtype)
            return J_vals

        J_t   = _eval_J(h_t)    # (B,)
        J_tp  = _eval_J(h_tp)   # (B,)
        J_tpp = _eval_J(out_t)  # (B,)

        return VariationalConsistencyTrainingOutput(
            pairwise_grad=out_t,
            pairwise_stopgrad=out_tp.detach(),
            anchor_at_t=anchor_t,
            anchor_at_tp=anchor_tp,
            x0=x.detach(),
            prior_reg=prior_reg,
            grad_J_t=grad_J_t,
            grad_J_tp=grad_J_tp,
            grad_J_tpp=grad_J_tpp,
            composed_t=composed_t,
            composed_tp=composed_tp,
            out_t=out_t,
            h_t=h_t.detach(),
            h_tp=h_tp.detach(),
            traj_t=traj_t,
            traj_tp=traj_tp,
            traj_tpp=traj_tpp,
            J_t=J_t,
            J_tp=J_tp,
            J_tpp=J_tpp,
            num_timesteps=num_timesteps,
            times=times,
            t_sampled=t_vals,
            t_prime_sampled=tp_vals,
            t_pp_sampled=tpp_vals,
        )

    @staticmethod
    def _normalized_mse(pred: Tensor, target: Tensor, eps: float = 1e-4) -> Tensor:
        """
        Scale-invariant MSE: normalise by the variance of the target before
        computing the mean squared error.

            L = mean( (pred - target)^2 ) / (var(target) + eps)

        This keeps L_pair in a [0, 1] range regardless of the absolute scale
        of the outputs (which can be ~sigma_noise^2 ~ 6400 at initialisation).
        eps is set relative to the expected signal variance (~1 after T2 kicks
        in) so it only activates when both pred and target are near-zero.
        """
        scale = target.detach().var() + eps
        return F.mse_loss(pred, target) / scale

    def warmup_weight(self, current_epoch: int) -> float:
        """
        Warmup schedule for the trajectory matching loss:
            lambda_traj(e) = 1                             if e <= warmup_epochs
            lambda_traj(e) = exp(-gamma * (e - E_warm))    if e >  warmup_epochs
        """
        if current_epoch <= self.warmup_epochs:
            return 1.0
        return math.exp(-self.warmup_gamma * (current_epoch - self.warmup_epochs))

    def compute_loss(
        self,
        output: VariationalConsistencyTrainingOutput,
        current_epoch: int = 0,
    ) -> Tensor:
        """
        Six-term composition-based consistency loss with warmup.

        T1 - Pairwise consistency:  nMSE(student, sg(teacher))
        T2 - Interpolant supervision
        T3 - Prior regularisation
        T4 - Gradient conjugacy:  (g_{t+2}^T (g_{t+1} - g_t))^2  (normalised)
        T5 - Descent constraint:  max(0, J(x_{t+1}) - J(x_t))
        T6 - Stochastic trajectory matching (warmup → decays to 0)
             L_traj = 1/3 * (nMSE(h_t, traj_t) + nMSE(h_tp, traj_tp) + nMSE(out_t, traj_tpp))

        Returns (total, loss_pair, loss_interp, loss_prior, loss_conj,
                 loss_desc, loss_traj, w_traj).
        """
        loss_pair   = self._normalized_mse(output.pairwise_grad,
                                           output.pairwise_stopgrad)
        loss_interp = 0.5 * (
            self._normalized_mse(output.anchor_at_t,  output.x0) +
            self._normalized_mse(output.anchor_at_tp, output.x0)
        )
        loss_prior  = output.prior_reg

        # --- Term 4: gradient conjugacy ---
        g0 = output.grad_J_t.flatten(1)    # (B, D)
        g1 = output.grad_J_tp.flatten(1)   # (B, D)
        g2 = output.grad_J_tpp.flatten(1)  # (B, D)

        dg = g1 - g0                                       # (B, D)
        dot_conj = (g2 * dg).sum(dim=1)                    # (B,)
        norm_g2  = g2.norm(dim=1).clamp(min=1e-8)          # (B,)
        norm_dg  = dg.norm(dim=1).clamp(min=1e-8)          # (B,)
        cos_conj = dot_conj / (norm_g2 * norm_dg)          # (B,) normalised
        loss_conj = (cos_conj ** 2).mean()                  # scalar

        # --- Term 5: descent constraint  max(0, J(x_{t+1}) - J(x_t)) ---
        loss_desc = (
            torch.clamp(output.J_tp  - output.J_t,  min=0.0).mean()
            + torch.clamp(output.J_tpp - output.J_tp, min=0.0).mean()
        )

        # --- Term 6: stochastic trajectory matching (warmup) ---
        # Use the same student iterates as the pairwise loss path:
        #   composed_t  = g_student(x_T, T, t)    at time t
        #   composed_tp = g_student(x_T, T, t')   at time t'
        #   out_t       = g_student(h_t, t, t'')   at time t''
        loss_traj = (1.0 / 3.0) * (
            self._normalized_mse(output.composed_t,   output.traj_t)
            + self._normalized_mse(output.composed_tp,  output.traj_tp)
            + self._normalized_mse(output.out_t,        output.traj_tpp)
        )

        # Warmup weight for trajectory loss
        w_traj = self.warmup_weight(current_epoch)

        total = (
            self.lambda_pair   * loss_pair
            + self.lambda_interp * loss_interp
            + self.lambda_prior  * loss_prior
            #+ self.lambda_conj   * (loss_conj + loss_desc)
            + self.lambda_traj   * w_traj * loss_traj
        )
        return total, loss_pair, loss_interp, loss_prior, loss_conj, loss_desc, loss_traj, w_traj


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
        Karras schedule exponent (same value used during training, default 7.0).
    schedule : class
        LinearSchedule (default) or TrigSchedule.
    """

    def __init__(
        self,
        sigma_noise: float = 1.0,
        rho: float = 7.0,
        schedule=None,
    ) -> None:
        self.sigma_noise = sigma_noise
        self.rho = rho
        self.schedule = schedule if schedule is not None else DEFAULT_SCHEDULE

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

        # Karras times in [0,1], ascending — then reversed for sampling (T→0)
        times = torch.flip(
            karras_schedule(max(nsteps, 3), rho=self.rho, device=device, as_time=True),
            dims=[0],
        )  # (nsteps,) descending: times[0]=1.0, times[-1]≈0

        # x_T = sigma_noise * z
        x = noise * self.sigma_noise
        all_xs = [x]

        step_iter = range(len(times) - 1)
        if verbose:
            step_iter = tqdm(step_iter, desc="Variational sampling")

        for i in step_iter:
            t_cur  = torch.full((B,), times[i].item(),     dtype=dtype, device=device)
            t_prev = torch.full((B,), times[i + 1].item(), dtype=dtype, device=device)

            with torch.no_grad():
                x, _ = model_variational_forward_wrapper(
                    model, obs_cost, prior_cost, lambda_reg,
                    x, y, t_cur, t_prev, schedule=self.schedule, **kwargs
                )

            if clip_denoised:
                x = x.clamp(-1.0, 1.0)
            all_xs.append(x)

        return x, torch.stack(all_xs, dim=0)
