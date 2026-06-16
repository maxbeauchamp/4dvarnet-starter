from .utils import *

# Modified consistency model for few-step dynamical-system sampling.
#
# DESIGN (v6 — spin-up FIRST, physical SECOND):
#   Solver progress s = si/(nsteps-1)  ∈ [0, 1]  (0 = pure noise, 1 = clean)
#   Diffusion time  t = 1 - s                     (1 = pure noise, 0 = clean)
#
#   s ∈ [0, SPINUP_FRAC]  (t ∈ [spinup_boundary, 1]):  SPIN-UP
#       pure noise → IC (frame 0)
#
#   s ∈ [SPINUP_FRAC, 1]  (t ∈ [0, spinup_boundary]):  PHYSICAL
#       IC → frame 1 → … → frame C-1  (forward physical time)
#
#   With spinup_boundary=0.7 and NSTEPS=4C+1:
#       SPINUP_FRAC = 1 - 0.7 = 0.3 → IC at step ~C+1 (s ≈ 0.3) ✓


def compute_sigma(t, t_initial_cond, sigma_min, sigma_max, rho=7.0):
    rho_inv = 1.0 / rho
    sigma = sigma_min**rho_inv + torch.min(torch.tensor(1.), t / t_initial_cond) * (
        sigma_max**rho_inv - sigma_min**rho_inv)
    return sigma**rho


def skip_spinup(t, physical_lag=0.7, steepness=100):
    """Gate: 1 (Karras residual) in spin-up (t > physical_lag), 0 (raw model) in physical (t < physical_lag)."""
    return 1. / (1 + torch.exp(-steepness * (t - physical_lag)))


def model_dynamical_systems_forward_wrapper(
    model, x, y, t1, t2,
    sigma_data=1, sigma_min=0.002, sigma_max=80.0,
    spinup_boundary=0.7,
    **kwargs,
):
    physical_lag  = spinup_boundary
    sigma1        = compute_sigma(t1, physical_lag, sigma_min, sigma_max)
    c_skip        = pad_dims_like(skip_scaling(sigma1, sigma_data, sigma_min), x)
    c_out         = pad_dims_like(output_scaling(sigma1, sigma_data, sigma_min), x)
    model_out     = model(x, y, t1, t2, **kwargs)
    # Karras parametrization applied uniformly: at small t (late physical steps),
    # c_skip → 1 anchors output to input, preventing error accumulation across steps.
    return c_skip * x + c_out * model_out


def _make_regime_input(x, times, physical_steps, phys_step_size, physical_lag,
                       sigma_min, sigma_max, noise, anchor_mode=False):
    """Build model input for a batch, routing spin-up vs physical regime.

    physical_steps = linspace(0, physical_lag, C)  INCREASING
      physical_steps[k]  ↔  frame C-1-k
        k=0:   t=0             → frame C-1 (last, cleanest)
        k=C-1: t=physical_lag  → frame 0   (IC)
    """
    B, C, H, W = x.shape

    # Spin-up (t > physical_lag): IC + sigma * noise
    sigma        = compute_sigma(times, physical_lag, sigma_min, sigma_max)
    spinup_noisy = x[:, [0], :, :] + sigma.view(B, 1, 1, 1) * noise

    # Physical (t ≤ physical_lag): interpolation between adjacent frame anchors
    idx = torch.searchsorted(
        physical_steps.unsqueeze(0).expand(B, -1).contiguous(),
        times.unsqueeze(1)
    ).squeeze(1).clamp(1, C - 1)

    b_idx  = torch.arange(B, device=x.device)
    x_prev = x[b_idx, C - idx,     :, :].unsqueeze(1)   # lower-t: NEXT fwd-time frame
    x_curr = x[b_idx, C - 1 - idx, :, :].unsqueeze(1)   # upper-t: current (IC-side)

    # a=0 at upper bound (x_curr), a=1 at lower bound (x_prev)
    a = ((physical_steps[idx] - times) / phys_step_size).clamp(0., 1.).view(B, 1, 1, 1)
    phys_interp = a * x_prev + (1. - a) * x_curr

    # anchor_mode: return x_prev (next clean frame in fwd time) — prevents identity collapse
    phys_out     = x_prev if anchor_mode else phys_interp
    spinup_mask  = (times > physical_lag).view(B, 1, 1, 1)
    return torch.where(spinup_mask, spinup_noisy, phys_out)


class ConsistencyTrainingDynamicalSystems:
    """Consistency training for dynamical systems."""

    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0, sigma_data=1,
                 initial_timesteps=2, final_timesteps=150, spinup_boundary=0.7):
        self.sigma_min         = sigma_min
        self.sigma_max         = sigma_max
        self.rho               = rho
        self.sigma_data        = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps   = final_timesteps
        self.spinup_boundary   = spinup_boundary

    def __call__(self, student_model, teacher_model, x, y,
                 current_training_step, total_training_steps, **kwargs):

        num_timesteps = max(timesteps_schedule(
            current_training_step, total_training_steps,
            self.initial_timesteps, self.final_timesteps), 3)

        steps        = torch.linspace(1.0, 1e-8, num_timesteps, device=x.device)
        physical_lag = self.spinup_boundary
        physical_steps = torch.linspace(0.0, physical_lag, x.shape[1], device=x.device)
        phys_step_size = physical_lag / max(x.shape[1] - 1, 1)
        noise          = torch.randn_like(x[:, [0], :, :])

        # k_IC = last index where steps > physical_lag (spin-up zone)
        k_IC = int((steps > physical_lag).sum().item()) - 1

        valid_spinup   = list(range(0, max(0, k_IC - 1)))           # all t > physical_lag
        valid_physical = list(range(k_IC + 1, num_timesteps - 2))   # all t ≤ physical_lag
        valid_indices  = valid_spinup + valid_physical
        if not valid_indices:
            valid_indices = list(range(num_timesteps - 2))

        valid_indices_t = torch.tensor(valid_indices, device=x.device)
        rand_pos   = torch.randint(0, len(valid_indices), (x.shape[0],), device=x.device)
        timesteps  = valid_indices_t[rand_pos]

        current_times      = steps[timesteps]
        intermediate_times = steps[timesteps + 1]
        next_times         = steps[timesteps + 2]

        # Student
        intermediate_noisy_x = _make_regime_input(
            x, intermediate_times, physical_steps, phys_step_size,
            physical_lag, self.sigma_min, self.sigma_max, noise)
        next_from_intermediate_x = model_dynamical_systems_forward_wrapper(
            student_model, intermediate_noisy_x, y,
            intermediate_times, next_times,
            self.sigma_data, self.sigma_min, self.sigma_max,
            spinup_boundary=physical_lag, **kwargs)

        # Target
        with torch.no_grad():
            gt_next = _make_regime_input(
                x, next_times, physical_steps, phys_step_size,
                physical_lag, self.sigma_min, self.sigma_max, noise,
                anchor_mode=True)   # next clean frame in fwd time → no identity collapse

            current_noisy_x = _make_regime_input(
                x, current_times, physical_steps, phys_step_size,
                physical_lag, self.sigma_min, self.sigma_max, noise)
            teacher_next = model_dynamical_systems_forward_wrapper(
                teacher_model, current_noisy_x, y,
                current_times, next_times,
                self.sigma_data, self.sigma_min, self.sigma_max,
                spinup_boundary=physical_lag, **kwargs)

            B = x.shape[0]
            is_physical = (current_times <= physical_lag).view(B, 1, 1, 1)
            target = torch.where(is_physical, gt_next, teacher_next)

        return ConsistencyTrainingOutputFewSteps(
            next_from_intermediate_x, target, num_timesteps, steps)


class ConsistencySamplingAndEditingDynamicalSystems:
    """Consistency sampling for dynamical systems."""

    def __init__(self, sigma_min=0.002, sigma_max=80., sigma_data=1, spinup_boundary=0.7,
                 stochastic=False):
        self.sigma_min       = sigma_min
        self.sigma_max       = sigma_max
        self.sigma_data      = sigma_data
        self.spinup_boundary = spinup_boundary
        self.stochastic      = stochastic   # False = deterministic (pure consistency)
                                            # True  = réinjecte Δσ·ε à chaque step spin-up

    def __call__(self, model, noise, y, nsteps, karras=False,
                 clip_denoised=False, verbose=False, **kwargs):
        """
        Returns (final_x, all_xs, phys_frame_indices).
          all_xs[si]          = state after step si  (shape: nsteps × B × 1 × H × W)
          phys_frame_indices  = IC-first list of si indices per physical frame
            IC   at si ≈ round((1-spinup_boundary)*(nsteps-1))  [~30% through for sb=0.7]
            last at si = nsteps-1

        stochastic=True: avant chaque step spin-up (i>0), réinjecte
          x ← x + sqrt(σ(t_curr)² − σ(t_next)²) · ε
        Ceci maintient x proche de la distribution d'entraînement IC+σ(t)·ε
        → trajectoire spin-up progressive (comme DDPM).
        Uniquement applicable au spin-up (SDE connue) ; pas en régime physique.
        """
        physical_lag = self.spinup_boundary

        # physical_steps_sample[k] = physical_lag * k/(C-1)  ↔  frame C-1-k
        physical_steps_sample = torch.linspace(0.0, physical_lag, y.shape[1])

        if not karras:
            times = torch.linspace(1.0, 1e-8, nsteps)
        else:
            times = torch.flip(karras_schedule(nsteps, as_time=True), dims=[0])

        noise  = noise * compute_sigma(times[0], physical_lag, self.sigma_min, self.sigma_max)
        x      = noise
        all_xs = [noise]

        for i in range(nsteps - 1):
            tc = torch.full((noise.shape[0],), times[i],     dtype=noise.dtype, device=noise.device)
            tn = torch.full((noise.shape[0],), times[i + 1], dtype=noise.dtype, device=noise.device)

            # Stochastic re-injection in spin-up only (t > spinup_boundary)
            if self.stochastic and times[i] > physical_lag and i > 0:
                sigma_curr = compute_sigma(times[i],     physical_lag, self.sigma_min, self.sigma_max)
                sigma_next = compute_sigma(times[i + 1], physical_lag, self.sigma_min, self.sigma_max)
                delta_sigma = (sigma_curr ** 2 - sigma_next ** 2).clamp(min=0.0).sqrt()
                x = x + delta_sigma * torch.randn_like(x)

            x  = model_dynamical_systems_forward_wrapper(
                model, x, y, tc, tn,
                self.sigma_data, self.sigma_min, self.sigma_max,
                spinup_boundary=physical_lag, **kwargs)
            if clip_denoised:
                x = x.clamp(-1.0, 1.0)
            all_xs.append(x)

        # physical_steps_sample: [0, ..., physical_lag] → frame C-1 first, IC last
        # After reversal → IC-first
        times_cpu = times.cpu()
        phys_frame_indices = [
            max(1, torch.argmin(torch.abs(times_cpu - t_k)).item())
            for t_k in physical_steps_sample.tolist()
        ]
        phys_frame_indices = phys_frame_indices[::-1]

        return x, torch.stack(all_xs, dim=0), phys_frame_indices
