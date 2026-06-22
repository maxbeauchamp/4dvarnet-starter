from .utils import *

# Modified consistency model for few-step dynamical-system sampling.
#
# DESIGN (v8 — fixed student/teacher pairwise assignment):
#   Solver progress s = si/(nsteps-1)  ∈ [0, 1]  (0 = pure noise, 1 = clean)
#   Diffusion time  t = 1 - s                     (1 = pure noise, 0 = clean)
#
#   s ∈ [0, SPINUP_FRAC]  (t ∈ [spinup_boundary, 1]):  SPIN-UP
#       IC + σ_sp(t)·ε → IC (frame 0)
#       Pairwise SDE-based: c_skip = σ_sp(t')/σ_sp(t), c_out = 1 - c_skip
#       σ_sp remaps [sb, 1] → [σ_min, σ_max] via Karras ρ-space
#       Same mechanism as standard CM pairwise.
#
#   s ∈ [SPINUP_FRAC, 1]  (t ∈ [0, spinup_boundary]):  PHYSICAL
#       IC → frame 1 → … → frame C-1  (forward physical time)
#       Pairwise interpolant: c_skip = t'/t, c_out = 1 - t'/t
#       Matches the interpolant structure of the physical input.
#
#   v8 fix: Student takes the BIGGER step (current → next), teacher takes
#   the SMALLER step (intermediate → next). This matches CM pairwise where
#   the teacher provides a reliable target (small step) and the student
#   learns the harder mapping (big step). v7 had these roles inverted,
#   causing spinup variance explosion as N grew.


def compute_sigma_spinup(t, sigma_min, sigma_max, spinup_boundary, rho=7.0):
    """Map spinup times t ∈ [sb, 1] → σ ∈ [σ_min, σ_max] via Karras ρ-space.

    At t=sb: σ = σ_min ≈ 0  (clean IC)
    At t=1:  σ = σ_max       (pure noise)
    """
    # Compute in float32 for precision, cast back to input dtype
    t_f = t.float() if t.is_floating_point() else t
    tau = ((t_f - spinup_boundary) / (1.0 - spinup_boundary)).clamp(0, 1)
    rho_inv = 1.0 / rho
    sigma = sigma_min**rho_inv + tau * (sigma_max**rho_inv - sigma_min**rho_inv)
    result = sigma**rho
    return result.to(t.dtype) if t.is_floating_point() else result


def model_dynamical_systems_forward_wrapper(
    model, x, y, t1, t2,
    sigma_min=0.002, sigma_max=10.0,
    spinup_boundary=0.7,
    _diag=None,
    **kwargs,
):
    """Regime-adaptive pairwise preconditioning.

    Spinup  (t1 > sb): SDE-based       c_skip = σ_sp(t2)/σ_sp(t1)
    Physical (t1 ≤ sb): interpolant     c_skip = t2/t1
    """
    is_spinup = (t1 > spinup_boundary)
    is_sp = pad_dims_like(is_spinup.to(x.dtype), x)

    # Spinup: σ-based pairwise (same as CM)
    sigma1 = compute_sigma_spinup(t1, sigma_min, sigma_max, spinup_boundary)
    sigma2 = compute_sigma_spinup(t2, sigma_min, sigma_max, spinup_boundary)
    c_skip_sp = pad_dims_like((sigma2 / sigma1.clamp(min=1e-8)).to(x.dtype), x)
    c_out_sp  = 1.0 - c_skip_sp

    # Physical: t-based pairwise (interpolant)
    c_skip_ph = pad_dims_like((t2 / t1.clamp(min=1e-8)).to(x.dtype), x)
    c_out_ph  = 1.0 - c_skip_ph

    c_skip = is_sp * c_skip_sp + (1 - is_sp) * c_skip_ph
    c_out  = is_sp * c_out_sp  + (1 - is_sp) * c_out_ph

    mdtype    = next(model.parameters()).dtype
    model_out = model(x.to(mdtype), y.to(mdtype), t1, t2, **kwargs)
    result    = (c_skip * x + c_out * model_out).to(x.dtype)

    if _diag is not None:
        cs_flat = c_skip.flatten(1).mean(1)
        co_flat = c_out.flatten(1).mean(1)
        _diag.update({
            "c_skip_min": cs_flat.min().item(),
            "c_skip_max": cs_flat.max().item(),
            "c_out_min":  co_flat.min().item(),
            "c_out_max":  co_flat.max().item(),
            "model_out_std": model_out.std().item(),
            "x_std":         x.std().item(),
            "result_std":    result.std().item(),
            "has_nan":       bool(torch.isnan(result).any()),
        })

    return result


def _make_regime_input(x, times, physical_steps, phys_step_size, physical_lag,
                       sigma_min, sigma_max, noise, anchor_mode=False):
    """Build model input for a batch, routing spin-up vs physical regime.

    Spin-up:  IC + σ_sp(t)·ε   (standard SDE noising of IC)
    Physical: interpolation between adjacent GT frame anchors
    """
    B, C, H, W = x.shape

    # Spin-up (t > physical_lag): IC + σ_spinup(t) · ε
    IC = x[:, 0:1, :, :]
    sigma_sp = compute_sigma_spinup(times, sigma_min, sigma_max, physical_lag).to(x.dtype)
    spinup_noisy = IC + sigma_sp.view(B, 1, 1, 1) * noise

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

    phys_out     = x_prev if anchor_mode else phys_interp
    spinup_mask  = (times > physical_lag).view(B, 1, 1, 1)
    return torch.where(spinup_mask, spinup_noisy, phys_out)


class ConsistencyTrainingDynamicalSystems:
    """Consistency training with regime-adaptive pairwise preconditioning."""

    def __init__(self, sigma_min=0.002, sigma_max=10.0, rho=7.0,
                 initial_timesteps=2, final_timesteps=150, spinup_boundary=0.7):
        self.sigma_min         = sigma_min
        self.sigma_max         = sigma_max
        self.rho               = rho
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

        # Diagnostics
        diag_student = {}
        diag_teacher = {}
        is_spinup = (current_times > physical_lag)
        n_spinup  = int(is_spinup.sum().item())
        n_phys    = x.shape[0] - n_spinup

        # Student — bigger step (current → next), matching CM pairwise design.
        # The student takes the harder step (noisiest input, biggest σ gap).
        current_noisy_x = _make_regime_input(
            x, current_times, physical_steps, phys_step_size,
            physical_lag, self.sigma_min, self.sigma_max, noise)
        student_pred = model_dynamical_systems_forward_wrapper(
            student_model, current_noisy_x, y,
            current_times, next_times,
            self.sigma_min, self.sigma_max,
            spinup_boundary=physical_lag,
            _diag=diag_student, **kwargs)

        # Teacher — smaller step (intermediate → next), reliable target.
        with torch.no_grad():
            gt_next = _make_regime_input(
                x, next_times, physical_steps, phys_step_size,
                physical_lag, self.sigma_min, self.sigma_max, noise,
                anchor_mode=True)

            intermediate_noisy_x = _make_regime_input(
                x, intermediate_times, physical_steps, phys_step_size,
                physical_lag, self.sigma_min, self.sigma_max, noise)

            with torch.amp.autocast('cuda', enabled=False):
                teacher_pred = model_dynamical_systems_forward_wrapper(
                    teacher_model.float(), intermediate_noisy_x.float(), y.float(),
                    intermediate_times.float(), next_times.float(),
                    self.sigma_min, self.sigma_max,
                    spinup_boundary=physical_lag,
                    _diag=diag_teacher, **kwargs)

            B = x.shape[0]
            is_physical = (current_times <= physical_lag).view(B, 1, 1, 1)
            target = torch.where(is_physical, gt_next, teacher_pred)

        diag = {
            "n_spinup": n_spinup, "n_phys": n_phys,
            "t_curr_range": (current_times.min().item(), current_times.max().item()),
            "t_int_range":  (intermediate_times.min().item(), intermediate_times.max().item()),
            "t_next_range": (next_times.min().item(), next_times.max().item()),
            "student": diag_student, "teacher": diag_teacher,
            "input_std":  current_noisy_x.std().item(),
            "target_std": target.std().item(),
            "gt_next_std": gt_next.std().item(),
        }

        return ConsistencyTrainingOutputFewSteps(
            student_pred, target, num_timesteps, steps, diag=diag)


class ConsistencySamplingAndEditingDynamicalSystems:
    """Consistency sampling with regime-adaptive pairwise preconditioning."""

    def __init__(self, sigma_min=0.002, sigma_max=10.0, spinup_boundary=0.7,
                 stochastic=False):
        self.sigma_min       = sigma_min
        self.sigma_max       = sigma_max
        self.spinup_boundary = spinup_boundary
        self.stochastic      = stochastic

    def __call__(self, model, noise, y, nsteps, karras=False,
                 clip_denoised=False, verbose=False, **kwargs):
        """
        Returns (final_x, all_xs, phys_frame_indices).

        stochastic=True: re-injects noise during spinup steps (SDE sampling).
        """
        physical_lag = self.spinup_boundary

        physical_steps_sample = torch.linspace(0.0, physical_lag, y.shape[1])

        if not karras:
            times = torch.linspace(1.0, 1e-8, nsteps)
        else:
            times = torch.flip(karras_schedule(nsteps, as_time=True), dims=[0])

        # Start from pure noise at σ_max (same as CM inference)
        x = (noise * self.sigma_max).to(noise.dtype)
        all_xs = [x]
        B = noise.shape[0]

        for i in range(nsteps - 1):
            tc = torch.full((B,), times[i],     dtype=x.dtype, device=x.device)
            tn = torch.full((B,), times[i + 1], dtype=x.dtype, device=x.device)

            # Stochastic re-injection in spin-up only
            if self.stochastic and times[i] > physical_lag and i > 0:
                sigma_curr = compute_sigma_spinup(
                    torch.tensor(times[i]), self.sigma_min, self.sigma_max, physical_lag).to(x.dtype)
                sigma_next = compute_sigma_spinup(
                    torch.tensor(times[i + 1]), self.sigma_min, self.sigma_max, physical_lag).to(x.dtype)
                delta_sigma = (sigma_curr ** 2 - sigma_next ** 2).clamp(min=0.0).sqrt()
                x = x + delta_sigma * torch.randn_like(x)

            x = model_dynamical_systems_forward_wrapper(
                model, x, y, tc, tn,
                self.sigma_min, self.sigma_max,
                spinup_boundary=physical_lag, **kwargs)
            if clip_denoised:
                x = x.clamp(-1.0, 1.0)
            all_xs.append(x)

        times_cpu = times.cpu()
        phys_frame_indices = [
            max(1, torch.argmin(torch.abs(times_cpu - t_k)).item())
            for t_k in physical_steps_sample.tolist()
        ]
        phys_frame_indices = phys_frame_indices[::-1]

        return x, torch.stack(all_xs, dim=0), phys_frame_indices
