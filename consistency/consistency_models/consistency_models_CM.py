from .utils import *

# Modified consistency for few-steps sampling 

def compute_sigma(t: Tensor,
                  sigma_min: float,
                  sigma_max: float,
                  rho: float=7.0) -> Tensor:

    """Compute the noise level (sigma) from the time embedding t.

    Parameters
    ----------
    t : Tensor
        Time embedding.

    Returns
    -------
    Tensor
        Computed noise level (sigma).
    """
    rho_inv = 1.0 / rho
    sigma = sigma_min**rho_inv + t * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigma = sigma**rho

    return sigma


def model_few_steps_time_embedding_forward_wrapper(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    t1: Tensor,
    t2: Tensor,
    sigma_data: float = 1,
    sigma_min: float = 0.002,
    sigma_max: float = 10.0,
    pairwise: bool = True,
    **kwargs: Any,
) -> Tensor:
    """Wrapper for the model call with residual connection and scaling.

    Parameters
    ----------
    model : nn.Module
        Model to call.
    x : Tensor
        Input to the model, e.g: the noisy samples.
    t1 : Tensor
        Current time embedding.
    t2 : Tensor
        Target time embedding.
    sigma_data : float, default=1
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_max : float, default=10.0
        Maximum standard deviation of the noise.
    pairwise : bool, default=True
        If True, use pairwise preconditioning: output targets x(σ') = x_clean + σ'·ε
        instead of x_clean.  c_skip = σ'/σ, c_out = 1 - σ'/σ.
    **kwargs : Any
        Extra arguments to be passed during the model call.

    Returns
    -------
    Tensor
        Scaled output from the model with the residual connection applied.
    """
    sigma1 = compute_sigma(t1, sigma_min, sigma_max)

    if pairwise:
        sigma2 = compute_sigma(t2, sigma_min, sigma_max)
        c_skip = sigma2 / sigma1.clamp(min=1e-8)
        c_out = 1.0 - c_skip
    else:
        c_skip = skip_scaling(sigma1, sigma_data, sigma_min)
        c_out = output_scaling(sigma1, sigma_data, sigma_min)

    c_skip = pad_dims_like(c_skip, x)
    c_out = pad_dims_like(c_out, x)

    return c_skip * x + c_out * model(x, y, t1, t2, **kwargs)

class ConsistencyTrainingFewSteps_TimeEmbedding:
    """Implements the Consistency Training algorithm proposed in the paper.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_max : float, default=10.0
        Maximum standard deviation of the noise.
    rho : float, default=7.0
        Schedule hyper-parameter.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    initial_timesteps : int, default=2
        Schedule timesteps at the start of training.
    final_timesteps : int, default=150
        Schedule timesteps at the end of training.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        rho: float = 7.0,
        sigma_data: float = 1,
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
        pairwise: bool = True,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
        self.pairwise = pairwise

    def __call__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        x: Tensor,
        y: Tensor,
        current_training_step: int,
        total_training_steps: int,
        **kwargs: Any,
    ) -> ConsistencyTrainingOutputFewSteps:
        """Runs one step of the consistency training algorithm.

        Parameters
        ----------
        student_model : nn.Module
            Model that is being trained.
        teacher_model : nn.Module
            An EMA of the student model.
        x : Tensor
            Clean data.
        current_training_step : int
            Current step in the training loop.
        total_training_steps : int
            Total number of steps in the training loop.
        **kwargs : Any
            Additional keyword arguments to be passed to the models.

        Returns
        -------
        ConsistencyTrainingOutput
            The predicted and target values for computing the loss as well as sigmas (noise levels).
        """
        num_timesteps = timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )

        # timesteps_schedule() has no built-in ceiling: if current_training_step
        # exceeds total_training_steps (e.g. a run trained for more actual steps
        # than the total_steps value used to pace the schedule), num_timesteps
        # keeps growing UNBOUNDED past final_timesteps instead of plateauing
        # there. This went unnoticed because GP/SSH_GF's tiny datasets never
        # accumulate that many steps within a normal run, but SIC's larger
        # dataset does -- observed as num_timesteps=28 with final_timesteps=17,
        # producing an over-fine discretization (tiny sigma gaps -> trivially
        # low consistency loss, blurry/washed-out samples, no genuine sharp
        # denoising). Clamp to [3, final_timesteps].
        num_timesteps = max(min(num_timesteps, self.final_timesteps), 3)

        steps = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, x.device, as_time=True
        )
        noise = torch.randn_like(x)

        timesteps = torch.randint(0, num_timesteps - 2, (x.shape[0],), device=x.device)

        current_times = steps[timesteps]
        intermediate_times = steps[timesteps + 1]
        next_times = steps[timesteps + 2]

        intermediate_noisy_x = x + pad_dims_like(compute_sigma(intermediate_times,
                                                              self.sigma_min,
                                                              self.sigma_max), x) * noise

        if self.pairwise:
            # Pairwise: both target current_times (lowest σ in triplet).
            # Student takes the bigger step (noisier input → more stable target from teacher)
            # Student: σ_next → σ_current          (c_skip = σ_curr/σ_next < 1)
            # Teacher: σ_intermediate → σ_current  (c_skip = σ_curr/σ_int, smaller step)
            target_times = current_times
            student_noisy_x = x + pad_dims_like(compute_sigma(next_times,
                                                              self.sigma_min,
                                                              self.sigma_max), x) * noise
            student_t1 = next_times
            teacher_noisy_x = intermediate_noisy_x
            teacher_t1 = intermediate_times
        else:
            # Non-pairwise: original CM — both student and teacher target next_times.
            target_times = next_times
            student_noisy_x = intermediate_noisy_x
            student_t1 = intermediate_times
            teacher_noisy_x = x + pad_dims_like(compute_sigma(current_times,
                                                              self.sigma_min,
                                                              self.sigma_max), x) * noise
            teacher_t1 = current_times

        # ── Student forward ──
        sigma_s = compute_sigma(student_t1, self.sigma_min, self.sigma_max)
        sigma_tgt = compute_sigma(target_times, self.sigma_min, self.sigma_max)
        if self.pairwise:
            c_skip_s = sigma_tgt / sigma_s.clamp(min=1e-8)
            c_out_s = 1.0 - c_skip_s
        else:
            c_skip_s = skip_scaling(sigma_s, self.sigma_data, self.sigma_min)
            c_out_s = output_scaling(sigma_s, self.sigma_data, self.sigma_min)

        raw_student = student_model(student_noisy_x, y, student_t1, target_times, **kwargs)
        c_skip_s_p = pad_dims_like(c_skip_s, student_noisy_x)
        c_out_s_p = pad_dims_like(c_out_s, student_noisy_x)
        next_from_intermediate_x = c_skip_s_p * student_noisy_x + c_out_s_p * raw_student

        # ── Teacher forward ──
        with torch.no_grad():
            sigma_t = compute_sigma(teacher_t1, self.sigma_min, self.sigma_max)
            if self.pairwise:
                c_skip_t = sigma_tgt / sigma_t.clamp(min=1e-8)
                c_out_t = 1.0 - c_skip_t
            else:
                c_skip_t = skip_scaling(sigma_t, self.sigma_data, self.sigma_min)
                c_out_t = output_scaling(sigma_t, self.sigma_data, self.sigma_min)

            if self.pairwise:
                with torch.amp.autocast('cuda', enabled=False):
                    raw_teacher = teacher_model.float()(
                        teacher_noisy_x.float(), y.float(),
                        teacher_t1.float(), target_times.float(), **kwargs,
                    )
                    c_skip_t_p = pad_dims_like(c_skip_t.float(), teacher_noisy_x)
                    c_out_t_p = pad_dims_like(c_out_t.float(), teacher_noisy_x)
                    next_from_current_x = c_skip_t_p * teacher_noisy_x.float() + c_out_t_p * raw_teacher
            else:
                raw_teacher = teacher_model(teacher_noisy_x, y, teacher_t1, target_times, **kwargs)
                c_skip_t_p = pad_dims_like(c_skip_t, teacher_noisy_x)
                c_out_t_p = pad_dims_like(c_out_t, teacher_noisy_x)
                next_from_current_x = c_skip_t_p * teacher_noisy_x + c_out_t_p * raw_teacher

        # ── Diagnostics ──
        diag = {
            'c_out_s': c_out_s.detach(), 'c_out_t': c_out_t.detach(),
            'c_skip_s': c_skip_s.detach(), 'c_skip_t': c_skip_t.detach(),
            'sigma_s': sigma_s.detach(), 'sigma_t': sigma_t.detach(),
            'sigma_tgt': sigma_tgt.detach(),
            'raw_student_absmax': raw_student.detach().abs().max(),
            'raw_teacher_absmax': raw_teacher.detach().abs().max(),
            'student_out_absmax': next_from_intermediate_x.detach().abs().max(),
            'teacher_out_absmax': next_from_current_x.detach().abs().max(),
            'raw_student_nan': raw_student.detach().isnan().any(),
            'raw_teacher_nan': raw_teacher.detach().isnan().any(),
            'student_input_absmax': student_noisy_x.detach().abs().max(),
            'teacher_input_absmax': teacher_noisy_x.detach().abs().max(),
        }

        return ConsistencyTrainingOutputFewSteps(next_from_intermediate_x,
                                                 next_from_current_x,
                                                 num_timesteps, steps,
                                                 diag=diag,
                                                 noise=noise.detach(),
                                                 sigma_tgt=sigma_tgt.detach(),
                                                 student_t1=student_t1.detach(),
                                                 target_times=target_times.detach(),
                                                 raw_student=raw_student)

class ConsistencySamplingAndEditingFewSteps_TimeEmbedding:
    """Implements the Consistency Sampling and Few-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """

    def __init__(self,
                 sigma_min: float = 0.002,
                 sigma_max: float = 10.0,
                 sigma_data: float = 1,
                 pairwise: bool = True) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.pairwise = pairwise

    def __call__(
        self,
        model: nn.Module,
        noise: Tensor,
        y: Tensor,
        nsteps: int,
        karras: bool = True,
        stochastic: bool = False,
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Runs the sampling/zero-shot editing loop.

        With the default parameters the function performs consistency sampling
        (ODE mode).  Setting ``stochastic=True`` enables SDE sampling: after
        each denoising step the prediction is re-noised to the next Karras
        level before the following denoising call, which increases sample
        diversity (Song et al. 2023, Algorithm 2).

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        noise : Tensor
            Reference sample noise, shape (B, C, H, W).
        y : Tensor
            Sparse observations, shape (B, C, H, W).
        nsteps : int
            Number of denoising steps.
        karras : bool, default=True
            Use Karras time schedule.
        stochastic : bool, default=False
            If True, use SDE sampling (re-noise between steps).
            If False, use ODE sampling (deterministic).
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1].
        verbose : bool, default=False
            Whether to display a progress bar.
        **kwargs : Any
            Additional keyword arguments passed to the model.

        Returns
        -------
        x : Tensor
            Final denoised sample, shape (B, C, H, W).
        all_xs : Tensor
            All intermediate samples stacked along dim 0.
        """
        if not karras:
            times = np.linspace(1.0, 1e-8, num=nsteps)
        else:
            times = torch.flip(karras_schedule(nsteps, as_time=True), dims=[0])

        noise = noise * compute_sigma(times[0], self.sigma_min, self.sigma_max)
        x = noise
        all_xs = [noise]

        for i in range(nsteps - 1):
            time_current = torch.full((x.shape[0],), times[i],     dtype=x.dtype, device=x.device)
            time_next    = torch.full((x.shape[0],), times[i + 1], dtype=x.dtype, device=x.device)

            # ── Denoising step ────────────────────────────────────────
            x = model_few_steps_time_embedding_forward_wrapper(
                model, x, y, time_current, time_next,
                self.sigma_data, self.sigma_min,
                pairwise=self.pairwise, **kwargs,
            )

            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)

            # ── SDE: re-noise to sigma level of the *next* step ───────
            # With pairwise preconditioning the output already sits at
            # noise level σ(t_{i+1}), so re-noising is skipped.
            if stochastic and not self.pairwise and i < nsteps - 2:
                sigma_next = compute_sigma(times[i + 1], self.sigma_min, self.sigma_max)
                x = x + sigma_next * torch.randn_like(x)

            all_xs.append(x)

        return x, torch.stack(all_xs, dim=0)