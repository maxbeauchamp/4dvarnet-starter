from ..utils import *

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
    sigma_max: float = 80.0,
    **kwargs: Any,
) -> Tensor:
    """Wrapper for the model call to ensure that the residual connection and scaling
    for the residual and output values are applied.

    Parameters
    ----------
    model : nn.Module
        Model to call.
    x : Tensor
        Input to the model, e.g: the noisy samples.
    sigma : Tensor
        Standard deviation of the noise. Normally referred to as t.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    **kwargs : Any
        Extra arguments to be passed during the model call.

    Returns
    -------
    Tensor
        Scaled output from the model with the residual connection applied.
    """
    sigma1 = compute_sigma(t1,sigma_min,sigma_max)
    c_skip = skip_scaling(sigma1, sigma_data, sigma_min)
    c_out = output_scaling(sigma1, sigma_data, sigma_min)

    # Pad dimensions as broadcasting will not work
    c_skip = pad_dims_like(c_skip, x)
    c_out = pad_dims_like(c_out, x)

    return c_skip * x + c_out * model(x, y,  t1, t2, **kwargs)

class ConsistencyTrainingFewSteps_TimeEmbedding:
    """Implements the Consistency Training algorithm proposed in the paper.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_max : float, default=80.0
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
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 1,
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps

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
        
        num_timesteps = max(num_timesteps,3)

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
        next_from_intermediate_x = model_few_steps_time_embedding_forward_wrapper(
            student_model,
            intermediate_noisy_x,
            y,
            intermediate_times,
            next_times,
            self.sigma_data,
            self.sigma_min,
            **kwargs,
        )
    
        with torch.no_grad():

            current_noisy_x = x + pad_dims_like(compute_sigma(current_times,
                                                               self.sigma_min,
                                                               self.sigma_max), x) * noise
            next_from_current_x = model_few_steps_time_embedding_forward_wrapper(
                teacher_model,
                current_noisy_x,
                y,
                current_times,
                next_times,
                self.sigma_data,
                self.sigma_min,
                **kwargs,
            )

        return ConsistencyTrainingOutputFewSteps(next_from_intermediate_x,
                                                 next_from_current_x,
                                                 num_timesteps, steps)

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
                 sigma_max: float = 80., 
                 sigma_data: float = 1) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

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

            # ── Denoising step (ODE or SDE) ───────────────────────────
            x = model_few_steps_time_embedding_forward_wrapper(
                model, x, y, time_current, time_next, self.sigma_data, self.sigma_min, **kwargs
            )

            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)

            # ── SDE: re-noise to sigma level of the *next* step ───────
            # x̂ = f_θ(x_noisy) is our denoised estimate.
            # We then add noise scaled to σ(t_{i+1}) to get a fresh
            # noisy sample before the next denoising call, following
            # the stochastic consistency sampling of Song et al. 2023.
            if stochastic and i < nsteps - 2:
                sigma_next = compute_sigma(times[i + 1], self.sigma_min, self.sigma_max)
                x = x + sigma_next * torch.randn_like(x)

            all_xs.append(x)

        return x, torch.stack(all_xs, dim=0)