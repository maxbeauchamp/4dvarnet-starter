from .utils import *

# Classic consistency with observation conditioning

def model_obs_cond_forward_wrapper(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    mask_y: Tensor,    
    sigma: Tensor,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
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
    c_skip = skip_scaling(sigma, sigma_data, sigma_min)
    c_out = output_scaling(sigma, sigma_data, sigma_min)

    # Pad dimensions as broadcasting will not work
    c_skip = pad_dims_like(c_skip, x)
    c_out = pad_dims_like(c_out, x)

    return c_skip * x + c_out * model(x, y, mask_y, sigma, **kwargs)


class ConsistencyTrainingObsCond:
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
        sigma_data: float = 0.5,
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
        mask_y: Tensor,
        current_training_step: int,
        total_training_steps: int,
        **kwargs: Any,
    ) -> ConsistencyTrainingOutput:
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
        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, x.device
        )
        noise = torch.randn_like(x)

        timesteps = torch.randint(0, num_timesteps - 1, (x.shape[0],), device=x.device)

        current_sigmas = sigmas[timesteps]
        next_sigmas = sigmas[timesteps + 1]

        next_noisy_x = x + pad_dims_like(next_sigmas, x) * noise
        next_x = model_obs_cond_forward_wrapper(
            student_model,
            next_noisy_x,
            y,
            mask_y,
            next_sigmas,
            self.sigma_data,
            self.sigma_min,
            **kwargs,
        )

        with torch.no_grad():
            current_noisy_x = x + pad_dims_like(current_sigmas, x) * noise
            current_x = model_obs_cond_forward_wrapper(
                teacher_model,
                current_noisy_x,
                y,
                mask_y,
                current_sigmas,
                self.sigma_data,
                self.sigma_min,
                **kwargs,
            )

        return ConsistencyTrainingOutput(next_x, current_x, num_timesteps, sigmas)
    
class ConsistencySamplingAndEditingObsCond:
    """Implements the Consistency Sampling and Zero-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_data: float = 0.5) -> None:
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

    def __call__(
        self,
        model: nn.Module,
        noise: Tensor,
        y: Tensor,
        mask_y: Tensor,
        sigmas: Iterable[Union[Tensor, float]],
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Runs the sampling/zero-shot editing loop.

        With the default parameters the function performs consistency sampling.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        noise : Tensor
            Reference sample noise.
        y : Tensor
            Obs.
        mask_y : Tensor
            Mask obs.
        sigmas : Iterable[Union[Tensor, float]]
            Decreasing standard deviations of the noise.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Edited/sampled sample.
        """

        sigma = torch.full((noise.shape[0],), sigmas[0], dtype=noise.dtype, device=noise.device)
        x = noise
        x = model_obs_cond_forward_wrapper(
            model, x, y, mask_y, sigma, self.sigma_data, self.sigma_min, **kwargs
        )
        if clip_denoised:
            x = x.clamp(min=-1.0, max=1.0)

        # Progressively denoise the sample and skip the first step as it has already
        # been run
        pbar = tqdm(sigmas[1:], disable=(not verbose))
        for sigma in pbar:
            pbar.set_description(f"sampling (σ={sigma:.4f})")
            sigma = torch.full((noise.shape[0],), sigma, dtype=x.dtype, device=x.device)
            x = x + pad_dims_like(
                (sigma**2 - self.sigma_min**2) ** 0.5, x
            ) * torch.randn_like(x)
            x = model_obs_cond_forward_wrapper(
                model, x,  y, mask_y, sigma, self.sigma_data, self.sigma_min, **kwargs
            )
            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)

        return x