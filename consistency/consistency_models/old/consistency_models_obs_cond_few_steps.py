from .utils import *

# Modified consistency for few-steps sampling 

def model_few_steps_forward_wrapper(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    sigma1: Tensor,
    sigma2: Tensor,
    sigma_data: float = 1,
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
    c_skip = skip_scaling(sigma1, sigma_data, sigma_min)
    c_out = output_scaling(sigma1, sigma_data, sigma_min)

    # Pad dimensions as broadcasting will not work
    c_skip = pad_dims_like(c_skip, x)
    c_out = pad_dims_like(c_out, x)

    return c_skip * x + c_out * model(x, y,  sigma1, sigma2, **kwargs)

eps = 1e-8

def a_sigma(sigma, sigma_p):
    return (sigma_p / (sigma + eps)).clamp(0.0, 1.0)   # ensure stable and in [0,1]

def cin(sigma, sigma_data):
    return 1.0 / torch.sqrt(sigma**2 + sigma_data**2)

def model_few_steps_forward_wrapper_new(
    model: nn.Module,  
    x,
    y,
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    sigma_data: float = 1,
    sigma_min: float = 0.002,
    **kwargs,
    ) -> torch.Tensor:
        
    a = a_sigma(sigma1, sigma2).view(-1,1,1,1)
    b = (1.0 - a)
    x_in = cin(sigma1, sigma_data).view(-1,1,1,1) * x
    # optionally pass both sigma and sigma_p as conditioning to D
    x_hat = model(x_in, y, sigma1, sigma2)   # network predicts approx x0
    return a * x + b * x_hat

class ConsistencyTrainingFewSteps:
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
        
        num_timesteps = max(num_timesteps,3)

        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, x.device
        )
        noise = torch.randn_like(x)

        timesteps = torch.randint(0, num_timesteps - 2, (x.shape[0],), device=x.device)

        current_sigmas = sigmas[timesteps]
        intermediate_sigmas = sigmas[timesteps + 1]
        next_sigmas = sigmas[timesteps + 2]

        current_noisy_x = x + pad_dims_like(current_sigmas, x) * noise
        intermediate_from_current_x = model_few_steps_forward_wrapper(
                student_model,
                current_noisy_x,
                y,
                current_sigmas,
                intermediate_sigmas,
                self.sigma_data,
                self.sigma_min,
                **kwargs,
            )
        
        intermediate_noisy_x = x + pad_dims_like(intermediate_sigmas, x) * noise
        next_from_intermediate_x = model_few_steps_forward_wrapper(
            student_model,
            intermediate_noisy_x,
            y,
            intermediate_sigmas,
            next_sigmas,
            self.sigma_data,
            self.sigma_min,
            **kwargs,
        )
    
        with torch.no_grad():

            current_noisy_x2 = x + pad_dims_like(current_sigmas, x) * noise
            next_from_current_x = model_few_steps_forward_wrapper(
                teacher_model,
                current_noisy_x2,
                y,
                current_sigmas,
                next_sigmas,
                self.sigma_data,
                self.sigma_min,
                **kwargs,
            )

        return ConsistencyTrainingOutputFewSteps(next_from_intermediate_x,
                                                 intermediate_from_current_x,
                                                 next_from_current_x,
                                                 num_timesteps, sigmas)

class ConsistencySamplingAndEditingFewSteps:
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
        nsteps : int
            Number od denoising/gradient descent steps
        karras : bool, default=True
            Use karras scheduler
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
        if not karras:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num=nsteps)
        else:
            sigmas = torch.flip(karras_schedule(nsteps), dims=[0])
        noise = noise * sigmas[0] 
        x = noise
        all_xs = [noise]        
        for i in range(nsteps-1):
            sigma_current = torch.full((noise.shape[0],), sigmas[i], dtype=noise.dtype, device=noise.device)
            sigma_next = torch.full((noise.shape[0],), sigmas[i+1], dtype=noise.dtype, device=noise.device)
            x = model_few_steps_forward_wrapper(
                model, x, y, sigma_current, sigma_next, self.sigma_data, self.sigma_min, **kwargs
            )
            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)
            all_xs.append(x)

        return x, torch.stack(all_xs,dim=0)