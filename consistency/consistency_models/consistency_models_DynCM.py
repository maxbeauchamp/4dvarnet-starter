from .utils import *

# Modified consistency for few-steps sampling 

def compute_sigma(t: Tensor,
                  t_initial_cond: float,
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
    sigma = sigma_min**rho_inv + torch.min(torch.tensor(1.),
                                           t/t_initial_cond) * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigma = sigma**rho

    return sigma

def skip_spinup(
    t: Tensor, 
    physical_lag: float = 0.2,
    steepness: float = 100  # steepness of the logistic function
) -> Tensor:
    """
    # Logistic function from 1 to 0
    """
    return 1. / (1 + torch.exp(steepness * (t - physical_lag)))


def model_dynamical_systems_forward_wrapper(
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
    physical_lag = 1./y.shape[1]
     
    sigma1 = compute_sigma(t1,physical_lag,sigma_min,sigma_max)
    c_skip = skip_scaling(sigma1,sigma_data, sigma_min)
    c_out = output_scaling(sigma1, sigma_data, sigma_min)

    # Pad dimensions as broadcasting will not work
    c_skip = pad_dims_like(c_skip, x)
    c_out = pad_dims_like(c_out, x)

    c_skip_spinup = skip_spinup(t1,physical_lag)
    c_skip_spinup = pad_dims_like(c_skip_spinup, x)
    
    return c_skip_spinup*(c_skip * x + c_out * model(x, y,  t1, t2, **kwargs)) + (1.-c_skip_spinup) * model(x, y,  t1, t2, **kwargs)

class ConsistencyTrainingDynamicalSystems:
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

        # Use linspace to match the sampler (which also uses linspace), so that the
        # distribution of training pairs (t_cur, t_int, t_next) matches the transitions
        # seen at inference time.  The old karras_schedule was biased toward t≈0 (spin-up),
        # leaving the physical regime (t>t_IC) severely under-represented during training.
        steps = torch.linspace(1.0, 1e-8, num_timesteps, device=x.device)
        physical_steps = torch.linspace(0,1,x.shape[1],device=x.device)
        physical_lag = 1./x.shape[1]
     
        noise = torch.randn_like(x[:,[0],:,:])

        timesteps = torch.randint(0, num_timesteps - 2, (x.shape[0],), device=x.device)

        current_times = steps[timesteps]
        intermediate_times = steps[timesteps + 1]
        next_times = steps[timesteps + 2]

        # ── Student: f(x_t_int, t_int → t_next) ──────────────────────────
        if intermediate_times<=physical_steps[1]:
            intermediate_noisy_x = x[:,[0],:,:] + pad_dims_like(compute_sigma(intermediate_times,
                                                                physical_lag,
                                                                self.sigma_min,
                                                                self.sigma_max), x[:,[0],:,:]) * noise
        else:
            idx = torch.searchsorted(physical_steps, intermediate_times).item()
            a = (physical_steps[idx]-intermediate_times)/physical_lag
            intermediate_noisy_x = a*x[:,[idx-1],:,:] + (1-a)*x[:,[idx],:,:]
        next_from_intermediate_x = model_dynamical_systems_forward_wrapper(
            student_model,
            intermediate_noisy_x,
            y,
            intermediate_times,
            next_times,
            self.sigma_data,
            self.sigma_min,
            **kwargs,
        )
    
        # ── Teacher: f_ema(x_t_cur, t_cur → t_next) ──────────────────────
        with torch.no_grad():
            idx = torch.searchsorted(physical_steps, current_times).item()
            if current_times<=physical_steps[1]:
                # spin-up regime: add Karras noise to x_0
                current_noisy_x = x[:,[0],:,:] + pad_dims_like(compute_sigma(current_times,
                                                                              physical_lag,
                                                                              self.sigma_min,
                                                                              self.sigma_max), x[:,[0],:,:]) * noise
            else:
                # physical regime: linear interpolation at current_times (fix: was using next_times)
                a = (physical_steps[idx]-current_times)/physical_lag
                current_noisy_x = a*x[:,[idx-1],:,:] + (1-a)*x[:,[idx],:,:]
            next_from_current_x = model_dynamical_systems_forward_wrapper(
                teacher_model,
                current_noisy_x,
                y,
                current_times,
                next_times,
                self.sigma_data,
                self.sigma_min,
                **kwargs,
            )

        # Return: predicted = student(t_int→t_next), target = teacher(t_cur→t_next)
        # intermediate_from_current_x slot is unused — pass None as placeholder
        return ConsistencyTrainingOutputFewSteps(next_from_intermediate_x,
                                                 None,
                                                 next_from_current_x,
                                                 num_timesteps, steps)

class ConsistencySamplingAndEditingDynamicalSystems:
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
        karras: bool = False,
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
            times = torch.linspace(torch.tensor(1.0),
                                   torch.tensor(1e-8), 
                                   nsteps)
        else:
            times = torch.flip(karras_schedule(nsteps, as_time=True), dims=[0])
        noise = noise * compute_sigma(times[0], 1./y.shape[1], self.sigma_min, self.sigma_max)
        x = noise
        all_xs = [noise]
        for i in range(nsteps-1):
            time_current = torch.full((noise.shape[0],), times[i], dtype=noise.dtype, device=noise.device)
            time_next = torch.full((noise.shape[0],), times[i+1], dtype=noise.dtype, device=noise.device)
            x = model_dynamical_systems_forward_wrapper(
                model, x, y, time_current, time_next, self.sigma_data, self.sigma_min, **kwargs
            )
            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)
            all_xs.append(x)

        return x, torch.stack(all_xs,dim=0)