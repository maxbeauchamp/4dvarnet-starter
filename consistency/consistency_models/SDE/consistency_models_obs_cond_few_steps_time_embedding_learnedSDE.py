"""
Fully Learned-SDE variant of ConsistencyTrainingFewSteps_TimeEmbedding.

sigma_phi(t) is learned end-to-end and plays BOTH roles:

  (1) Forward process (bruitage) :
        x_noisy = x + sigma_phi(t) * noise
      This replaces the fixed Karras schedule compute_sigma(t).
      The Karras time-grid is kept only as a curriculum for the number of
      discretisation steps (num_timesteps), but the actual noise levels are
      entirely determined by sigma_phi.

  (2) Reverse-process uncertainty (loss) :
        L_SDE^iso = E[ ||x_target - mu_phi||^2 / (2*sigma_phi^2)
                       + (d/2) * log(sigma_phi^2) ]
      where mu_phi is the consistency prediction.

Because sigma_phi is also used in skip_scaling / output_scaling, the
boundary condition  f(x, sigma_min) = x  is maintained: when
sigma_phi(t) -> sigma_min the model learns to return the input unchanged.

References
----------
Song et al., "Consistency Models", ICML 2023.
"""

from .utils import *


# ─── Consistency model wrapper using learned sigma ────────────────────────────

def model_learned_sde_forward_wrapper(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    t1: Tensor,
    t2: Tensor,
    sigma1: Tensor,      # sigma_phi(t1), shape (B,)  — learned
    sigma_data: float = 1.0,
    sigma_min: float = 0.002,
    **kwargs: Any,
) -> Tensor:
    """Consistency model forward pass with skip/output scaling based on learned sigma.

    Parameters
    ----------
    sigma1 : Tensor, shape (B,)
        sigma_phi evaluated at t1 — comes from SigmaNet, NOT from Karras.
    """
    c_skip = skip_scaling(sigma1, sigma_data, sigma_min)    # (B,)
    c_out  = output_scaling(sigma1, sigma_data, sigma_min)  # (B,)
    c_skip = pad_dims_like(c_skip, x)                       # (B,1,1,1)
    c_out  = pad_dims_like(c_out,  x)
    return c_skip * x + c_out * model(x, y, t1, t2, **kwargs)


# ─── Learned sigma network ────────────────────────────────────────────────────

class SigmaNet(nn.Module):
    """Small MLP that maps a scalar time t ∈ [0,1] to a positive scalar sigma_phi(t).

    The output is constrained to [sigma_min, +inf) via softplus so that the
    learned variance never collapses to zero.

    Parameters
    ----------
    hidden_dim : int
        Width of the hidden layers.
    n_layers : int
        Number of hidden layers.
    sigma_min : float
        Minimum output value (soft floor via softplus shift).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        sigma_min: float = 0.002,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min

        layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        t : Tensor, shape (B,)
            Normalised time steps in [0, 1].

        Returns
        -------
        Tensor, shape (B,)
            Positive sigma_phi(t) >= sigma_min.
        """
        t_in = t.float().unsqueeze(-1)          # (B, 1)
        raw  = self.net(t_in).squeeze(-1)       # (B,)
        return self.sigma_min + nn.functional.softplus(raw)


# ─── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class ConsistencyTrainingOutputLearnedSDE:
    """Output of ConsistencyTrainingFewSteps_LearnedSDE.__call__.

    Attributes
    ----------
    predicted_next_from_intermediate : Tensor
        Student prediction mu_phi  (B, C, H, W).
    target_next_from_current : Tensor
        Teacher target              (B, C, H, W).
    sigma_phi : Tensor
        Learned noise level per sample, shape (B,).
    num_timesteps : int
        Number of Karras steps at this training step.
    sigmas : Tensor
        Full Karras schedule used for this step.
    sde_loss : Tensor
        Scalar NLL loss L_SDE^iso (for logging).
    consistency_loss : Tensor
        Scalar MSE consistency loss (for logging).
    """
    predicted_next_from_intermediate: Tensor
    target_next_from_current: Tensor
    sigma_phi: Tensor
    num_timesteps: int
    sigmas: Tensor
    sde_loss: Tensor
    consistency_loss: Tensor


# ─── Trainer ─────────────────────────────────────────────────────────────────

class ConsistencyTrainingFewSteps_LearnedSDE:
    """Consistency training with a learned isotropic SDE variance.

    The SDE parameters are NOT part of this object — they live in ``sigma_net``
    which is a regular ``nn.Module`` passed by the Lightning module so that its
    parameters are optimised by the same Adam optimiser.

    Parameters
    ----------
    sigma_min, sigma_max, rho, sigma_data : float
        Karras schedule hyper-parameters (used only for the Karras schedule
        and the consistency model skip/output scaling).
    initial_timesteps, final_timesteps : int
        Curriculum for the number of Karras steps.
    lambda_consistency : float
        Weight of the plain MSE consistency loss added to L_SDE^iso.
        Set to 0.0 to use only the NLL loss.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 1.0,
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
        lambda_consistency: float = 1.0,
    ) -> None:
        self.sigma_min         = sigma_min
        self.sigma_max         = sigma_max
        self.rho               = rho
        self.sigma_data        = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps   = final_timesteps
        self.lambda_consistency = lambda_consistency

    # ------------------------------------------------------------------
    def __call__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        sigma_net: SigmaNet,
        x: Tensor,
        y: Tensor,
        current_training_step: int,
        total_training_steps: int,
        **kwargs: Any,
    ) -> ConsistencyTrainingOutputLearnedSDE:
        """Run one consistency training step with learned SDE variance.

        Parameters
        ----------
        student_model : nn.Module
            Model being trained (UNet).
        teacher_model : nn.Module
            EMA copy of student (frozen during forward).
        sigma_net : SigmaNet
            Learned sigma_phi network (trainable).
        x : Tensor
            Clean data, shape (B, C, H, W).
        y : Tensor
            Sparse observations, shape (B, C, H, W).
        current_training_step : int
        total_training_steps : int
        """
        # ── Karras time-grid (curriculum only — noise levels from sigma_net) ──
        num_timesteps = timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )
        num_timesteps = max(num_timesteps, 3)

        # Normalised time-steps t ∈ [0, 1].  We use the Karras grid as a
        # curriculum for *how many* discretisation points to use, but the
        # actual noise level at each t is determined by sigma_net(t).
        steps = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho,
            x.device, as_time=True,
        )   # shape (num_timesteps,), values in [0, 1]

        noise = torch.randn_like(x)
        timesteps = torch.randint(0, num_timesteps - 2, (x.shape[0],), device=x.device)

        current_times      = steps[timesteps]       # t_n
        intermediate_times = steps[timesteps + 1]   # t_{n+1}
        next_times         = steps[timesteps + 2]   # t_{n+2}

        # ── Learned sigma_phi for both t_n and t_{n+1} ───────────────
        # sigma_net weights are always float32; cast inputs to float32 and
        # outputs back to x.dtype so that mixed-precision training (bf16/fp16)
        # does not cause dtype mismatches.
        sigma_phi_intermediate = sigma_net(intermediate_times.float()).to(x.dtype)  # (B,)
        sigma_phi_current      = sigma_net(current_times.float()).to(x.dtype)       # (B,)

        # ── Student forward (differentiable) ─────────────────────────
        # Forward process: x_noisy = x + sigma_phi(t) * eps   [LEARNED]
        intermediate_noisy_x = x + pad_dims_like(sigma_phi_intermediate, x) * noise

        mu_phi = model_learned_sde_forward_wrapper(
            student_model,
            intermediate_noisy_x, y,
            intermediate_times, next_times,
            sigma1=sigma_phi_intermediate,
            sigma_data=self.sigma_data,
            sigma_min=self.sigma_min,
            **kwargs,
        )   # shape (B, C, H, W)

        # ── Teacher forward (no grad) ─────────────────────────────────
        with torch.no_grad():
            current_noisy_x = x + pad_dims_like(sigma_phi_current, x) * noise

            # sigma_net is frozen during teacher forward (no_grad context)
            sigma_phi_current_t = sigma_net(current_times.float()).to(x.dtype)   # same values, detached

            target = model_learned_sde_forward_wrapper(
                teacher_model,
                current_noisy_x, y,
                current_times, next_times,
                sigma1=sigma_phi_current_t,
                sigma_data=self.sigma_data,
                sigma_min=self.sigma_min,
                **kwargs,
            ).detach()   # shape (B, C, H, W)

        # ── L_SDE^iso  ───────────────────────────────────────────────
        # The same sigma_phi(t_{n+1}) is used both to noise x and to
        # parameterise the Gaussian likelihood → fully consistent VE-SDE.
        diff_sq = (mu_phi - target) ** 2                        # (B, C, H, W)
        d       = diff_sq[0].numel()                            # C * H * W

        sigma_phi_sq = pad_dims_like(sigma_phi_intermediate ** 2, diff_sq)   # (B,1,1,1)
        nll_per_sample = (
            diff_sq / (2.0 * sigma_phi_sq)
        ).sum(dim=(1, 2, 3)) + (d / 2.0) * torch.log(sigma_phi_intermediate ** 2)

        sde_loss = nll_per_sample.mean()

        # ── Optional MSE consistency term ────────────────────────────
        consistency_loss = torch.nn.functional.mse_loss(mu_phi, target)
        total_loss = sde_loss + self.lambda_consistency * consistency_loss

        return ConsistencyTrainingOutputLearnedSDE(
            predicted_next_from_intermediate=mu_phi,
            target_next_from_current=target,
            sigma_phi=sigma_phi_intermediate.detach(),
            num_timesteps=num_timesteps,
            sigmas=steps,
            sde_loss=sde_loss,
            consistency_loss=consistency_loss,
        ), total_loss


# ─── Sampling ─────────────────────────────────────────────────────────────────

class ConsistencySamplingAndEditingFewSteps_LearnedSDE:
    """Consistency sampling using a fully learned VE-SDE (sigma_phi replaces Karras).

    The noise level at each time-step is determined by ``sigma_net`` instead of
    the fixed Karras schedule.  The time-grid itself is still drawn from
    ``karras_schedule`` (as a curriculum/ordering device), but the actual amount
    of noise added at each step is ``sigma_net(t)``.

    Parameters
    ----------
    sigma_min, sigma_max, rho, sigma_data : float
        Same Karras hyper-parameters used during training (for the time grid and
        skip/output scaling boundary condition).
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 1.0,
    ) -> None:
        self.sigma_min  = sigma_min
        self.sigma_max  = sigma_max
        self.rho        = rho
        self.sigma_data = sigma_data

    @torch.no_grad()
    def __call__(
        self,
        model: nn.Module,
        sigma_net: SigmaNet,
        noise: Tensor,
        y: Tensor,
        nsteps: int,
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """Run consistency sampling with learned SDE schedule.

        Parameters
        ----------
        model : nn.Module
            Trained consistency model (EMA UNet).
        sigma_net : SigmaNet
            Trained sigma_phi network (must be on the same device as ``noise``).
        noise : Tensor
            Standard Gaussian noise, shape (B, C, H, W).
        y : Tensor
            Sparse observations, shape (B, C, H, W).
        nsteps : int
            Number of denoising steps.
        clip_denoised : bool
            Whether to clamp outputs to [-1, 1].
        verbose : bool
            Whether to show a progress bar.

        Returns
        -------
        x : Tensor
            Final denoised sample, shape (B, C, H, W).
        all_xs : Tensor
            All intermediate denoised samples stacked along dim 0,
            shape (nsteps, B, C, H, W).
        """
        # Karras time grid — descending (noisiest first), values in [0, 1]
        times = torch.flip(
            karras_schedule(nsteps, self.sigma_min, self.sigma_max, self.rho,
                            noise.device, as_time=True),
            dims=[0],
        )   # shape (nsteps,)

        # ── Initial noisy sample using sigma_net(t_0) ─────────────────
        # sigma_net weights are always float32 → feed t in float32,
        # then cast the output to match x / noise dtype (e.g. bfloat16).
        t0 = torch.full((noise.shape[0],), times[0].item(),
                        dtype=torch.float32, device=noise.device)
        sigma_0 = sigma_net(t0).to(noise.dtype)             # (B,)
        x = noise * pad_dims_like(sigma_0, noise)           # scale noise

        all_xs = [x]

        iterator = range(nsteps - 1)
        if verbose:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="Sampling (learned SDE)")

        for i in iterator:
            time_current = torch.full(
                (x.shape[0],), times[i].item(), dtype=torch.float32, device=x.device
            )
            time_next = torch.full(
                (x.shape[0],), times[i + 1].item(), dtype=torch.float32, device=x.device
            )
            sigma_i = sigma_net(time_current).to(x.dtype)   # (B,)

            x = model_learned_sde_forward_wrapper(
                model, x, y,
                t1=time_current, t2=time_next,
                sigma1=sigma_i,
                sigma_data=self.sigma_data,
                sigma_min=self.sigma_min,
                **kwargs,
            )
            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)
            all_xs.append(x)

        return x, torch.stack(all_xs, dim=0)
