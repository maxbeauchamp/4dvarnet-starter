"""ODE-teacher targets for pairwise consistency training (ICLR rebuttal fix, reviewers XETn W1-W2, qyTj W1).

Problem with the submitted training (`ConsistencyTrainingFewSteps_TimeEmbedding`, pairwise=True): student and
teacher inputs are built from ONE noise draw, x_s = x0 + s*eps. Those states lie on the forward interpolant,
not on a probability-flow ODE trajectory, and the L2 optimum is a linear regression that shrinks every
one-step output towards E[x0 | y] (see consistency/rebuttal/toy_gaussian_targets.py).

Fix (CTM-style distillation, Kim et al. 2024, with a local anchor):
    n ~ U{1..N-1}, k ~ U{0..n-1} (k = n-1 with probability p_local)
    x_n      = x0 + s_n * eps                                   (a sample of the marginal at s_n)
    xhat     = Phi_teacher(x_n; s_n -> s_{n-1})                 (one Heun step of the trained FM, no grad)
    student  = g_theta(x_n; s_n -> s_k)
    target   = xhat                       if k = n-1            (local anchor: the student's short jump must
                                                                 match the teacher ODE, i.e. its tangent)
             = g_ema(xhat; s_{n-1} -> s_k) otherwise            (pairwise consistency along the teacher path)
At the optimum g(.; s -> s') is the PF-ODE flow map of the teacher between any two levels, so composition
holds (semigroup) and one-step samples keep the posterior spread.

The FM teacher uses the rectified-flow interpolant z_t = t*x1 + (1-t)*eps (t = 1 data, t = 0 noise). The VE state
x_s = x0 + s*eps maps to it exactly by t = 1 / (1 + s), z_t = t * x_s, so the teacher step is taken in t and
mapped back. The returned object is the same dataclass as the original training, so the notebooks' loss
(`F.mse_loss(output.predicted_next_from_intermediate, output.target_next_from_current)`) is unchanged.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn

from .consistency_models_CM import (
    ConsistencyTrainingFewSteps_TimeEmbedding,
    compute_sigma,
    model_few_steps_time_embedding_forward_wrapper,
)
from .utils import ConsistencyTrainingOutputFewSteps, karras_schedule, pad_dims_like


class FMTeacherVE:
    """Probability-flow step of a trained rectified-flow model, expressed on VE noise levels.

    velocity(z, y, t) -> dz/dt, with z_t = t*x_data + (1-t)*eps. `velocity` defaults to `fm_model(z, y, t)`
    (UNetFM signature); pass a callable for other conditionings (e.g. DynFM).
    """

    def __init__(self, fm_model: nn.Module, n_sub: int = 1,
                 velocity: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None) -> None:
        self.fm = fm_model.eval()
        for p in self.fm.parameters():
            p.requires_grad_(False)
        self.n_sub = n_sub
        self.velocity = velocity or (lambda z, y, t: self.fm(z, y, t))

    @staticmethod
    def t_of_sigma(sigma: Tensor) -> Tensor:
        return 1.0 / (1.0 + sigma)

    @torch.no_grad()
    def step(self, x: Tensor, y: Tensor, sigma_from: Tensor, sigma_to: Tensor) -> Tensor:
        """Heun integration of the teacher ODE from VE level sigma_from to sigma_to (per sample, (B,))."""
        dtype = x.dtype
        x, y = x.float(), y.float()
        t0, t1 = self.t_of_sigma(sigma_from.float()), self.t_of_sigma(sigma_to.float())
        z = pad_dims_like(t0, x) * x
        with torch.autocast(device_type=x.device.type, enabled=False):
            for i in range(self.n_sub):
                ta = t0 + (t1 - t0) * (i / self.n_sub)
                tb = t0 + (t1 - t0) * ((i + 1) / self.n_sub)
                dt = pad_dims_like(tb - ta, x)
                v0 = self.velocity(z, y, ta)
                z_e = z + dt * v0
                v1 = self.velocity(z_e, y, tb)
                z = z + dt * 0.5 * (v0 + v1)
        return (z / pad_dims_like(t1, x)).to(dtype)


class TeacherPairwiseConsistencyTraining(ConsistencyTrainingFewSteps_TimeEmbedding):
    """Drop-in replacement of ConsistencyTrainingFewSteps_TimeEmbedding (pairwise) with ODE-teacher targets.

    Parameters (in addition to the parent's): ode_teacher (FMTeacherVE), num_timesteps (fixed Karras grid size N),
    p_local (probability of the local-anchor pair k = n-1).
    """

    def __init__(self, ode_teacher: FMTeacherVE, num_timesteps: int = 18, p_local: float = 0.25,
                 sigma_min: float = 0.002, sigma_max: float = 10.0, rho: float = 7.0, sigma_data: float = 1,
                 **_: Any) -> None:
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data,
                         initial_timesteps=num_timesteps, final_timesteps=num_timesteps, pairwise=True)
        self.ode_teacher = ode_teacher
        self.num_timesteps = num_timesteps
        self.p_local = p_local

    def _g(self, model: nn.Module, x: Tensor, y: Tensor, t1: Tensor, t2: Tensor, **kwargs: Any) -> Tensor:
        return model_few_steps_time_embedding_forward_wrapper(
            model, x, y, t1, t2, self.sigma_data, self.sigma_min, self.sigma_max, pairwise=True, **kwargs)

    def __call__(self, student_model: nn.Module, teacher_model: nn.Module, x: Tensor, y: Tensor,
                 current_training_step: int = 0, total_training_steps: int = 1,
                 **kwargs: Any) -> ConsistencyTrainingOutputFewSteps:
        N, B, dev = self.num_timesteps, x.shape[0], x.device
        steps = karras_schedule(N, self.sigma_min, self.sigma_max, self.rho, dev, as_time=True).to(x.dtype)
        x = torch.nan_to_num(x, nan=0.0)                 # targets with gaps: masked by the caller's loss if needed
        n = torch.randint(1, N, (B,), device=dev)
        k = (torch.rand(B, device=dev) * n).long().clamp(max=N - 1)          # uniform in {0..n-1}
        local = torch.rand(B, device=dev) < self.p_local
        k = torch.where(local, n - 1, k)

        t_n, t_m, t_k = steps[n], steps[n - 1], steps[k]
        s_n = compute_sigma(t_n, self.sigma_min, self.sigma_max, self.rho)
        s_m = compute_sigma(t_m, self.sigma_min, self.sigma_max, self.rho)
        eps = torch.randn_like(x)
        x_n = x + pad_dims_like(s_n, x) * eps

        xhat = self.ode_teacher.step(x_n, y, s_n, s_m)                   # on the teacher's PF-ODE path
        student = self._g(student_model, x_n, y, t_n, t_k, **kwargs)
        with torch.no_grad():
            far = self._g(teacher_model, xhat, y, t_m, t_k, **kwargs)
            is_local = pad_dims_like((k == n - 1).to(x.dtype), x)
            target = is_local * xhat + (1 - is_local) * far

        sigma_k = compute_sigma(t_k, self.sigma_min, self.sigma_max, self.rho)
        diag = dict(frac_local=(k == n - 1).float().mean().detach(),
                    teacher_step_absmax=xhat.detach().abs().max(),
                    student_out_absmax=student.detach().abs().max())
        return ConsistencyTrainingOutputFewSteps(student, target.detach(), N, steps, diag=diag, noise=eps.detach(),
                                                 sigma_tgt=sigma_k.detach(), student_t1=t_n.detach(),
                                                 target_times=t_k.detach())
