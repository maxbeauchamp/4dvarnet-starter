"""
generators.py
=============
One conditional flow-matching generator G_l of the cascade (formulation
section 4): Swin U-Net velocity network (FMSwinWrapper) + LogScaleModel, the
masked NLL velocity loss of the former CROSCIM FM model (same censoring
rule), and Heun sampling with FlowMatchingSampler.

For the detail levels (dX_M, dX_H) both the noise and the velocity are
projected by Pi0 = I - PR, so samples satisfy R dX = 0 exactly.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .fm import (
    FMSwinWrapper,
    FlowMatchingSampler,
    LogScaleModel,
    neglogcdf,
    neglogpdf,
    sample_uniform_time,
)

Proj = Optional[Callable[[Tensor], Tensor]]


class _Velocity:
    """Velocity callable for FlowMatchingSampler, with Pi0 projection and
    guidance v_w = v(.|null) + w [v(.|c) - v(.|null)]."""

    def __init__(self, net, n_state, proj: Proj = None, cond_null: Optional[Tensor] = None, w: float = 1.0):
        self.net, self.n_state, self.proj, self.cond_null, self.w = net, n_state, proj, cond_null, w

    def __call__(self, in_tensor, pseudo_time, mask=None):
        v = self.net(in_tensor, pseudo_time=pseudo_time, mask=mask)
        if self.cond_null is not None and self.w != 1.0:
            z = in_tensor[:, :self.n_state]
            v0 = self.net(torch.cat([z, self.cond_null], 1), pseudo_time=pseudo_time, mask=mask)
            v = v0 + self.w * (v - v0)
        return self.proj(v) if self.proj is not None else v


class Generator(nn.Module):

    def __init__(self, n_state: int, n_cond: int, grid: tuple[int, int], res_km: float, swin: dict):
        super().__init__()
        self.n_state, self.res_km = n_state, res_km
        self.net = FMSwinWrapper(n_input=n_state + n_cond, n_output=n_state, grid_size=grid,
                                 resolution_km=res_km, **swin)
        self.log_scale = LogScaleModel(n_embedding=64, n_time_in=1, n_res_in=1, n_augment_in=3, n_vars=n_state)

    def _log_scale(self, t: Tensor) -> Tensor:
        bsz = t.shape[0]
        return self.log_scale(t.view(bsz, 1), t.new_zeros(bsz, 3), t.new_full((bsz, 1), self.res_km)).view(bsz, -1, 1, 1)

    def loss(self, z1: Tensor, m: Tensor, cond: Tensor, valid: Tensor, floor: Optional[Tensor] = None,
             lo: Optional[Tensor] = None, hi: Optional[Tensor] = None, proj: Proj = None,
             noise: Optional[Tensor] = None, t: Optional[Tensor] = None):
        """Masked NLL of the velocity. Returns (per-channel loss [C], channel
        has-data flags [C], one-step prediction z1_hat, pseudo-time t)."""
        t = sample_uniform_time(z1) if t is None else t
        eps = torch.randn_like(z1) if noise is None else noise
        if proj is not None:
            eps = proj(eps)
        zt = t * z1 + (1 - t) * eps
        u = z1 - eps
        v = self.net(torch.cat([zt, cond], 1), pseudo_time=t.view(-1, 1), mask=valid)
        if proj is not None:
            v = proj(v)
        ls = self._log_scale(t)
        sigma = torch.exp(ls) if floor is None else torch.sqrt(torch.exp(2 * ls) + floor ** 2)
        r = (u - v) / sigma
        nll = neglogpdf(r, sigma.log())
        z_pred = zt + (1 - t) * v
        if lo is not None:
            # censoring at the bounds (same rule as the former CROSCIM FM model)
            nll = torch.where(z_pred <= lo, neglogcdf(-r), torch.where(z_pred >= hi, neglogcdf(r), nll))
        mf = m.to(nll.dtype)
        per_ch = (nll * mf).sum((0, 2, 3)) / mf.sum((0, 2, 3)).clamp_min(1)
        return per_ch, m.any(3).any(2).any(0), z_pred, t

    @torch.no_grad()
    def sample(self, cond: Tensor, valid: Tensor, n_steps: int, proj: Proj = None,
               lo: Optional[Tensor] = None, hi: Optional[Tensor] = None,
               cond_null: Optional[Tensor] = None, w: float = 1.0,
               generator: Optional[torch.Generator] = None) -> Tensor:
        """Heun sample; `generator` fixes the initial noise (paired noise)."""
        shape = (cond.shape[0], self.n_state, *cond.shape[-2:])
        eps = torch.randn(shape, generator=generator, device=cond.device, dtype=cond.dtype)
        if proj is not None:
            eps = proj(eps)
        bounds = (lo, hi) if lo is not None else None
        sampler = FlowMatchingSampler(model=_Velocity(self.net, self.n_state, proj, cond_null, w),
                                      n_steps=n_steps, second_order=True, censoring=bounds is not None)
        return sampler.sample(eps, cond, latent_bounds=bounds, mask=valid)


def channel_mean(per_ch: Tensor, has: Tensor) -> Tensor:
    """Mean of the per-channel losses over channels with data (0 if none)."""
    return per_ch[has].mean() if has.any() else per_ch.new_zeros(())
