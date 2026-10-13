"""
fm.py
=====
Flow-matching building blocks reused from the former CROSCIM FM solver
(solvers/utils_flowmatching.py and solvers/flowmatching_swin_solver.py),
copied verbatim: velocity NLL (neglogpdf / neglogcdf), low-discrepancy
pseudo-time sampling, LogScaleModel, Euler / Heun FlowMatchingSampler and the
FMSwinWrapper around the Swin U-Net backbone.

Authors: adapted from Tobias Sebastian Finn (gensim, ENPC, 2025).
"""

from __future__ import annotations

import math
from math import sqrt
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import SwinUNetBackbone

_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


def neglogpdf(value: Tensor, log_scale: Tensor) -> Tensor:
    """Negative log-PDF of a unit normal evaluated at ``value / exp(log_scale)``.

    Parameters
    ----------
    value : Tensor
        Normalised residual  (v_target - v_pred) / exp(log_scale).
    log_scale : Tensor
        Log standard deviation (broadcastable with ``value``).
    """
    return 0.5 * value.pow(2) + log_scale + _LOG_SQRT_2PI


def neglogcdf(value: Tensor) -> Tensor:
    """Negative log-CDF of the standard normal: -log Φ(value)."""
    return -torch.special.log_ndtr(value)


def sample_uniform_time(template_tensor: Tensor) -> Tensor:
    """Low-discrepancy uniform time sampling (Kingma et al., 2021).

    Returns a ``(B, 1, 1, 1)``-shaped tensor with one pseudo-time value per
    batch element, drawn such that the batch collectively covers [0, 1] evenly.
    """
    time_shape = torch.Size(
        [template_tensor.size(0)] + [1] * (template_tensor.ndim - 1)
    )
    time_shift = torch.rand(
        1, dtype=template_tensor.dtype, device=template_tensor.device
    )
    sampled_time = torch.linspace(
        0, 1, template_tensor.size(0) + 1,
        dtype=template_tensor.dtype, device=template_tensor.device,
    )[: template_tensor.size(0)]
    sampled_time = (time_shift + sampled_time) % 1
    return sampled_time.reshape(time_shape)


class _RandomFourierEmbedding(nn.Module):
    def __init__(self, n_in: int = 1, n_features: int = 512, scale: float = 1.0) -> None:
        super().__init__()
        half = n_features // 2
        self.register_buffer("frequencies", torch.randn(n_in, half) / scale)
        self.scaling = 1 / sqrt(half)

    def forward(self, x: Tensor) -> Tensor:
        emb = x @ self.frequencies
        return torch.cat([emb.sin(), emb.cos()], dim=-1) * self.scaling


class _Embedder(nn.Module):
    def __init__(
        self,
        n_embedding: int = 512,
        n_time_in: int = 1,
        n_res_in: int = 1,
        n_augment_in: int = 3,
    ) -> None:
        super().__init__()
        self.n_embedding = n_embedding

        def _block(n_in, scale):
            return nn.Sequential(
                _RandomFourierEmbedding(n_in, n_embedding, scale),
                nn.Linear(n_embedding, n_embedding),
                nn.SiLU(),
                nn.Linear(n_embedding, n_embedding),
            )

        self.time_embedder = _block(n_time_in, 0.15) if n_time_in > 0 else None
        self.augment_embedder = _block(n_augment_in, 1.0) if n_augment_in > 0 else None
        self.resolution_embedder = _block(n_res_in, 1.5) if n_res_in > 0 else None
        self.activation = nn.SiLU()

    def forward(self, in_tensor: Tensor, pseudo_time: Tensor, labels: Tensor, resolution: Tensor) -> Tensor:
        emb = torch.zeros(in_tensor.size(0), self.n_embedding, device=in_tensor.device, dtype=in_tensor.dtype)
        if self.time_embedder is not None:
            emb = emb + self.time_embedder(pseudo_time)
        if self.augment_embedder is not None:
            emb = emb + self.augment_embedder(labels)
        if self.resolution_embedder is not None:
            emb = emb + self.resolution_embedder(resolution.log())
        return self.activation(emb)


# ─────────────────────────────────────────────────────────────────────────────
# LogScaleModel  (from gensim.embedding)
# ─────────────────────────────────────────────────────────────────────────────

class LogScaleModel(nn.Module):
    """Learnable per-variable log-scale conditioned on pseudo-time and resolution.

    Initialised to zero so training starts with unit variance.
    """

    def __init__(
        self,
        n_embedding: int = 512,
        n_time_in: int = 1,
        n_res_in: int = 1,
        n_augment_in: int = 3,
        n_vars: int = 6,
    ) -> None:
        super().__init__()
        self.embedder = _Embedder(n_embedding, n_time_in, n_res_in, n_augment_in)
        self.out_layer = nn.Linear(n_embedding, n_vars)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, pseudo_time: Tensor, labels: Tensor, resolution: Tensor) -> Tensor:
        features = self.embedder(pseudo_time, pseudo_time, labels, resolution)
        return self.out_layer(features)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    remainder = value % multiple
    return value if remainder == 0 else value + (multiple - remainder)


class FMSwinWrapper(nn.Module):
    """Wraps ``SwinUNetBackbone`` to match the calling convention used by
    ``FlowMatchingSampler``::

        model(cat(states, encoded), pseudo_time=t, mesh=..., mask=...,
              labels=..., resolution=...)

    ``mesh``/``labels``/``resolution`` are accepted for signature
    compatibility but ignored: the Swin backbone builds its own RoPE
    positions internally from ``grid_size``, and ``labels``/``resolution``
    are unused (each CROSCIM resolution gets its own solver/network
    instance, and ``labels`` is always zero in the existing pipeline).
    ``mask`` (1 = valid/ocean, 0 = land), when given, is forwarded to the
    backbone's window/global attention to exclude land tokens from the
    attention context -- see ``SwinUNetBackbone.forward(valid_mask=...)``.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        grid_size: Tuple[int, int],
        resolution_km: float = 50.,
        base_dim: int = 128,
        encoder_depths: Sequence[int] = (2, 2, 18, 2),
        decoder_depths: Sequence[int] = (6, 2, 2),
        num_heads: Sequence[int] = (2, 4, 8, 16),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        time_dim: int = 512,
        modulation_rank: int = 128,
        drop_path_rate: float = 0.1,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.n_output = n_output
        self.resolution_km = resolution_km
        grid_size = tuple(int(v) for v in grid_size)
        padded_size = (_round_up_to_multiple(grid_size[0], 16),
                       _round_up_to_multiple(grid_size[1], 16))

        self.backbone = SwinUNetBackbone(
            condition_channels=n_input - n_output,
            output_channels=n_output,
            grid_size=grid_size,
            padded_size=padded_size,
            base_dim=base_dim,
            encoder_depths=encoder_depths,
            decoder_depths=decoder_depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            time_dim=time_dim,
            modulation_rank=modulation_rank,
            drop_path_rate=drop_path_rate,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        in_tensor: Tensor,
        pseudo_time: Tensor,
        mesh: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        resolution: Optional[Tensor] = None,
    ) -> Tensor:
        noisy_state = in_tensor[:, :self.n_output]
        conditions = in_tensor[:, self.n_output:]
        return self.backbone(noisy_state, conditions, pseudo_time, valid_mask=mask)


def _get_schedule(n_steps: int, scale: float = 0.1, shift: float = 0.0) -> Tensor:
    t = torch.linspace(0.0, 1.0, n_steps + 1)
    if scale > 0:
        sig = torch.sigmoid(scale * (t - 0.5) + shift)
        t = (sig - sig[0]) / (sig[-1] - sig[0])
    return t


class FlowMatchingSampler:
    """Euler / Heun ODE integrator for conditional flow matching.

    Faithful re-implementation of ``gensim.sampler.FlowMatchingSampler``.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_steps: int = 20,
        schedule_scale: float = 0.1,
        schedule_shift: float = 0.0,
        second_order: bool = True,
        censoring: bool = True,
    ) -> None:
        self.model = model
        self.n_steps = n_steps
        self.schedule = _get_schedule(n_steps, schedule_scale, schedule_shift)
        self.second_order = second_order
        self.censoring = censoring

    # ------------------------------------------------------------------
    def _activate_bounding(
        self,
        pseudo_time: Tensor,
        latent_bounds: Optional[Tuple[Tensor, Tensor]],
    ) -> bool:
        return (
            latent_bounds is not None
            and (pseudo_time < 1).all().item()
            and self.censoring
        )

    @staticmethod
    def _bound_grad(
        states: Tensor,
        grad: Tensor,
        pseudo_time: Tensor,
        latent_bounds: Tuple[Tensor, Tensor],
    ) -> Tensor:
        view = (-1,) + (1,) * (states.dim() - 1)
        prediction = (states + (1 - pseudo_time.view(view)) * grad).clamp(
            min=latent_bounds[0], max=latent_bounds[1]
        )
        return (prediction - states) / (1 - pseudo_time.view(view))

    def _compute_grad(
        self,
        states: Tensor,
        encoded: Tensor,
        pseudo_time: Tensor,
        latent_bounds: Optional[Tuple[Tensor, Tensor]] = None,
        **model_kwargs,
    ) -> Tensor:
        in_tensor = torch.cat((states, encoded), dim=1)
        grad = self.model(in_tensor, pseudo_time=pseudo_time, **model_kwargs)
        if self._activate_bounding(pseudo_time, latent_bounds):
            grad = self._bound_grad(states, grad, pseudo_time, latent_bounds)
        return grad

    @staticmethod
    def _update(state: Tensor, grad: Tensor, t0: float, t1: float) -> Tensor:
        return state + (t1 - t0) * grad

    @torch.no_grad()
    def sample(
        self,
        states: Tensor,
        encoded: Tensor,
        latent_bounds: Optional[Tuple[Tensor, Tensor]] = None,
        **model_kwargs,
    ) -> Tensor:
        if self.model is None:
            raise ValueError("Set FlowMatchingSampler.model before calling sample().")
        pseudo_time = torch.ones(states.size(0), 1, device=encoded.device, dtype=encoded.dtype)
        schedule = self.schedule.to(device=encoded.device, dtype=encoded.dtype)

        for idx in range(len(schedule) - 1):
            curr_t = schedule[idx]
            next_t = schedule[idx + 1]
            pseudo_time.fill_(curr_t.item())

            grad = self._compute_grad(states, encoded, pseudo_time, latent_bounds, **model_kwargs)

            if self.second_order and next_t < 1:
                next_states = self._update(states, grad, curr_t.item(), next_t.item())
                pseudo_time.fill_(next_t.item())
                grad_next = self._compute_grad(next_states, encoded, pseudo_time, latent_bounds, **model_kwargs)
                grad = 0.5 * (grad + grad_next)

            states = self._update(states, grad, curr_t.item(), next_t.item())

        return states
