"""
utils_flowmatching.py
=====================
Self-contained utilities for Flow Matching in the CROSCIM framework.

Replaces all gensim imports used by ``flowmatching_solver.py`` and
``models_flowmatching.py``:

    gensim.utils    → neglogpdf, neglogcdf, sample_uniform_time, masked_average
    gensim.embedding → LogScaleModel
    gensim.network   → Transformer  (full ViT-style patch Transformer)
    gensim.sampler   → FlowMatchingSampler (Euler / Heun ODE)

No external dependencies beyond PyTorch and einops.

Authors: adapted from Tobias Sebastian Finn (gensim, ENPC, 2025).
"""

from __future__ import annotations

import math
import logging
from math import sqrt
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from einops import rearrange, reduce
except ImportError:  # pragma: no cover
    raise ImportError("einops is required: pip install einops")

main_logger = logging.getLogger(__name__)

__all__ = [
    # loss / probability utilities
    "neglogpdf",
    "neglogcdf",
    "sample_uniform_time",
    "masked_average",
    # embedding
    "LogScaleModel",
    # network
    "Transformer",
    # sampler
    "FlowMatchingSampler",
]


# ─────────────────────────────────────────────────────────────────────────────
# Loss / probability utilities  (from gensim.utils)
# ─────────────────────────────────────────────────────────────────────────────

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


def masked_average(
    to_average: Tensor,
    mask: Tensor,
    dim=None,
) -> Tensor:
    """Compute the mean of ``to_average`` only where ``mask`` is True/nonzero."""
    expanded_mask = mask.expand_as(to_average)
    masked_sum = (to_average * expanded_mask).sum(dim=dim)
    return masked_sum / expanded_mask.sum(dim=dim).clamp(min=1)


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


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper used by Transformer blocks  (from gensim.network)
# ─────────────────────────────────────────────────────────────────────────────

def _mask_tensor(in_tensor: Tensor, mask: Tensor) -> Tensor:
    return in_tensor * mask.to(dtype=in_tensor.dtype)


def _self_attention(
    q: Tensor, k: Tensor, v: Tensor
) -> Tensor:
    """Scaled dot-product attention with optional flash-attention backend."""
    try:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)
    except Exception:
        scale = q.size(-1) ** -0.5
        attn = torch.einsum("bthd,bshd->bths", q * scale, k).softmax(dim=-1)
        return torch.einsum("bths,bshd->bthd", attn, v)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE  (from gensim.network)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_sine_features(
    mesh: Tensor, freqs: Tensor
) -> Tuple[Tensor, Tensor]:
    """Project a spatial mesh onto sinusoidal RoPE features."""
    mesh_f32 = mesh.float()
    freqs_f32 = freqs.float()
    embedded = torch.einsum("blc,chk->blhk", mesh_f32, freqs_f32)
    return embedded.sin().to(mesh.dtype), embedded.cos().to(mesh.dtype)


def _apply_rope(
    in_tensor: Tensor,
    features: Tuple[Tensor, Tensor],
    n_features: int = 32,
) -> Tensor:
    sin_f, cos_f = features
    rotated_first = (
        in_tensor[..., :n_features] * cos_f
        - in_tensor[..., n_features : 2 * n_features] * sin_f
    )
    rotated_second = (
        in_tensor[..., :n_features] * sin_f
        + in_tensor[..., n_features : 2 * n_features] * cos_f
    )
    return torch.cat((rotated_first, rotated_second, in_tensor[..., 2 * n_features :]), dim=-1)


class _RopeLayer(nn.Module):
    def __init__(
        self,
        n_features: int = 16,
        n_heads: int = 8,
        min_theta: float = 0.0,
        max_theta: float = 0.333333,
        random_angle: bool = True,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.freqs = nn.Parameter(torch.empty(2, n_heads, n_features))
        self._init_freqs(n_heads, min_theta, max_theta, random_angle)

    def _init_freqs(self, n_heads, min_theta, max_theta, random_angle):
        mag = 1 / torch.logspace(min_theta, max_theta, self.n_features // 2, base=10.0)[None, :]
        if random_angle:
            angles = torch.rand(n_heads, 1) * 2 * math.pi
        else:
            angles = torch.zeros(n_heads, 1)
        freqs_x = torch.cat([mag * angles.cos(), mag * (math.pi / 2 + angles).cos()], dim=-1)
        freqs_y = torch.cat([mag * angles.sin(), mag * (math.pi / 2 + angles).sin()], dim=-1)
        self.freqs.data.copy_(torch.stack([freqs_x, freqs_y], dim=0))

    def forward(self, q: Tensor, k: Tensor, mesh: Tensor) -> Tuple[Tensor, Tensor]:
        sine_features = _estimate_sine_features(mesh, self.freqs)
        return _apply_rope(q, sine_features, self.n_features), _apply_rope(k, sine_features, self.n_features)


# ─────────────────────────────────────────────────────────────────────────────
# Attention + MLP blocks  (from gensim.network)
# ─────────────────────────────────────────────────────────────────────────────

class _SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        n_features: int = 512,
        n_features_head: int = 64,
        n_heads: int = 8,
        n_rope_features: int = 16,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_features_head = n_features_head
        self.n_heads = n_heads

        self.qkv_layer = nn.Linear(n_features, n_features_head * n_heads * 3, bias=False)
        self.q_norm = nn.RMSNorm(n_features_head)
        self.k_norm = nn.RMSNorm(n_features_head)
        self.rope_layer = _RopeLayer(n_features=n_rope_features, n_heads=n_heads) if n_rope_features > 0 else None
        self.out_layer = nn.Linear(n_features_head * n_heads, n_features, bias=False)

    def forward(self, in_tensor: Tensor, mesh: Tensor) -> Tensor:
        B, T, _ = in_tensor.shape
        qkv = self.qkv_layer(in_tensor).view(B, T, 3, self.n_heads, self.n_features_head)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.rope_layer is not None:
            q, k = self.rope_layer(q, k, mesh)
        out = _self_attention(q, k, v).reshape(B, T, -1)
        return self.out_layer(out)


class _MLPLayer(nn.Module):
    def __init__(self, n_features: int, mult: int = 1) -> None:
        super().__init__()
        hidden = n_features * mult
        self.in_layer = nn.Linear(n_features, hidden * 2, bias=False)
        self.act = nn.SiLU()
        self.out_layer = nn.Linear(hidden, n_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        branch, gate = self.in_layer(x).chunk(2, dim=-1)
        return self.out_layer(branch * self.act(gate))


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        n_features: int = 512,
        n_features_head: int = 64,
        n_heads: int = 8,
        n_embedding: int = 256,
        n_rope_features: int = 16,
        mult: int = 1,
    ) -> None:
        super().__init__()
        self.gate_layer = nn.Linear(n_embedding, n_features * 2, bias=False)
        self.pre_attn_norm = nn.RMSNorm(n_features, elementwise_affine=False)
        self.self_attention = _SelfAttentionLayer(n_features, n_features_head, n_heads, n_rope_features)
        self.pre_mlp_norm = nn.RMSNorm(n_features, elementwise_affine=False)
        self.mlp = _MLPLayer(n_features, mult)

    def forward(self, x: Tensor, mesh: Tensor, mask: Tensor, emb: Tensor) -> Tensor:
        gate_attn, gate_mlp = (self.gate_layer(emb)[:, None] + 1).chunk(2, dim=-1)
        # attention branch
        res = _mask_tensor(self.self_attention(self.pre_attn_norm(x) * gate_attn, mesh), mask)
        x = x + res
        # MLP branch
        res = _mask_tensor(self.mlp(self.pre_mlp_norm(x) * gate_mlp), mask)
        return x + res


# ─────────────────────────────────────────────────────────────────────────────
# Embedder  (from gensim.embedding)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer + Head  (from gensim.network)
# ─────────────────────────────────────────────────────────────────────────────

class _Tokenizer(nn.Module):
    def __init__(self, n_input: int, n_features: int, patch_size: int = 4, lengthscale: float = 100.0) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.lengthscale = lengthscale
        self.in_encoder = nn.Linear((n_input + 1) * patch_size * patch_size, n_features, bias=False)

    def forward(self, in_tensor: Tensor, mesh: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        p = self.patch_size
        # pad with ones channel (bias-like)
        in_pad = F.pad(in_tensor, (0, 0, 0, 0, 0, 1), value=1.0)
        in_masked = _mask_tensor(in_pad, mask)
        tokens = rearrange(in_masked, "b c (h hp) (w wp) -> b (h w) (c hp wp)", hp=p, wp=p)
        tokens = self.in_encoder(tokens)
        tokens_mesh = (
            reduce(mesh, "b c (h hp) (w wp) -> b (h w) c", "mean", hp=p, wp=p) / self.lengthscale
        ).to(tokens)
        tokens_mask = reduce(mask, "b c (h hp) (w wp) -> b (h w) c", "max", hp=p, wp=p).to(tokens)
        return tokens, tokens_mesh, tokens_mask


class _Head(nn.Module):
    def __init__(self, n_output: int, n_features: int, n_embedding: int, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.gate_layer = nn.Linear(n_embedding, n_features, bias=False)
        self.in_norm = nn.RMSNorm(n_features, elementwise_affine=False)
        self.in_activation = nn.ReLU()
        self.out_layer = nn.Linear(n_features, n_output * patch_size ** 2, bias=False)
        # init: nearest-neighbour-like interpolation
        w = torch.empty(n_output, n_features)
        nn.init.kaiming_normal_(w, nonlinearity="linear")
        self.out_layer.weight.data.copy_(w.repeat_interleave(patch_size ** 2, dim=0))

    def forward(self, tokens: Tensor, mask: Tensor, embedding: Tensor) -> Tensor:
        p = self.patch_size
        h = mask.size(-2) // p
        w = mask.size(-1) // p
        gate = self.gate_layer(embedding)[:, None] + 1
        out = self.out_layer(self.in_activation(self.in_norm(tokens) * gate))
        out = rearrange(out, "b (h w) (c h2 w2) -> b c (h h2) (w w2)", h=h, w=w, h2=p, w2=p)
        return _mask_tensor(out, mask)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer  (from gensim.network)
# ─────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """ViT-style patch Transformer with RoPE attention and U-Net long skips.

    This is a faithful re-implementation of ``gensim.network.Transformer``
    without any gensim dependency.
    """

    def __init__(
        self,
        n_input: int = 11,
        n_output: int = 5,
        n_features: int = 512,
        n_blocks: int = 8,
        n_embedding: int = 256,
        n_time_in: int = 1,
        n_res_in: int = 1,
        n_augment_in: int = 3,
        n_features_head: int = 64,
        n_heads: int = 8,
        n_rope_features: int = 16,
        mult: int = 1,
        patch_size: int = 4,
        long_skips: bool = True,
        lengthscale: float = 100.0,
        dropout_mlp: float = 0.0,   # kept for API compat, unused
    ) -> None:
        super().__init__()
        self.n_skips = (n_blocks - 1) // 2 if long_skips else 0
        self.long_skips = long_skips
        self.patch_size = patch_size

        self.embedder = _Embedder(n_embedding, n_time_in, n_res_in, n_augment_in)
        self.tokenizer = _Tokenizer(n_input, n_features, patch_size, lengthscale)

        _block = lambda: _TransformerBlock(n_features, n_features_head, n_heads, n_embedding, n_rope_features, mult)
        self.in_blocks = nn.ModuleList([_block() for _ in range(self.n_skips)])
        self.bottleneck = nn.ModuleList([_block() for _ in range(n_blocks - 2 * self.n_skips)])
        self.out_blocks = nn.ModuleList([_block() for _ in range(self.n_skips)])

        self.skip_embedding = nn.Linear(n_embedding, n_features * (self.n_skips + 1), bias=False)
        self.head = _Head(n_output, n_features, n_embedding, patch_size)

    def forward(
        self,
        in_tensor: Tensor,
        mesh: Tensor,
        mask: Tensor,
        pseudo_time: Tensor,
        labels: Tensor,
        resolution: Tensor,
    ) -> Tensor:
        embedding = self.embedder(in_tensor, pseudo_time=pseudo_time, labels=labels, resolution=resolution)
        tokens, tokens_mesh, tokens_mask = self.tokenizer(in_tensor, mesh, mask)
        tokens_mesh = tokens_mesh / resolution.unsqueeze(1)

        gates = self.skip_embedding(embedding)[:, None, :].chunk(self.n_skips + 1, dim=2)
        skips = [tokens]

        for block in self.in_blocks:
            tokens = block(tokens, tokens_mesh, tokens_mask, embedding)
            skips.append(tokens)

        for block in self.bottleneck:
            tokens = block(tokens, tokens_mesh, tokens_mask, embedding)

        for k, block in enumerate(self.out_blocks):
            tokens = torch.lerp(skips[-(k + 1)], tokens, gates[k].expand_as(tokens))
            tokens = block(tokens, tokens_mesh, tokens_mask, embedding)

        if self.long_skips:
            tokens = torch.lerp(skips[0], tokens, gates[-1].expand_as(tokens))

        return self.head(tokens, mask, embedding)  # pass original spatial mask


# ─────────────────────────────────────────────────────────────────────────────
# FlowMatchingSampler  (from gensim.sampler)
# ─────────────────────────────────────────────────────────────────────────────

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
