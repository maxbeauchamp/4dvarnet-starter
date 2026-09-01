"""
Hierarchical Swin U-Net velocity backbone for CROSCIM Flow Matching.

Adapted from a colleague's air-quality Flow Matching model
(deep-chimere/models/flow_swin.py) for CROSCIM's multi-resolution setting:
the fixed single "tendency" output channel becomes a generic
``n_output_channels`` (CROSCIM packs n_vars x n_timesteps into channels),
and grid/padding sizes are set per CROSCIM resolution (x50: 294x304,
x10: 256x256) instead of hardcoded.

Architecture
------------
Patch-embed (Conv2d, stride=patch_size=2) -> 4 encoder stages (window
attention, except the coarsest stage which uses global attention) with 2D
RoPE + QK-norm self-attention and a gated MLP, each block modulated by a
low-rank AdaRMSNorm time embedding (adaLN-zero style) -> 3 decoder stages
(patch-expand + skip fusion + window attention) -> final upsample + output
head. ``padded_size`` must be a multiple of ``patch_size * 8`` (3 encoder
downsamples of x2 each); the backbone reflect-pads on the right/bottom and
crops back at the end, mirroring the pad-to-multiple-of-patch-size fix
already used for the ViT tokenizer in ``utils_flowmatching.py``.

This is a drop-in backbone for ``FMSwinSolver`` (``flowmatching_swin_solver.py``)
-- everything else in the FM pipeline (loss, EMA, sampler, boundary
conditioning, multi-res coupling) is unchanged and backbone-agnostic.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Norm / regularisation
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype) * self.weight.to(dtype)


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


# ─────────────────────────────────────────────────────────────────────────────
# Windowing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _window_partition(x: Tensor, window_size: int) -> Tensor:
    """(B, H, W, C) -> (B*n_windows, window_size**2, C)."""
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size,
               width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)


def _window_reverse(windows: Tensor, window_size: int, height: int, width: int,
                     batch_size: int) -> Tensor:
    channels = windows.shape[-1]
    x = windows.view(batch_size, height // window_size, width // window_size,
                      window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, channels)


def _apply_axis_rope(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    """Rotary embedding along one spatial axis, applied to half a head's dims."""
    pair_count = x.shape[-1] // 2
    x_pairs = x.float().reshape(*x.shape[:-1], pair_count, 2)
    angles = positions[:, None, :, None].float() * inv_freq[None, None, None]
    cos, sin = angles.cos(), angles.sin()
    even, odd = x_pairs[..., 0], x_pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────

class RotaryAttention(nn.Module):
    """Multi-head self-attention with QK-normalisation and 2D RoPE."""

    def __init__(self, dim: int, num_heads: int, rope_base: float = 10_000.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 4:
            raise ValueError(f"head_dim={self.head_dim} must be divisible by 4 for 2D RoPE")
        axis_dim = self.head_dim // 2
        inv_freq = rope_base ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.logit_scale = nn.Parameter(torch.full((num_heads,), math.log(10.0)))

    def forward(self, x: Tensor, positions: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        half = self.head_dim // 2
        q = torch.cat((_apply_axis_rope(q[..., :half], positions[..., 0], self.rope_inv_freq),
                       _apply_axis_rope(q[..., half:], positions[..., 1], self.rope_inv_freq)), dim=-1)
        k = torch.cat((_apply_axis_rope(k[..., :half], positions[..., 0], self.rope_inv_freq),
                       _apply_axis_rope(k[..., half:], positions[..., 1], self.rope_inv_freq)), dim=-1)
        q = F.normalize(q.float(), dim=-1).to(v.dtype)
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp().to(q.dtype)
        q = q * (scale[None, :, None, None] * math.sqrt(self.head_dim))
        if attention_mask is not None:
            attention_mask = attention_mask.to(q.dtype)

        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(attended)


class WindowAttention(nn.Module):
    """Shifted-window local attention at a fixed stage resolution."""

    def __init__(self, dim: int, num_heads: int, input_resolution: Tuple[int, int],
                 window_size: int, shift_size: int, stage_stride: int):
        super().__init__()
        self.height, self.width = input_resolution
        self.window_size = min(window_size, self.height, self.width)
        self.shift_size = 0 if self.window_size >= min(input_resolution) else shift_size
        self.attention = RotaryAttention(dim, num_heads)

        padded_h = math.ceil(self.height / self.window_size) * self.window_size
        padded_w = math.ceil(self.width / self.window_size) * self.window_size
        self.padded_h, self.padded_w = padded_h, padded_w
        self.pad_h, self.pad_w = padded_h - self.height, padded_w - self.width

        self.register_buffer("attention_mask", self._build_mask(), persistent=False)
        self.register_buffer("window_positions", self._build_positions(stage_stride), persistent=False)

    def _build_mask(self) -> Optional[Tensor]:
        ws = self.window_size
        mask = torch.zeros((1, self.padded_h, self.padded_w, 1), dtype=torch.int64)
        if self.shift_size:
            slices = (slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None))
            label = 0
            for h_slice in slices:
                for w_slice in slices:
                    mask[:, h_slice, w_slice] = label
                    label += 1
        region = _window_partition(mask, ws).squeeze(-1)
        blocked = region.unsqueeze(1) != region.unsqueeze(2)

        valid = torch.zeros((1, self.padded_h, self.padded_w, 1), dtype=torch.bool)
        valid[:, :self.height, :self.width] = True
        if self.shift_size:
            valid = torch.roll(valid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        valid_windows = _window_partition(valid, ws).squeeze(-1)
        blocked = blocked | (~valid_windows.unsqueeze(1))
        if not blocked.any():
            return None
        float_mask = torch.zeros(blocked.shape, dtype=torch.float32)
        return float_mask.masked_fill(blocked, float("-inf")).unsqueeze(1)

    def _build_positions(self, stage_stride: int) -> Tensor:
        y = (torch.arange(self.padded_h, dtype=torch.float32) + 0.5) * stage_stride
        x = (torch.arange(self.padded_w, dtype=torch.float32) + 0.5) * stage_stride
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        positions = torch.stack((yy, xx), dim=-1).unsqueeze(0)
        if self.shift_size:
            positions = torch.roll(positions, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        return _window_partition(positions, self.window_size)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, _ = x.shape
        if self.pad_h or self.pad_w:
            x = F.pad(x, (0, 0, 0, self.pad_w, 0, self.pad_h))
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = _window_partition(x, self.window_size)
        positions = self.window_positions.repeat(batch, 1, 1)
        attention_mask = self.attention_mask.repeat(batch, 1, 1, 1) if self.attention_mask is not None else None
        windows = self.attention(windows, positions, attention_mask)
        x = _window_reverse(windows, self.window_size, self.padded_h, self.padded_w, batch)
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x[:, :height, :width]


class GlobalAttention(nn.Module):
    """Full (non-windowed) attention, used at the coarsest encoder stage."""

    def __init__(self, dim: int, num_heads: int, input_resolution: Tuple[int, int], stage_stride: int):
        super().__init__()
        self.height, self.width = input_resolution
        self.attention = RotaryAttention(dim, num_heads)
        y = (torch.arange(self.height, dtype=torch.float32) + 0.5) * stage_stride
        x = (torch.arange(self.width, dtype=torch.float32) + 0.5) * stage_stride
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("positions", torch.stack((yy, xx), dim=-1).view(1, -1, 2), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, channels = x.shape
        flat = x.reshape(batch, height * width, channels)
        flat = self.attention(flat, self.positions.repeat(batch, 1, 1))
        return flat.view(batch, height, width, channels)


# ─────────────────────────────────────────────────────────────────────────────
# MLP + time modulation
# ─────────────────────────────────────────────────────────────────────────────

class GatedMLP(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hidden = int(dim * ratio)
        self.input = nn.Linear(dim, 2 * hidden)
        self.output = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


class TimeModulation(nn.Module):
    """Low-rank AdaRMSNorm (adaLN-zero style) time conditioning per block."""

    def __init__(self, time_dim: int, dim: int, rank: int):
        super().__init__()
        self.input = nn.Linear(time_dim, rank, bias=False)
        self.output = nn.Linear(rank, 6 * dim, bias=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, time_embedding: Tensor) -> Tuple[Tensor, ...]:
        values = self.output(F.silu(self.input(time_embedding))).chunk(6, dim=-1)
        return tuple(value[:, None, None] for value in values)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        half = dim // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(1, half - 1))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, pseudo_time: Tensor) -> Tensor:
        pseudo_time = pseudo_time.reshape(-1, 1).float()
        angles = 2.0 * math.pi * pseudo_time * self.frequencies[None]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Swin block / stage
# ─────────────────────────────────────────────────────────────────────────────

class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, input_resolution: Tuple[int, int],
                 window_size: int, shift_size: int, stage_stride: int, time_dim: int,
                 modulation_rank: int, mlp_ratio: float, drop_path: float, global_attention: bool):
        super().__init__()
        self.norm_attention = RMSNorm(dim)
        if global_attention:
            self.attention = GlobalAttention(dim, num_heads, input_resolution, stage_stride)
        else:
            self.attention = WindowAttention(dim, num_heads, input_resolution, window_size,
                                              shift_size, stage_stride)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = GatedMLP(dim, mlp_ratio)
        self.modulation = TimeModulation(time_dim, dim, modulation_rank)
        self.drop_path = DropPath(drop_path)

    @staticmethod
    def _modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        return x * (1.0 + scale) + shift

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = self.modulation(time_embedding)
        x = x + self.drop_path(gate_a * self.attention(self._modulate(self.norm_attention(x), scale_a, shift_a)))
        return x + self.drop_path(gate_m * self.mlp(self._modulate(self.norm_mlp(x), scale_m, shift_m)))


class SwinStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, input_resolution: Tuple[int, int],
                 window_size: int, stage_stride: int, time_dim: int, modulation_rank: int,
                 mlp_ratio: float, drop_paths: Sequence[float], global_attention: bool,
                 activation_checkpointing: bool):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, num_heads=num_heads, input_resolution=input_resolution,
                      window_size=window_size,
                      shift_size=0 if global_attention or index % 2 == 0 else window_size // 2,
                      stage_stride=stage_stride, time_dim=time_dim, modulation_rank=modulation_rank,
                      mlp_ratio=mlp_ratio, drop_path=drop_paths[index], global_attention=global_attention)
            for index in range(depth)
        ])

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        for block in self.blocks:
            if self.activation_checkpointing and self.training and x.requires_grad:
                x = checkpoint(block, x, time_embedding, use_reentrant=False)
            else:
                x = block(x, time_embedding)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Patch merge / expand
# ─────────────────────────────────────────────────────────────────────────────

class PatchMerging(nn.Module):
    """2x2 token concatenation, 4C -> 2C."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        merged = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1)
        return self.reduction(self.norm(merged))


class PatchExpand(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.expand = nn.Linear(input_dim, 4 * output_dim, bias=False)
        self.norm = RMSNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, height, width, _ = x.shape
        x = self.expand(x).view(batch, height, width, 2, 2, self.output_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, 2 * height, 2 * width, self.output_dim)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full backbone
# ─────────────────────────────────────────────────────────────────────────────

class SwinUNetBackbone(nn.Module):
    """Fixed-grid Swin U-Net predicting an ``n_output_channels`` velocity field.

    Parameters
    ----------
    condition_channels : int
        Channels of the conditioning tensor (processed obs [+ boundary
        channels if add_bounds] -- built by ``FMSolver._encode_obs``).
    output_channels : int
        Channels of the noisy state / predicted velocity (CROSCIM:
        n_target_vars * n_timesteps, e.g. 15 for x50, 8 for x10).
    grid_size : (H, W)
        Native patch size for this resolution (CROSCIM: (294, 304) for x50,
        (256, 256) for x10).
    padded_size : (H, W)
        ``grid_size`` rounded up to a multiple of ``patch_size * 8``.
    """

    def __init__(
        self,
        condition_channels: int,
        output_channels: int,
        grid_size: Tuple[int, int],
        padded_size: Tuple[int, int],
        patch_size: int = 2,
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
        parameter_count_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        if patch_size != 2:
            raise ValueError("SwinUNetBackbone requires patch_size=2")
        if len(encoder_depths) != 4 or len(decoder_depths) != 3 or len(num_heads) != 4:
            raise ValueError("SwinUNetBackbone expects four encoder and three decoder stages")
        self.condition_channels = int(condition_channels)
        self.output_channels = int(output_channels)
        self.grid_size = tuple(int(v) for v in grid_size)
        self.padded_size = tuple(int(v) for v in padded_size)
        self.pad_h = self.padded_size[0] - self.grid_size[0]
        self.pad_w = self.padded_size[1] - self.grid_size[1]
        if self.pad_h < 0 or self.pad_w < 0:
            raise ValueError("padded_size must contain grid_size")
        if self.padded_size[0] % 16 or self.padded_size[1] % 16:
            raise ValueError("padded_size must be divisible by patch_size(2) * 8 = 16")

        dims = [base_dim * (2 ** index) for index in range(4)]
        resolutions = [
            (self.padded_size[0] // patch_size // (2 ** index),
             self.padded_size[1] // patch_size // (2 ** index))
            for index in range(4)
        ]
        total_blocks = sum(encoder_depths) + sum(decoder_depths)
        drop_paths = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        drop_index = 0

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.patch_embedding = nn.Conv2d(
            self.condition_channels + self.output_channels, dims[0],
            kernel_size=patch_size, stride=patch_size,
        )

        self.encoder_stages = nn.ModuleList()
        self.patch_merges = nn.ModuleList()
        for stage_index in range(4):
            depth = int(encoder_depths[stage_index])
            stage_drops = drop_paths[drop_index:drop_index + depth]
            drop_index += depth
            self.encoder_stages.append(SwinStage(
                dim=dims[stage_index], depth=depth, num_heads=int(num_heads[stage_index]),
                input_resolution=resolutions[stage_index], window_size=window_size,
                stage_stride=patch_size * (2 ** stage_index), time_dim=time_dim,
                modulation_rank=modulation_rank, mlp_ratio=mlp_ratio, drop_paths=stage_drops,
                global_attention=stage_index == 3, activation_checkpointing=activation_checkpointing,
            ))
            if stage_index < 3:
                self.patch_merges.append(PatchMerging(dims[stage_index]))

        self.patch_expands = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for decoder_index, encoder_index in enumerate((2, 1, 0)):
            output_dim = dims[encoder_index]
            self.patch_expands.append(PatchExpand(dims[encoder_index + 1], output_dim))
            self.skip_fusions.append(nn.Linear(2 * output_dim, output_dim))
            depth = int(decoder_depths[decoder_index])
            stage_drops = drop_paths[drop_index:drop_index + depth]
            drop_index += depth
            self.decoder_stages.append(SwinStage(
                dim=output_dim, depth=depth, num_heads=int(num_heads[encoder_index]),
                input_resolution=resolutions[encoder_index], window_size=window_size,
                stage_stride=patch_size * (2 ** encoder_index), time_dim=time_dim,
                modulation_rank=modulation_rank, mlp_ratio=mlp_ratio, drop_paths=stage_drops,
                global_attention=False, activation_checkpointing=activation_checkpointing,
            ))

        self.final_expand = PatchExpand(dims[0], dims[0] // 2)
        self.output_norm = RMSNorm(dims[0] // 2)
        self.output_head = nn.Linear(dims[0] // 2, self.output_channels)

        if parameter_count_range is not None:
            count = self.parameter_count()
            lower, upper = parameter_count_range
            if not int(lower) <= count <= int(upper):
                raise RuntimeError(
                    f"SwinUNetBackbone parameter count {count:,} is outside [{int(lower):,}, {int(upper):,}]"
                )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, noisy_state: Tensor, conditions: Tensor, pseudo_time: Tensor) -> Tensor:
        """
        Parameters
        ----------
        noisy_state : (B, output_channels, H, W)
        conditions  : (B, condition_channels, H, W)
        pseudo_time : (B, 1)

        Returns
        -------
        (B, output_channels, H, W) -- predicted velocity field.
        """
        x = torch.cat((noisy_state, conditions), dim=1)
        if self.pad_h or self.pad_w:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="reflect")
        time_embedding = self.time_embedding(pseudo_time)
        x = self.patch_embedding(x).permute(0, 2, 3, 1)

        skips = []
        for stage_index, stage in enumerate(self.encoder_stages):
            x = stage(x, time_embedding)
            skips.append(x)
            if stage_index < len(self.patch_merges):
                x = self.patch_merges[stage_index](x)

        for expand, fusion, stage, skip in zip(
            self.patch_expands, self.skip_fusions, self.decoder_stages, reversed(skips[:-1])
        ):
            x = expand(x)
            x = fusion(torch.cat((x, skip), dim=-1))
            x = stage(x, time_embedding)

        x = self.final_expand(x)
        x = self.output_head(self.output_norm(x)).permute(0, 3, 1, 2)
        return x[:, :, :self.grid_size[0], :self.grid_size[1]]
