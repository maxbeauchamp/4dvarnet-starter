"""
Consistency Model Solver for CROSCIM multi-resolution framework.

Defines a UNet with time-level embeddings for pairwise consistency training,
wrapped in a CROSCIM-compatible solver interface (GradSolvers / sBatch).

The UNet follows the architecture from the PWCM notebook:
- Input: concatenation of [x, y, mask_obs] → channels * 3
- Time embeddings: dual TimeLevelEmbedding (time, time_prime)
- Output: channels (denoised prediction)

The ConsistencyUNetSolver wraps this UNet so that it can be called from
CROSCIM's multi-resolution pipeline via `solver(batch)`.
"""

import os
import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


# ──────────────────────────────────────────────────────────────────────
# Building blocks (same as notebook UNet)
# ──────────────────────────────────────────────────────────────────────

def GroupNorm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, channels // 4), num_channels=channels)


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.qkv_projection = nn.Sequential(
            GroupNorm(in_channels),
            nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False),
            Rearrange("b (i h d) x y -> i b h (x y) d", i=3, h=n_heads),
        )
        self.output_projection = nn.Sequential(
            Rearrange("b h l d -> b l (h d)"),
            nn.Linear(in_channels, out_channels, bias=False),
            Rearrange("b l d -> b d l"),
            GroupNorm(out_channels),
            nn.Dropout1d(dropout),
        )
        self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.qkv_projection(x).unbind(dim=0)
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        output = self.output_projection(output)
        output = rearrange(output, "b c (x y) -> b c x y", x=x.shape[-2], y=x.shape[-1])
        return output + self.residual_projection(x)


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_level_channels: int, dropout: float = 0.3):
        super().__init__()
        self.input_projection = nn.Sequential(
            GroupNorm(in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.time_level_projection = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(time_level_channels, out_channels, kernel_size=1),
        )
        self.output_projection = nn.Sequential(
            GroupNorm(out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, time_level: Tensor) -> Tensor:
        h = self.input_projection(x)
        h = h + self.time_level_projection(time_level)
        return self.output_projection(h) + self.residual_projection(x)


class UNetBlockWithSelfAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_level_channels: int,
                 n_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.unet_block = UNetBlock(in_channels, out_channels, time_level_channels, dropout)
        self.self_attention = SelfAttention(out_channels, out_channels, n_heads, dropout)

    def forward(self, x: Tensor, time_level: Tensor) -> Tensor:
        return self.self_attention(self.unet_block(x, time_level))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange("b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2),
            nn.Conv2d(4 * channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class TimeLevelEmbedding(nn.Module):
    def __init__(self, channels: int, scale: float = 16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)
        self.projection = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
            Rearrange("b c -> b c () ()"),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x[:, None] * self.W[None, :] * 2 * torch.pi
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)
        return self.projection(h)


# ──────────────────────────────────────────────────────────────────────
# UNet with time-level embeddings for consistency models
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ConsistencyUNetConfig:
    """Configuration for the consistency UNet.
    
    Attributes:
        channels: Number of output channels (= n_target_vars * n_time).
        n_input_channels: Number of input channels (= n_input_vars * n_time).
            The UNet's input projection will be: n_input_channels + channels + channels
            corresponding to [obs_input, noisy_x, mask_obs].
            If None, defaults to channels (backward compat with same-dim input/output).
        time_level_channels: Dimension of time-level embeddings.
        time_level_scale: Scale for sinusoidal frequency basis.
        n_heads: Number of attention heads in mid blocks.
        top_blocks_channels: Channel widths for top encoder/decoder.
        top_blocks_n_blocks_per_resolution: Number of UNet blocks per top resolution.
        top_blocks_has_resampling: Whether each top level has down/up-sampling.
        top_blocks_dropout: Dropout per top level.
        mid_blocks_channels: Channel widths for mid encoder/decoder.
        mid_blocks_n_blocks_per_resolution: Number of UNet blocks per mid resolution.
        mid_blocks_has_resampling: Whether each mid level has down/up-sampling.
        mid_blocks_dropout: Dropout per mid level.
    """
    channels: int = 15
    n_input_channels: int = None  # if None, same as channels
    time_level_channels: int = 256
    time_level_scale: float = 16.0
    n_heads: int = 8
    top_blocks_channels: Tuple[int, ...] = (128, 128)
    top_blocks_n_blocks_per_resolution: Tuple[int, ...] = (2, 2)
    top_blocks_has_resampling: Tuple[bool, ...] = (True, True)
    top_blocks_dropout: Tuple[float, ...] = (0.0, 0.0)
    mid_blocks_channels: Tuple[int, ...] = (256, 512)
    mid_blocks_n_blocks_per_resolution: Tuple[int, ...] = (4, 4)
    mid_blocks_has_resampling: Tuple[bool, ...] = (True, False)
    mid_blocks_dropout: Tuple[float, ...] = (0.0, 0.0)


class ConsistencyUNet(nn.Module):
    """UNet with dual time-level embeddings for pairwise consistency models.
    
    Forward signature: forward(x, y, time, time_prime)
        - x: noisy sample (B, channels, H, W)
        - y: observations  (B, n_input_channels, H, W)
        - time: current time level (B,)
        - time_prime: target time level (B,)
    
    Input to the network is cat([x, y, mask_obs], dim=1) where 
    mask_obs = ~isnan(y), giving (B, channels + 2*n_input_channels, H, W) total
    input channels projected down to top_blocks_channels[0].
    """

    def __init__(self, config: ConsistencyUNetConfig):
        super().__init__()
        self.config = config
        n_input = config.n_input_channels if config.n_input_channels is not None else config.channels

        # Input: [x(channels), y(n_input), mask(n_input)]
        total_input_channels = config.channels + 2 * n_input

        self.input_projection = nn.Conv2d(
            total_input_channels, config.top_blocks_channels[0],
            kernel_size=3, padding="same",
        )
        self.time_level_embedding = TimeLevelEmbedding(
            config.time_level_channels, config.time_level_scale
        )
        self.top_encoder_blocks = self._make_encoder_blocks(
            config.top_blocks_channels + config.mid_blocks_channels[:1],
            config.top_blocks_n_blocks_per_resolution,
            config.top_blocks_has_resampling,
            config.top_blocks_dropout,
            self._make_top_block,
        )
        self.mid_encoder_blocks = self._make_encoder_blocks(
            config.mid_blocks_channels + config.mid_blocks_channels[-1:],
            config.mid_blocks_n_blocks_per_resolution,
            config.mid_blocks_has_resampling,
            config.mid_blocks_dropout,
            self._make_mid_block,
        )
        self.mid_decoder_blocks = self._make_decoder_blocks(
            config.mid_blocks_channels + config.mid_blocks_channels[-1:],
            config.mid_blocks_n_blocks_per_resolution,
            config.mid_blocks_has_resampling,
            config.mid_blocks_dropout,
            self._make_mid_block,
        )
        self.top_decoder_blocks = self._make_decoder_blocks(
            config.top_blocks_channels + config.mid_blocks_channels[:1],
            config.top_blocks_n_blocks_per_resolution,
            config.top_blocks_has_resampling,
            config.top_blocks_dropout,
            self._make_top_block,
        )
        self.output_projection = nn.Conv2d(
            config.top_blocks_channels[0], config.channels,
            kernel_size=3, padding="same",
        )

    @staticmethod
    def _pad_to_multiple(t: Tensor, multiple: int = 8):
        """Pad H and W to the nearest multiple of `multiple` (reflect padding)."""
        _, _, H, W = t.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        return F.pad(t, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)

    @staticmethod
    def _unpad(t: Tensor, pad_h: int, pad_w: int) -> Tensor:
        H, W = t.shape[-2], t.shape[-1]
        return t[..., : H - pad_h if pad_h else H, : W - pad_w if pad_w else W]

    def forward(self, x: Tensor, y: Tensor, time: Tensor, time_prime: Tensor) -> Tensor:
        """
        Args:
            x: noisy target (B, channels, H, W)
            y: observations (B, n_input_channels, H, W) — may contain NaN
            time: current noise level (B,)
            time_prime: target noise level (B,)
        Returns:
            Raw network output (B, channels, H, W) — before skip/output scaling
        """
        mask_obs = ~torch.isnan(y)
        y_clean = torch.nan_to_num(y)
        inp = torch.cat((x, y_clean, mask_obs.float()), dim=1)

        # Pad to multiple of 8 so all Downsample layers (ph=2,pw=2) can divide evenly
        n_downsamples = sum(
            1 for blocks in (self.top_encoder_blocks, self.mid_encoder_blocks)
            for b in blocks if isinstance(b, Downsample)
        )
        multiple = 2 ** n_downsamples
        inp, (pad_h, pad_w) = self._pad_to_multiple(inp, multiple)

        h = self.input_projection(inp)

        emb1 = self.time_level_embedding(time)
        emb2 = self.time_level_embedding(time_prime)
        time_level = torch.cat([emb1, emb2], dim=1)

        top_encoder_embeddings = []
        for block in self.top_encoder_blocks:
            if isinstance(block, UNetBlock):
                h = block(h, time_level)
                top_encoder_embeddings.append(h)
            else:
                h = block(h)

        mid_encoder_embeddings = []
        for block in self.mid_encoder_blocks:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = block(h, time_level)
                mid_encoder_embeddings.append(h)
            else:
                h = block(h)

        for block in self.mid_decoder_blocks:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = torch.cat((h, mid_encoder_embeddings.pop()), dim=1)
                h = block(h, time_level)
            else:
                h = block(h)

        for block in self.top_decoder_blocks:
            if isinstance(block, UNetBlock):
                h = torch.cat((h, top_encoder_embeddings.pop()), dim=1)
                h = block(h, time_level)
            else:
                h = block(h)

        return self._unpad(self.output_projection(h), pad_h, pad_w)

    # ── Encoder / Decoder block construction ──────────────────────────

    def _make_encoder_blocks(self, channels, n_blocks_per_resolution,
                             has_resampling, dropout, block_fn):
        blocks = nn.ModuleList()
        channel_pairs = list(zip(channels[:-1], channels[1:]))
        for idx, (in_ch, out_ch) in enumerate(channel_pairs):
            for _ in range(n_blocks_per_resolution[idx]):
                blocks.append(block_fn(in_ch, out_ch, dropout[idx]))
                in_ch = out_ch
            if has_resampling[idx]:
                blocks.append(Downsample(out_ch))
        return blocks

    def _make_decoder_blocks(self, channels, n_blocks_per_resolution,
                             has_resampling, dropout, block_fn):
        blocks = nn.ModuleList()
        channel_pairs = list(zip(channels[:-1], channels[1:]))[::-1]
        for idx, (out_ch, in_ch) in enumerate(channel_pairs):
            if has_resampling[::-1][idx]:
                blocks.append(Upsample(in_ch))
            inner = []
            for _ in range(n_blocks_per_resolution[::-1][idx]):
                inner.append(block_fn(in_ch * 2, out_ch, dropout[::-1][idx]))
                out_ch = in_ch
            blocks.extend(inner[::-1])
        return blocks

    def _make_top_block(self, in_ch, out_ch, dropout):
        return UNetBlock(in_ch, out_ch, 2 * self.config.time_level_channels, dropout)

    def _make_mid_block(self, in_ch, out_ch, dropout):
        return UNetBlockWithSelfAttention(
            in_ch, out_ch, 2 * self.config.time_level_channels,
            self.config.n_heads, dropout,
        )

    # ── Serialization ─────────────────────────────────────────────────

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(asdict(self.config), f)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str) -> "ConsistencyUNet":
        with open(os.path.join(path, "config.json"), "r") as f:
            cfg = ConsistencyUNetConfig(**json.load(f))
        model = cls(cfg)
        model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        )
        return model


# ──────────────────────────────────────────────────────────────────────
# Consistency forward wrapper (skip + output scaling)
# ──────────────────────────────────────────────────────────────────────

def compute_sigma(t: Tensor, sigma_min: float, sigma_max: float, rho: float = 7.0) -> Tensor:
    rho_inv = 1.0 / rho
    return (sigma_min ** rho_inv + t * (sigma_max ** rho_inv - sigma_min ** rho_inv)) ** rho


def skip_scaling(sigma: Tensor, sigma_data: float, sigma_min: float) -> Tensor:
    return sigma_data ** 2 / ((sigma - sigma_min) ** 2 + sigma_data ** 2)


def output_scaling(sigma: Tensor, sigma_data: float, sigma_min: float) -> Tensor:
    return (sigma_data * (sigma - sigma_min)) / (sigma_data ** 2 + sigma ** 2) ** 0.5


def pad_dims_like(x: Tensor, other: Tensor) -> Tensor:
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))


def consistency_forward_wrapper(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    t1: Tensor,
    t2: Tensor,
    sigma_data: float = 1.0,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    **kwargs,
) -> Tensor:
    """Apply consistency model skip / output scaling around a raw UNet call.
    
    f_θ(x, t1→t2) = c_skip(σ₁) · x + c_out(σ₁) · F_θ(x, y, t1, t2)
    """
    sigma1 = compute_sigma(t1, sigma_min, sigma_max)
    c_skip = pad_dims_like(skip_scaling(sigma1, sigma_data, sigma_min), x)
    c_out = pad_dims_like(output_scaling(sigma1, sigma_data, sigma_min), x)
    return c_skip * x + c_out * model(x, y, t1, t2, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# CROSCIM-compatible solver wrapper
# ──────────────────────────────────────────────────────────────────────

class ConsistencyUNetSolver(nn.Module):
    """Wraps a ConsistencyUNet so it can be used as a drop-in solver
    inside CROSCIM's GradSolvers (nn.ModuleDict keyed by resolution).
    
    In *training* mode the forward is NOT called through this wrapper;
    instead the LightningModule drives student/teacher calls explicitly.
    
    In *test* mode (inference / sampling), this wrapper runs iterative
    consistency sampling to produce a prediction from the sBatch.
    
        Args:
            unet_config: ConsistencyUNetConfig for building the UNet.
            n_input_channels: Total observation channels (n_input_vars * n_time).
            n_output_channels: Total target channels (n_target_vars * n_time).
            sigma_min, sigma_max, rho, sigma_data: Noise schedule parameters.
            sampling_steps: Number of denoising steps at inference.
            stochastic: If True, use reverse-SDE sampling (re-noise between steps
                for ensemble diversity). If False (default), use deterministic
                probability-flow ODE sampling.
    """

    def __init__(
        self,
        unet_config: ConsistencyUNetConfig = None,
        n_input_channels: int = None,
        n_output_channels: int = None,
        n_hidden: int = 128,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 1.0,
        sampling_steps: int = 15,
        stochastic: bool = False,
    ):
        super().__init__()

        # Build config if not provided
        if unet_config is None:
            unet_config = ConsistencyUNetConfig(
                channels=n_output_channels,
                n_input_channels=n_input_channels,
                top_blocks_channels=(n_hidden, n_hidden),
                mid_blocks_channels=(n_hidden * 2, n_hidden * 4),
            )
        self.unet_config = unet_config
        self.unet = ConsistencyUNet(unet_config)

        self.n_input_channels = n_input_channels or unet_config.n_input_channels or unet_config.channels
        self.n_output_channels = n_output_channels or unet_config.channels

        # Noise schedule hyper-parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.sampling_steps = sampling_steps
        self.stochastic = stochastic

    # ── Raw UNet access (for training) ────────────────────────────────

    def get_unet(self) -> ConsistencyUNet:
        return self.unet

    # ── Inference: consistency sampling from sBatch ───────────────────

    @torch.no_grad()
    def forward(self, batch):
        """Consistency sampling from an sBatch(input, tgt).
        
        batch.input has shape (B, C_in, H, W) where C_in = n_input_vars * n_time.
        The solver produces output of shape (B, C_out, H, W).
        
        During inference we:
        1. Extract observations y = batch.input (the full obs tensor).
        2. Start from pure noise scaled to sigma(T).
        3. Iteratively denoise using the consistency model.

        When ``self.stochastic=True``, reverse-SDE sampling is used: after each
        denoising step the prediction is re-noised to the sigma level of the next
        step, increasing sample diversity (Song et al. 2023, Algorithm 2).
        When ``self.stochastic=False`` (default), deterministic probability-flow
        ODE sampling is used.
        """
        y = batch.input
        B, _, H, W = y.shape
        device = y.device
        dtype = y.dtype

        # Karras schedule (descending in time: T → 0)
        nsteps = self.sampling_steps
        steps = torch.arange(nsteps, device=device, dtype=dtype) / max(nsteps - 1, 1)
        # steps goes 0 → 1; we flip for sampling (1 → 0)
        times = torch.flip(steps, dims=[0])

        # Initial noise at t = times[0] ≈ 1
        sigma_init = compute_sigma(times[0], self.sigma_min, self.sigma_max)
        x = torch.randn(B, self.n_output_channels, H, W, device=device, dtype=dtype) * sigma_init

        for i in range(nsteps - 1):
            t_curr = torch.full((B,), times[i].item(), device=device, dtype=dtype)
            t_next = torch.full((B,), times[i + 1].item(), device=device, dtype=dtype)

            # ── Denoising step (ODE or SDE) ──────────────────────────────────
            x = consistency_forward_wrapper(
                self.unet, x, y, t_curr, t_next,
                self.sigma_data, self.sigma_min, self.sigma_max,
            )

            # ── SDE: re-noise to sigma level of the *next* step ──────────────
            # x̂ = f_θ(x_noisy) is our denoised estimate.  We then add noise
            # scaled to σ(t_{i+2}) before the following denoising call,
            # following the stochastic consistency sampling of Song et al. 2023.
            if self.stochastic and i < nsteps - 2:
                sigma_next = compute_sigma(times[i + 1], self.sigma_min, self.sigma_max)
                x = x + sigma_next * torch.randn_like(x)

        return x


class ConsistencyGradSolvers(nn.Module):
    """Drop-in replacement for GradSolvers that holds per-resolution
    ConsistencyUNetSolver instances.
    
    Instantiation via Hydra:
        _target_: contrib.CROSCIM.consistency_solver.ConsistencyGradSolvers
        solvers:
          solver_x50:
            _target_: contrib.CROSCIM.consistency_solver.ConsistencyUNetSolver
            ...
          solver_x10:
            _target_: contrib.CROSCIM.consistency_solver.ConsistencyUNetSolver
            ...
    """

    def __init__(self, solvers, **kwargs):
        super().__init__()
        self.solvers = nn.ModuleDict(solvers)

    def forward(self, batch, res=1, stochastic: bool = None):
        """Run the solver for the given resolution.

        Args:
            batch: sBatch with `input` tensor.
            res: Resolution key (e.g. 50 → ``solver_x50``).
            stochastic: Override the solver's ``stochastic`` flag for this call.
                If None (default), use the flag set at construction time.
        """
        solver = self.solvers[f"solver_x{res}"]
        if stochastic is not None:
            # Temporarily override stochastic flag without modifying the module
            orig = solver.stochastic
            solver.stochastic = stochastic
            out = solver(batch)
            solver.stochastic = orig
            return out
        return solver(batch)
