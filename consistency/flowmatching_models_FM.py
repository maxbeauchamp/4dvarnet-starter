"""
Flow Matching model for the SPDE / notebook framework.

Architecture
------------
* ``UNetFM``         – same UNet as the CM notebook, but conditioned on a single
                       pseudo-time t ∈ [0, 1] (velocity prediction) instead of the
                       dual (σ, σ') Karras embeddings used by the CM.
* ``LogScaleModel``  – lightweight heteroscedastic log-scale model conditioned on t.
* ``euler_sample``   – deterministic Euler ODE integration (1st order).
* ``heun_sample``    – deterministic Heun (2nd-order) ODE integration.
* ``LitFlowMatchingModel`` – LightningModule with:
    - linear interpolant FM loss  (x_t = t·x1 + (1-t)·ε, v = x1 - ε)
    - ``LogScaleModel``-weighted NegLogPDF loss
    - EMA via ``torch.optim.swa_utils.AveragedModel``
    - on_train_epoch_end validation with Heun ODE sampling

Training loss
-------------
    x_t   = t · x_1  +  (1-t) · ε          (linear interpolant)
    v_tgt = x_1 - ε                         (target velocity)
    s     = log_scale_model(t)               (per-channel learned log-scale)
    loss  = mean[ (v_tgt - v_pred)² / (2·exp(2s))  +  s ]
          = mean[ neglogpdf( (v_tgt-v_pred)/exp(s), s ) ]
          (masked to observed pixels only)

Inference
---------
Starting from ε ~ N(0,I) at t=0, integrate the ODE dx/dt = v_θ(x_t, y, t)
from t=0 to t=1 using Heun's method with ``n_steps`` sub-steps.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from pytorch_lightning import LightningModule
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions  (self-contained, no gensim dependency)
# ─────────────────────────────────────────────────────────────────────────────

_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


def neglogpdf(value: Tensor, log_scale: Tensor) -> Tensor:
    """NegLogPDF of N(0,1) scaled by exp(log_scale):
        -log p(x) = 0.5·x² + log_scale + log(√2π)
    """
    return 0.5 * value.pow(2) + log_scale + _LOG_SQRT_2PI


def neglogcdf(value: Tensor) -> Tensor:
    """Negative log CDF of N(0,1): -log Φ(value)."""
    return -torch.special.log_ndtr(value)


def sample_uniform_time(template: Tensor) -> Tensor:
    """Sample equidistant pseudo-times in [0,1] for the batch.

    Uses the Kingma et al. (2021) low-discrepancy trick: a single uniform
    shift is added to an equidistant grid and wrapped mod 1.

    Returns a (B, 1, 1, 1) tensor on the same device/dtype as *template*.
    """
    B = template.size(0)
    shift = torch.rand(1, dtype=template.dtype, device=template.device)
    t = (shift + torch.linspace(0, 1, B + 1, dtype=template.dtype,
                                 device=template.device)[:B]) % 1.0
    return t.view(B, 1, 1, 1)


def masked_average(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of *x* restricted to pixels where *mask* is True (or > 0)."""
    m = mask.to(dtype=x.dtype).expand_as(x)
    return (x * m).sum() / m.sum().clamp(min=1)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks  (identical to CM notebook)
# ─────────────────────────────────────────────────────────────────────────────

def GroupNorm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, channels // 4), num_channels=channels)


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 n_heads: int = 8, dropout: float = 0.3) -> None:
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
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        out = self.output_projection(out)
        out = rearrange(out, "b c (x y) -> b c x y", x=x.shape[-2], y=x.shape[-1])
        return out + self.residual_projection(x)


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 time_channels: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            GroupNorm(in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(time_channels, out_channels, kernel_size=1),
        )
        self.output_proj = nn.Sequential(
            GroupNorm(out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.input_proj(x)
        h = h + self.time_proj(t_emb)
        return self.output_proj(h) + self.residual_proj(x)


class UNetBlockWithSelfAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 time_channels: int, n_heads: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        self.unet_block = UNetBlock(in_channels, out_channels, time_channels, dropout)
        self.self_attention = SelfAttention(out_channels, out_channels, n_heads, dropout)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        return self.self_attention(self.unet_block(x, t_emb))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            Rearrange("b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2),
            nn.Conv2d(4 * channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class TimeEmbedding(nn.Module):
    """Fourier embedding for a scalar pseudo-time t ∈ [0, 1] → (B, channels, 1, 1)."""

    def __init__(self, channels: int, scale: float = 16.0) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)
        self.proj = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
            Rearrange("b c -> b c () ()"),
        )

    def forward(self, t: Tensor) -> Tensor:
        # t : (B,) scalar in [0, 1]
        h = t[:, None] * self.W[None, :] * 2 * math.pi
        h = torch.cat([h.sin(), h.cos()], dim=-1)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# Padding utilities
# ─────────────────────────────────────────────────────────────────────────────

def _pad_to_multiple(x: Tensor, multiple: int = 8) -> Tuple[Tensor, Tuple[int, int, int, int]]:
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    padding = (0, pad_w, 0, pad_h)
    return F.pad(x, padding, mode="reflect"), padding


def _unpad(x: Tensor, padding: Tuple[int, int, int, int]) -> Tensor:
    _, pad_w, _, pad_h = padding
    H, W = x.shape[-2], x.shape[-1]
    return x[..., : H - pad_h if pad_h else H, : W - pad_w if pad_w else W]


# ─────────────────────────────────────────────────────────────────────────────
# UNetFM — velocity-prediction network
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UNetFMConfig:
    channels: int = 5                                # = window_size C
    cond_channels: Optional[int] = None              # conditioning channels (default = channels)
    time_channels: int = 128
    time_scale: float = 16.0
    n_heads: int = 8
    top_blocks_channels: Tuple[int, ...] = (64, 64)
    top_blocks_n_blocks: Tuple[int, ...] = (2, 2)
    top_blocks_resampling: Tuple[bool, ...] = (True, True)
    top_blocks_dropout: Tuple[float, ...] = (0.0, 0.0)
    mid_blocks_channels: Tuple[int, ...] = (128, 256)
    mid_blocks_n_blocks: Tuple[int, ...] = (4, 4)
    mid_blocks_resampling: Tuple[bool, ...] = (True, False)
    mid_blocks_dropout: Tuple[float, ...] = (0.0, 0.0)


class UNetFM(nn.Module):
    """UNet conditioned on pseudo-time t for velocity-field prediction.

    Input : cat(x_t, y_filled, obs_mask)  →  (B, 3*C, H, W)
    Output: predicted velocity v           →  (B, C, H, W)

    The single pseudo-time embedding replaces the dual (σ, σ') embeddings
    used by the CM UNet.
    """

    def __init__(self, config: UNetFMConfig) -> None:
        super().__init__()
        self.config = config
        tc = config.time_channels

        cc = config.cond_channels if config.cond_channels is not None else config.channels
        self.input_proj = nn.Conv2d(
            config.channels + 2 * cc, config.top_blocks_channels[0],
            kernel_size=3, padding="same",
        )
        self.time_emb = TimeEmbedding(tc, config.time_scale)

        self.top_enc = self._make_enc(
            config.top_blocks_channels + config.mid_blocks_channels[:1],
            config.top_blocks_n_blocks, config.top_blocks_resampling,
            config.top_blocks_dropout, self._top_block,
        )
        self.mid_enc = self._make_enc(
            config.mid_blocks_channels + config.mid_blocks_channels[-1:],
            config.mid_blocks_n_blocks, config.mid_blocks_resampling,
            config.mid_blocks_dropout, self._mid_block,
        )
        self.mid_dec = self._make_dec(
            config.mid_blocks_channels + config.mid_blocks_channels[-1:],
            config.mid_blocks_n_blocks, config.mid_blocks_resampling,
            config.mid_blocks_dropout, self._mid_block,
        )
        self.top_dec = self._make_dec(
            config.top_blocks_channels + config.mid_blocks_channels[:1],
            config.top_blocks_n_blocks, config.top_blocks_resampling,
            config.top_blocks_dropout, self._top_block,
        )
        self.output_proj = nn.Conv2d(
            config.top_blocks_channels[0], config.channels,
            kernel_size=3, padding="same",
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x_t: Tensor, y: Tensor, t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x_t : (B, C, H, W)  noisy / interpolated state at pseudo-time t
        y   : (B, C, H, W)  observations (NaN where missing)
        t   : (B,)           pseudo-time in [0, 1]

        Returns
        -------
        v : (B, C, H, W)  predicted velocity
        """
        mask = (~torch.isnan(y)).to(x_t.dtype)
        y_c  = torch.nan_to_num(y, nan=0.0)

        inp = torch.cat((x_t, y_c, mask), dim=1)    # (B, 3C, H, W)
        inp, pad = _pad_to_multiple(inp, 8)

        h     = self.input_proj(inp)
        t_emb = self.time_emb(t)                    # (B, tc, 1, 1)

        top_skips, mid_skips = [], []

        for block in self.top_enc:
            if isinstance(block, UNetBlock):
                h = block(h, t_emb); top_skips.append(h)
            else:
                h = block(h)

        for block in self.mid_enc:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = block(h, t_emb); mid_skips.append(h)
            else:
                h = block(h)

        for block in self.mid_dec:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = torch.cat((h, mid_skips.pop()), dim=1)
                h = block(h, t_emb)
            else:
                h = block(h)

        for block in self.top_dec:
            if isinstance(block, UNetBlock):
                h = torch.cat((h, top_skips.pop()), dim=1)
                h = block(h, t_emb)
            else:
                h = block(h)

        return _unpad(self.output_proj(h), pad)

    # ── builder helpers ────────────────────────────────────────────────────────

    def _top_block(self, ic, oc, dropout):
        return UNetBlock(ic, oc, self.config.time_channels, dropout)

    def _mid_block(self, ic, oc, dropout):
        return UNetBlockWithSelfAttention(
            ic, oc, self.config.time_channels, self.config.n_heads, dropout
        )

    def _make_enc(self, channels, n_blocks, resampling, dropout, block_fn):
        blocks = nn.ModuleList()
        for idx, (ic, oc) in enumerate(zip(channels[:-1], channels[1:])):
            for _ in range(n_blocks[idx]):
                blocks.append(block_fn(ic, oc, dropout[idx])); ic = oc
            if resampling[idx]:
                blocks.append(Downsample(oc))
        return blocks

    def _make_dec(self, channels, n_blocks, resampling, dropout, block_fn):
        blocks = nn.ModuleList()
        for idx, (oc, ic) in enumerate(list(zip(channels[:-1], channels[1:]))[::-1]):
            if resampling[::-1][idx]:
                blocks.append(Upsample(ic))
            inner = []
            for _ in range(n_blocks[::-1][idx]):
                inner.append(block_fn(ic * 2, oc, dropout[::-1][idx])); oc = ic
            blocks.extend(inner[::-1])
        return blocks

    # ── serialization ─────────────────────────────────────────────────────────

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(asdict(self.config), f)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str) -> "UNetFM":
        with open(os.path.join(path, "config.json")) as f:
            cfg = UNetFMConfig(**json.load(f))
        model = cls(cfg)
        model.load_state_dict(
            torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        )
        return model


# ─────────────────────────────────────────────────────────────────────────────
# LogScaleModel — heteroscedastic loss scale  (depends only on t)
# ─────────────────────────────────────────────────────────────────────────────

class LogScaleModel(nn.Module):
    """Lightweight per-channel log-scale model conditioned on pseudo-time t.

    f(t) → (B, n_vars)  — learned log-scale per output channel.
    Initialised at zero so training starts with a standard Gaussian loss.
    """

    def __init__(self, n_embedding: int = 128, n_vars: int = 5,
                 scale: float = 16.0) -> None:
        super().__init__()
        half = n_embedding // 2
        self.register_buffer(
            "freqs", torch.randn(half) * scale
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embedding, n_embedding),
            nn.SiLU(),
            nn.Linear(n_embedding, n_vars),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t: Tensor) -> Tensor:
        """t : (B,) → log_scale : (B, n_vars)"""
        h = t[:, None] * self.freqs[None, :] * 2 * math.pi
        h = torch.cat([h.sin(), h.cos()], dim=-1)  # (B, n_embedding)
        return self.mlp(h)                          # (B, n_vars)


# ─────────────────────────────────────────────────────────────────────────────
# ODE samplers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def euler_sample(
    model: UNetFM,
    noise: Tensor,
    y: Tensor,
    n_steps: int = 20,
) -> Tuple[Tensor, list[Tensor]]:
    """1st-order Euler ODE integration from t=0 to t=1.

    Returns (x_1, trajectory) where trajectory is a list of intermediate states.
    """
    dt = 1.0 / n_steps
    x  = noise.clone()
    traj = [x.clone()]
    for i in range(n_steps):
        t_val = i * dt
        t     = torch.full((x.size(0),), t_val, device=x.device, dtype=x.dtype)
        v     = model(x, y, t)
        x     = x + dt * v
        traj.append(x.clone())
    return x, traj


@torch.no_grad()
def heun_sample(
    model: UNetFM,
    noise: Tensor,
    y: Tensor,
    n_steps: int = 20,
) -> Tuple[Tensor, list[Tensor]]:
    """2nd-order Heun ODE integration from t=0 to t=1.

    Returns (x_1, trajectory).
    """
    dt = 1.0 / n_steps
    x  = noise.clone()
    traj = [x.clone()]
    for i in range(n_steps):
        t0 = i * dt
        t1 = (i + 1) * dt
        t0_t = torch.full((x.size(0),), t0, device=x.device, dtype=x.dtype)
        t1_t = torch.full((x.size(0),), t1, device=x.device, dtype=x.dtype)

        v0 = model(x, y, t0_t)
        x1_pred = x + dt * v0                       # Euler prediction
        v1 = model(x1_pred, y, t1_t)
        x  = x + dt * 0.5 * (v0 + v1)              # Heun correction
        traj.append(x.clone())
    return x, traj


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LitFMConfig:
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor: float = 1e-5
    lr_scheduler_iters: int = 10_000
    ema_decay: float = 0.999
    eval_n_steps: int = 20          # Heun steps for val monitoring
    lower_bound: Optional[float] = None   # censoring lower bound (normalised)
    upper_bound: Optional[float] = None   # censoring upper bound (normalised)


class LitFlowMatchingModel(LightningModule):
    """Flow Matching Lightning module for the SPDE dataset.

    Parameters
    ----------
    network     : ``UNetFM`` (student / live model)
    config      : ``LitFMConfig``
    n_log_scale_embedding : embedding dim for the ``LogScaleModel``
    """

    def __init__(
        self,
        network: UNetFM,
        config: LitFMConfig,
        n_log_scale_embedding: int = 128,
    ) -> None:
        super().__init__()
        self.network = network
        self.config  = config

        C = network.config.channels
        self.log_scale = LogScaleModel(n_log_scale_embedding, n_vars=C)

        # EMA model (frozen, updated after each training step)
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            network,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema_decay),
            device="cpu",
        )
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self._val_batch = None   # filled in on_fit_start

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_fit_start(self) -> None:
        try:
            dl = self.trainer.datamodule.val_dataloader()
            batch = next(iter(dl))
            if isinstance(batch, list):
                batch = batch[0]
            self._val_batch = batch
            print(f"✅ Val batch cached: tgt={batch.tgt.shape}")
        except Exception as e:
            print(f"⚠️  Could not cache val batch: {e}")

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.ema_model.update_parameters(self.network)

    # ── training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx: int):
        if isinstance(batch, list):
            batch = batch[0]

        x1 = batch.tgt                                      # (B, C, H, W) clean
        y  = batch.input                                    # (B, C, H, W) obs (NaN)
        B, C, H, W = x1.shape

        # ── linear interpolant ────────────────────────────────────────────
        t      = sample_uniform_time(x1)                   # (B, 1, 1, 1)
        t_flat = t.view(B)                                  # (B,)
        noise  = torch.randn_like(x1)
        x_t    = t * x1 + (1.0 - t) * noise
        v_tgt  = x1 - noise

        # ── velocity prediction ────────────────────────────────────────────
        v_pred = self.network(x_t, y, t_flat)               # (B, C, H, W)

        # ── log-scale & loss ──────────────────────────────────────────────
        log_s  = self.log_scale(t_flat)                     # (B, C)
        log_s  = log_s.view(B, C, 1, 1).expand_as(v_pred)

        residual = (v_tgt - v_pred) / log_s.exp().clamp(min=1e-5)
        loss_map = neglogpdf(residual, log_s)               # (B, C, H, W)

        # Optional censoring at physical bounds
        if self.config.lower_bound is not None:
            lb = torch.full_like(x1, self.config.lower_bound)
            censor = (x1 <= lb).float()
            loss_map = loss_map * (1 - censor) + censor * neglogcdf(-residual)
        if self.config.upper_bound is not None:
            ub = torch.full_like(x1, self.config.upper_bound)
            censor = (x1 >= ub).float()
            loss_map = loss_map * (1 - censor) + censor * neglogcdf(residual)

        # FM supervises velocity everywhere; obs conditioning is via the input (y_filled, mask).
        # Masking the loss to obs pixels only kills gradient at unobserved locations → RMSE ≈ 1.
        loss = loss_map.mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ── validation monitoring ─────────────────────────────────────────────────

    def on_train_epoch_end(self) -> None:
        if self._val_batch is None:
            return

        device = self.device
        dtype  = next(self.network.parameters()).dtype
        x_gt   = self._val_batch.tgt.to(device=device, dtype=dtype)
        y      = self._val_batch.input.to(device=device, dtype=dtype)
        B, C, H, W = x_gt.shape

        ema_net = self.ema_model.module.to(device=device, dtype=dtype)
        ema_net.eval()

        noise   = torch.randn_like(x_gt)
        x_hat, _ = heun_sample(ema_net, noise, y, n_steps=self.config.eval_n_steps)

        rmse = (x_hat.float() - x_gt.float()).pow(2).mean().sqrt().item()
        print(
            f"[Epoch {self.current_epoch}]  "
            f"Val RMSE ({self.config.eval_n_steps} Heun steps, EMA) = {rmse:.4f}"
        )
        self.log("val_rmse_ema", rmse)

    # ── optimizers ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        params = list(self.network.parameters()) + list(self.log_scale.parameters())
        opt    = torch.optim.Adam(params, lr=self.config.lr, betas=self.config.betas)
        sched  = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.config.lr_scheduler_start_factor,
            total_iters=self.config.lr_scheduler_iters,
        )
        return [opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]
