"""
Spatial Graph Neural Network backbone for CROSCIM.

Architecture
------------
The network operates on a regular (H×W) grid where some pixels are invalid
(land, out-of-domain).  Instead of ignoring the grid structure, we treat it
as a *graph*: valid pixels are nodes, 8-connected valid neighbours are edges.

Message passing is implemented efficiently in pure PyTorch via F.pad + manual
3×3 neighbourhood unfolding — no PyTorch-Geometric dependency.

For each node:
  1. Gather features from its (up to 8) valid neighbours.
  2. Compute multi-head attention scores over the neighbourhood.
  3. Aggregate neighbour messages using the attention weights.
  4. Apply a residual connection + LayerNorm.
  5. Pass through a position-wise FFN.

Positional encoding is injected from lat/lon (or from normalised pixel
coordinates as a fallback).

Pooling
-------
By default (``n_levels=1``) the whole grid is kept at full resolution
throughout, which means every GNNBlock attends over the entire H×W node set
(e.g. ~65k nodes for a 256×256 patch) with only a 3×3 receptive field per
layer — the receptive field grows linearly with depth, unlike a UNet's
exponential growth via downsampling. Setting ``n_levels>1`` turns the stack
into a Graph-UNet: ``n_levels`` steps of (GNNBlock(s) at current resolution
→ strided-conv downsample ×2), a bottleneck at the coarsest resolution, then
``n_levels`` steps of (upsample ×2 → fuse with the matching skip connection
→ GNNBlock(s)). This both shrinks the node count at deeper layers and lets
information cross large physical distances in few hops. ``n_levels=1`` is
the exact previous flat behaviour (checkpoint-compatible).

Classes
-------
PositionalEncoding2D       Lat/lon → learnable embedding (B, pos_dim, H, W)
SpatialGraphAttention      One graph-attention message-passing step
GNNBlock                   Attention + FFN with residual
GNNDownBlock                GNNBlock(s) + strided-conv downsample (Graph-UNet encoder step)
GNNUpBlock                  Upsample + skip fusion + GNNBlock(s) (Graph-UNet decoder step)
MultiResGridGNN             Full encoder-decoder pipeline (flat, or Graph-UNet if n_levels>1)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """
    Encode (lat, lon) coordinates as a learnable dense embedding.

    Args:
        embed_dim  : output embedding dimension (split equally between lat & lon)
        scale      : Fourier frequency scale for sinusoidal projection
    """

    def __init__(self, embed_dim: int, scale: float = 10.0) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        half = embed_dim // 2
        # Random Fourier features (frozen) → learnable linear projection
        self.register_buffer("W_lat", torch.randn(half // 2) * scale)
        self.register_buffer("W_lon", torch.randn(half // 2) * scale)
        self.lat_proj = nn.Linear(half, half)
        self.lon_proj = nn.Linear(half, half)

    def _rff(self, x: Tensor, W: Tensor) -> Tensor:
        """Random Fourier Features: (B, N) → (B, N, 2*len(W))."""
        h = x.unsqueeze(-1) * W.unsqueeze(0) * 2.0 * math.pi  # (B, N, K)
        return torch.cat([torch.sin(h), torch.cos(h)], dim=-1)  # (B, N, 2K)

    def forward(self, lat: Tensor, lon: Tensor) -> Tensor:
        """
        Args:
            lat, lon : (B, H, W)  — raw geographic coordinates
        Returns:
            (B, embed_dim, H, W)
        """
        B, H, W = lat.shape
        lat_f = lat.reshape(B, H * W)  # (B, N)
        lon_f = lon.reshape(B, H * W)

        lat_e = self.lat_proj(self._rff(lat_f, self.W_lat))  # (B, N, half)
        lon_e = self.lon_proj(self._rff(lon_f, self.W_lon))  # (B, N, half)

        pe = torch.cat([lat_e, lon_e], dim=-1)              # (B, N, embed_dim)
        return pe.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, embed_dim, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Core message-passing layer
# ─────────────────────────────────────────────────────────────────────────────

# 3×3 kernel offsets (row, col) — 9 positions including centre
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
            ( 0, -1), ( 0, 0), ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)]


def _gather_3x3(x: Tensor, valid_mask: Tensor):
    """
    Gather 3×3 neighbourhood for every pixel.

    Args:
        x          : (B, C, H, W)  — feature map, 0 on invalid pixels
        valid_mask : (B, 1, H, W)  — float, 1 = valid node

    Returns:
        neigh  : (B, H, W, 9, C)   — neighbour features
        nmask  : (B, H, W, 9, 1)   — 1 where neighbour is valid
    """
    B, C, H, W = x.shape
    x_pad = F.pad(x,            (1, 1, 1, 1), "constant", 0.0)  # (B, C, H+2, W+2)
    m_pad = F.pad(valid_mask,   (1, 1, 1, 1), "constant", 0.0)  # (B, 1, H+2, W+2)

    neigh_f, neigh_m = [], []
    for di, dj in _OFFSETS:
        neigh_f.append(x_pad[:, :, 1 + di: 1 + di + H, 1 + dj: 1 + dj + W])
        neigh_m.append(m_pad[:, :, 1 + di: 1 + di + H, 1 + dj: 1 + dj + W])

    # (B, 9, C, H, W) → (B, H, W, 9, C)
    neigh = torch.stack(neigh_f, dim=1).permute(0, 3, 4, 1, 2)
    nmask = torch.stack(neigh_m, dim=1).permute(0, 3, 4, 1, 2)
    return neigh, nmask


class SpatialGraphAttention(nn.Module):
    """
    One graph-attention message-passing step on the 8-connected grid graph.

    For each valid pixel *i*:
      - collect features from valid neighbours (3×3 patch, 9 positions)
      - compute per-head attention scores via a learned function of
        (center_feature, neighbour_feature)
      - aggregate value projections of neighbours with the attention weights
      - residual + LayerNorm

    Args:
        in_ch    : input channel dimension
        out_ch   : output channel dimension
        n_heads  : number of attention heads (out_ch must be divisible by n_heads)
        dropout  : dropout on attention weights
    """

    def __init__(self, in_ch: int, out_ch: int,
                 n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert out_ch % n_heads == 0, f"out_ch={out_ch} must be divisible by n_heads={n_heads}"
        self.n_heads  = n_heads
        self.head_dim = out_ch // n_heads

        # Attention: concat(centre, neighbour) → score per head per position
        self.attn_proj = nn.Linear(2 * in_ch, n_heads, bias=True)

        # Value: project each neighbour to out_ch
        self.val_proj  = nn.Linear(in_ch, out_ch, bias=False)

        # Output
        self.out_proj  = nn.Linear(out_ch, out_ch)
        self.norm      = nn.LayerNorm(out_ch)
        self.res_proj  = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        """
        Args:
            x          : (B, in_ch, H, W)
            valid_mask : (B, 1,     H, W)  float in {0, 1}

        Returns:
            (B, out_ch, H, W)
        """
        B, C, H, W = x.shape

        # Zero out invalid nodes so they send no information
        x_valid = x * valid_mask                                # (B, C, H, W)

        neigh, nmask = _gather_3x3(x_valid, valid_mask)
        # neigh : (B, H, W, 9, C)
        # nmask : (B, H, W, 9, 1)

        # Expand centre to match neighbourhood: (B, H, W, 9, C)
        center = x.permute(0, 2, 3, 1).unsqueeze(3).expand_as(neigh)

        # --- Attention ---
        msg_in = torch.cat([center, neigh], dim=-1)             # (B, H, W, 9, 2C)
        attn   = self.attn_proj(msg_in)                         # (B, H, W, 9, n_heads)
        # Mask invalid neighbours (-inf before softmax → 0 after)
        attn   = attn.masked_fill(nmask == 0, float("-inf"))
        attn   = F.softmax(attn, dim=3)                         # (B, H, W, 9, n_heads)
        attn   = torch.nan_to_num(attn, nan=0.0)                # all-invalid node → 0
        attn   = self.drop(attn)

        # --- Value aggregation ---
        vals = self.val_proj(neigh)                             # (B, H, W, 9, out_ch)
        vals = vals.view(B, H, W, 9, self.n_heads, self.head_dim)  # split heads

        # attn: (B, H, W, 9, n_heads) → unsqueeze last dim for broadcast
        attn_w = attn.unsqueeze(-1)                             # (B, H, W, 9, n_heads, 1)
        agg = (attn_w * vals).sum(dim=3)                        # (B, H, W, n_heads, head_dim)
        agg = agg.reshape(B, H, W, -1)                         # (B, H, W, out_ch)

        # --- Output + residual + norm ---
        out = self.drop(self.out_proj(agg))                     # (B, H, W, out_ch)
        res = self.res_proj(x.permute(0, 2, 3, 1))             # (B, H, W, out_ch)
        out = self.norm(out + res)                              # (B, H, W, out_ch)

        return out.permute(0, 3, 1, 2)                          # (B, out_ch, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# GNN block = attention + position-wise FFN
# ─────────────────────────────────────────────────────────────────────────────

class GNNBlock(nn.Module):
    """
    Graph attention + position-wise feed-forward network (FFN).

    Both sub-layers use residual connections and LayerNorm (Pre-LN style).
    """

    def __init__(self, in_ch: int, out_ch: int,
                 n_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.attn   = SpatialGraphAttention(in_ch, out_ch, n_heads, dropout)
        # Pre-LN FFN
        self.norm   = nn.LayerNorm(out_ch)
        self.ff     = nn.Sequential(
            nn.Linear(out_ch, ff_mult * out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * out_ch, out_ch),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        # Graph-attention step
        h = self.attn(x, valid_mask)                            # (B, out_ch, H, W)
        # FFN in channel-last format
        h2 = h.permute(0, 2, 3, 1)                             # (B, H, W, out_ch)
        h2 = h2 + self.ff(self.norm(h2))
        return h2.permute(0, 3, 1, 2)                          # (B, out_ch, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Graph-UNet encoder/decoder steps (used when n_levels > 1)
# ─────────────────────────────────────────────────────────────────────────────

class GNNDownBlock(nn.Module):
    """GNNBlock(s) at the current resolution, then strided-conv downsample ×2.

    Invalid nodes are zeroed before the strided conv so land does not leak
    into coarse cells; the mask itself is downsampled with max-pool (a coarse
    cell is valid if any of its 4 finer cells was valid).
    """

    def __init__(self, ch: int, n_heads: int, n_blocks: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            GNNBlock(ch, ch, n_heads=n_heads, ff_mult=4, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.down = nn.Conv2d(ch, ch, kernel_size=2, stride=2)

    def forward(self, x: Tensor, mask: Tensor):
        for blk in self.blocks:
            x = blk(x, mask)
        skip = x
        x_down = self.down(x * mask)
        mask_down = F.max_pool2d(mask, kernel_size=2, stride=2)
        return x_down, mask_down, skip


class GNNUpBlock(nn.Module):
    """Upsample ×2, fuse with the matching skip connection, then GNNBlock(s)."""

    def __init__(self, ch: int, n_heads: int, n_blocks: int, dropout: float) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(2 * ch, ch, kernel_size=1)
        self.blocks = nn.ModuleList([
            GNNBlock(ch, ch, n_heads=n_heads, ff_mult=4, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: Tensor, skip: Tensor, mask: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.fuse(torch.cat([x, skip], dim=1))
        for blk in self.blocks:
            x = blk(x, mask)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full GNN pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MultiResGridGNN(nn.Module):
    """
    Full GNN pipeline for spatially masked grid data.

    Flow
    ----
    Input (B, C_in, H, W)
        ↓  [optional positional encoding concat]
    Input projection  Conv1×1 → (B, hidden_dim, H, W)
        ↓  × n_layers
    GNN block (spatial graph attention + FFN)
        ↓
    Output projection  Conv1×1 → (B, C_out, H, W)
        ↓  apply land mask (→ NaN on invalid nodes)
    Output (B, C_out, H, W)

    Args:
        in_channels  : C_in  (n_input_vars × T)
        out_channels : C_out (n_target_vars × T)
        hidden_dim   : width of all internal GNN layers
        n_layers     : number of GNNBlock stacked (flat mode), or number of
                       GNNBlocks per resolution level (Graph-UNet mode)
        n_heads      : attention heads (hidden_dim must be divisible)
        pos_dim      : dimension of positional encoding (0 = disabled)
        dropout      : dropout rate throughout the network
        n_levels     : number of ×2 pooling levels. 1 (default) = flat,
                       full-resolution stack, identical to the previous
                       behaviour. >1 = Graph-UNet: n_levels encoder steps
                       (GNNBlock(s) + downsample ×2), a bottleneck, then
                       n_levels decoder steps (upsample ×2 + skip + GNNBlock(s)).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        hidden_dim:   int = 128,
        n_layers:     int = 6,
        n_heads:      int = 4,
        pos_dim:      int = 16,
        dropout:      float = 0.1,
        n_levels:     int = 1,
    ) -> None:
        super().__init__()
        self.pos_dim      = pos_dim
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_levels     = n_levels

        if pos_dim > 0:
            self.pos_enc = PositionalEncoding2D(pos_dim)

        total_in = in_channels + (pos_dim if pos_dim > 0 else 0)

        # Input projection
        self.input_proj = nn.Conv2d(total_in, hidden_dim, kernel_size=1)

        if n_levels <= 1:
            # Flat stack of GNN blocks (all same hidden_dim)
            self.gnn_blocks = nn.ModuleList([
                GNNBlock(hidden_dim, hidden_dim, n_heads=n_heads,
                         ff_mult=4, dropout=dropout)
                for _ in range(n_layers)
            ])
        else:
            # Graph-UNet: encoder / bottleneck / decoder
            self.down_blocks = nn.ModuleList([
                GNNDownBlock(hidden_dim, n_heads, n_layers, dropout)
                for _ in range(n_levels)
            ])
            self.bottleneck = nn.ModuleList([
                GNNBlock(hidden_dim, hidden_dim, n_heads=n_heads,
                         ff_mult=4, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.up_blocks = nn.ModuleList([
                GNNUpBlock(hidden_dim, n_heads, n_layers, dropout)
                for _ in range(n_levels)
            ])

        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def _make_fallback_pe(self, B: int, H: int, W: int, device) -> Tensor:
        """Sinusoidal position encoding from normalised pixel coordinates."""
        yy = torch.linspace(-1.0, 1.0, H, device=device)
        xx = torch.linspace(-1.0, 1.0, W, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        gy = gy.unsqueeze(0).expand(B, -1, -1)
        gx = gx.unsqueeze(0).expand(B, -1, -1)
        return self.pos_enc(gy, gx)                             # (B, pos_dim, H, W)

    def forward(
        self,
        x:          Tensor,
        valid_mask: Tensor,
        lat:        Optional[Tensor] = None,
        lon:        Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x          : (B, C_in, H, W) — input features, NaN on invalid nodes
            valid_mask : (B, 1, H, W)    — float 1=valid, 0=land
            lat        : (B, H, W)       — latitude  (optional)
            lon        : (B, H, W)       — longitude (optional)

        Returns:
            (B, C_out, H, W) — predictions, NaN on invalid nodes
        """
        B, _, H, W = x.shape
        device = x.device

        # Replace NaN with 0 (invalid nodes send/receive nothing)
        x_clean = torch.nan_to_num(x, nan=0.0)

        # Positional encoding
        if self.pos_dim > 0:
            if lat is not None and lon is not None:
                # Normalise coordinates to avoid extreme magnitudes
                lat_n = lat / 90.0   # [-1, 1]
                lon_n = lon / 180.0  # [-1, 1]
                pe = self.pos_enc(lat_n, lon_n)              # (B, pos_dim, H, W)
            else:
                pe = self._make_fallback_pe(B, H, W, device)
            x_clean = torch.cat([x_clean, pe], dim=1)

        # Input projection
        h = self.input_proj(x_clean)                         # (B, hidden_dim, H, W)

        if self.n_levels <= 1:
            # GNN blocks
            for block in self.gnn_blocks:
                h = block(h, valid_mask)
        else:
            # Graph-UNet: pad to a multiple of 2**n_levels so every pooling
            # step divides evenly, then crop back before the output projection.
            factor = 2 ** self.n_levels
            pad_h = (-H) % factor
            pad_w = (-W) % factor
            h = F.pad(h, (0, pad_w, 0, pad_h), "constant", 0.0)
            mask = F.pad(valid_mask, (0, pad_w, 0, pad_h), "constant", 0.0)

            skips, masks_at_level = [], [mask]
            for down in self.down_blocks:
                h, mask, skip = down(h, mask)
                skips.append(skip)
                masks_at_level.append(mask)

            for block in self.bottleneck:
                h = block(h, masks_at_level[-1])

            for level in reversed(range(self.n_levels)):
                h = self.up_blocks[level](h, skips[level], masks_at_level[level])

            h = h[:, :, :H, :W]                               # crop back padding

        # Output projection
        out = self.output_proj(h)                            # (B, C_out, H, W)

        # Re-apply land mask: land pixels become NaN in the output
        out = out.masked_fill(valid_mask.expand_as(out) == 0, float("nan"))

        return out
