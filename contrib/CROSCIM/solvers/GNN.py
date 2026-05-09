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

Classes
-------
PositionalEncoding2D       Lat/lon → learnable embedding (B, pos_dim, H, W)
SpatialGraphAttention      One graph-attention message-passing step
GNNBlock                   Attention + FFN with residual
MultiResGridGNN            Full encoder-decoder pipeline
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
        n_layers     : number of GNNBlock stacked
        n_heads      : attention heads (hidden_dim must be divisible)
        pos_dim      : dimension of positional encoding (0 = disabled)
        dropout      : dropout rate throughout the network
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
    ) -> None:
        super().__init__()
        self.pos_dim      = pos_dim
        self.in_channels  = in_channels
        self.out_channels = out_channels

        if pos_dim > 0:
            self.pos_enc = PositionalEncoding2D(pos_dim)

        total_in = in_channels + (pos_dim if pos_dim > 0 else 0)

        # Input projection
        self.input_proj = nn.Conv2d(total_in, hidden_dim, kernel_size=1)

        # Stack of GNN blocks (all same hidden_dim)
        self.gnn_blocks = nn.ModuleList([
            GNNBlock(hidden_dim, hidden_dim, n_heads=n_heads,
                     ff_mult=4, dropout=dropout)
            for _ in range(n_layers)
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

        # GNN blocks
        for block in self.gnn_blocks:
            h = block(h, valid_mask)

        # Output projection
        out = self.output_proj(h)                            # (B, C_out, H, W)

        # Re-apply land mask: land pixels become NaN in the output
        out = out.masked_fill(valid_mask.expand_as(out) == 0, float("nan"))

        return out
