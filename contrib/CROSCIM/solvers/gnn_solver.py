"""
GNN solver for the CROSCIM multi-resolution framework.

Provides a drop-in replacement for ConsistencyUNetSolver / UNet-based solvers
while using a spatial Graph Neural Network (MultiResGridGNN) as backbone.

Classes
-------
GraphSBatch          Extended sBatch that carries lat, lon and land_mask
                     needed by the GNN for positional encoding and masking.
GNNSolver            Wraps MultiResGridGNN in the CROSCIM solver interface.
MultiResGNNSolvers   Container of per-resolution GNNSolver instances
                     (mirrors ConsistencyGradSolvers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .GNN import MultiResGridGNN
from contrib.CROSCIM.models.models import sBatch  # base batch dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Extended batch dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphSBatch:
    """
    sBatch extended with spatial context for the GNN.

    Fields
    ------
    input      : (B, C_in, H, W)  — concatenated input features (NaN = missing)
    tgt        : (B, C_tgt, H, W) — concatenated target fields
    lat        : (B, H, W)        — latitude  of each pixel [degrees]
    lon        : (B, H, W)        — longitude of each pixel [degrees]
    land_mask  : (B, H, W) or (B, 1, H, W)
                 float, 1 = valid ocean pixel, 0 = land / out-of-domain
                 If None, the valid mask is inferred from NaN in `tgt`.
    """
    input:     Tensor
    tgt:       Tensor
    lat:       Optional[Tensor] = None
    lon:       Optional[Tensor] = None
    land_mask: Optional[Tensor] = None


# ─────────────────────────────────────────────────────────────────────────────
# GNN solver
# ─────────────────────────────────────────────────────────────────────────────

class GNNSolver(nn.Module):
    """
    CROSCIM solver backed by a spatial graph attention network.

    Accepts both :class:`sBatch` (legacy / compatibility) and
    :class:`GraphSBatch` (with explicit coordinates / land mask).

    Parameters
    ----------
    n_input_channels  : total number of input channels  (n_vars × T)
    n_output_channels : total number of output channels (n_tgt_vars × T)
    hidden_dim        : internal GNN width
    n_layers          : number of GNNBlock layers stacked
    n_heads           : multi-head attention heads
    pos_dim           : positional-encoding embedding dimension (0 = disabled)
    dropout           : dropout throughout the network
    n_levels          : number of ×2 pooling levels (1 = flat, previous
                         behaviour; >1 = Graph-UNet, see MultiResGridGNN)
    """

    def __init__(
        self,
        n_input_channels:  int,
        n_output_channels: int,
        hidden_dim:        int   = 128,
        n_layers:          int   = 6,
        n_heads:           int   = 4,
        pos_dim:           int   = 16,
        dropout:           float = 0.1,
        n_levels:          int   = 1,
    ) -> None:
        super().__init__()
        self.n_input_channels  = n_input_channels
        self.n_output_channels = n_output_channels

        self.gnn = MultiResGridGNN(
            in_channels  = n_input_channels,
            out_channels = n_output_channels,
            hidden_dim   = hidden_dim,
            n_layers     = n_layers,
            n_heads      = n_heads,
            pos_dim      = pos_dim,
            dropout      = dropout,
            n_levels     = n_levels,
        )

    # ------------------------------------------------------------------
    # Helper: build (B, 1, H, W) float valid mask
    # ------------------------------------------------------------------
    def _build_valid_mask(self, sbatch) -> Tensor:
        """
        Priority:
        1. land_mask field of GraphSBatch (most reliable)
        2. NaN pattern in the *first channel* of tgt (target variables
           contain NaN on land/out-of-domain by construction)
        3. NaN pattern in the first channel of input (fallback)
        """
        # --- Option 1: explicit land_mask ---
        if isinstance(sbatch, GraphSBatch) and sbatch.land_mask is not None:
            lm = sbatch.land_mask
            if lm.ndim == 3:
                lm = lm.unsqueeze(1)        # (B, 1, H, W)
            return lm.float()

        # --- Option 2: NaN in tgt ---
        tgt = sbatch.tgt
        if tgt is not None and tgt.numel() > 0:
            # Take union of finite pixels across all tgt channels
            valid = tgt.isfinite().any(dim=1, keepdim=True).float()
            return valid

        # --- Option 3: NaN in input ---
        inp = sbatch.input
        valid = inp[:, :1].isfinite().float()
        return valid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, sbatch) -> Tensor:
        """
        Args:
            sbatch : sBatch | GraphSBatch

        Returns:
            Tensor of shape (B, n_output_channels, H, W)
        """
        valid_mask = self._build_valid_mask(sbatch)             # (B, 1, H, W)

        lat = getattr(sbatch, "lat", None)
        lon = getattr(sbatch, "lon", None)

        # Squeeze extra time dimension in coordinates if present
        if lat is not None and lat.ndim == 4:
            lat = lat[:, 0]   # (B, H, W)
        if lon is not None and lon.ndim == 4:
            lon = lon[:, 0]

        return self.gnn(sbatch.input, valid_mask, lat=lat, lon=lon)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-resolution container
# ─────────────────────────────────────────────────────────────────────────────

class MultiResGNNSolvers(nn.Module):
    """
    Container of per-resolution :class:`GNNSolver` instances.

    Mirrors the interface of ``ConsistencyGradSolvers`` so that
    :class:`Lit4dVarNet_CROSCIM_GNN` can swap it in without changing
    the training loop.

    Parameters
    ----------
    solvers : dict mapping ``"solver_x{res}"`` → :class:`GNNSolver`
    """

    def __init__(self, solvers: Dict[str, GNNSolver]) -> None:
        super().__init__()
        self.solvers = nn.ModuleDict(solvers)

    def forward(self, sbatch, res: int = 1) -> Tensor:
        return self.solvers[f"solver_x{res}"](sbatch)
