"""
Flow Matching Solver for CROSCIM multi-resolution framework.

Provides a CROSCIM-compatible solver built on the gensim Transformer
architecture and FlowMatchingSampler (Euler / Heun ODE integration).

Architecture overview
---------------------
* ``FMTransformerWrapper`` – thin wrapper around ``gensim.network.Transformer``
  that handles:
    - mesh generation (normalised (y, x) coordinate grid)
    - observation NaN filling + mask channel concatenation
    - fixed ``labels`` (zeros) and ``resolution`` tensors when not provided
* ``FMSolver`` – CROSCIM-compatible solver (mirrors ConsistencyUNetSolver):
    - builds ``FMTransformerWrapper`` + ``FlowMatchingSampler``
    - ``forward(batch)`` → calls ``sample_one(batch.input)``
    - ``setup_mask()`` for boundary conditioning support
* ``FMGradSolvers`` – mirrors ``ConsistencyGradSolvers``; keyed by resolution

Boundary conditioning utilities (``make_boundary_mask``,
``build_boundary_conditioning``, ``random_boundary_dropout``,
``raster_order_from_coords``) are re-exported from ``consistency_solver`` so
that ``models_flowmatching`` can import them from a single place.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── local self-contained utilities (no external gensim dependency) ──────────
from .utils_flowmatching import Transformer, FlowMatchingSampler  # noqa: E402

# ── Re-export boundary utilities from consistency_solver ─────────────────────
from contrib.CROSCIM.solvers.consistency_solver import (               # noqa: F401  (re-export)
    make_boundary_mask,
    build_boundary_conditioning,
    random_boundary_dropout,
    raster_order_from_coords,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create a simple 2-D coordinate mesh
# ─────────────────────────────────────────────────────────────────────────────

def _make_grid_mesh(
    batch_size: int, height: int, width: int,
    device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Return a (B, 2, H, W) tensor with normalised (row, col) coordinates
    in the range [0, 1], broadcast across the batch dimension."""
    rows = torch.linspace(0., 1., height, device=device, dtype=dtype)
    cols = torch.linspace(0., 1., width,  device=device, dtype=dtype)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")  # (H, W)
    mesh = torch.stack([grid_r, grid_c], dim=0)                  # (2, H, W)
    return mesh.unsqueeze(0).expand(batch_size, -1, -1, -1)      # (B, 2, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# FMTransformerWrapper
# ─────────────────────────────────────────────────────────────────────────────

class FMTransformerWrapper(nn.Module):
    """Wraps ``gensim.network.Transformer`` to match the CROSCIM calling
    convention used by ``FlowMatchingSampler``.

    The sampler calls::

        model(cat(states, encoded), pseudo_time=t, mesh=..., mask=...,
              labels=..., resolution=...)

    where ``states`` is the noisy target and ``encoded`` is the processed
    observation context.  This wrapper:

    1. Builds a coordinate mesh on-the-fly when ``mesh`` is ``None``.
    2. Builds an all-ones spatial mask when ``mask`` is ``None``.
    3. Builds zero labels and a fixed resolution scalar when those are ``None``.
    4. Forwards through the underlying ``Transformer``.

    Parameters
    ----------
    n_input : int
        Total input channels = n_output_channels + n_obs_channels
        (noisy target + processed observations + optional boundary channels).
    n_output : int
        Output channels = number of target channels (predicted velocity).
    resolution_km : float
        Nominal spatial resolution in km (e.g. 50 for x50, 10 for x10).
        Used as the default resolution scalar when ``resolution`` is not passed.
    n_features : int
        Width of the Transformer hidden dimension.
    n_blocks : int
        Number of Transformer blocks.
    n_embedding : int
        Dimension of the time / resolution / augmentation embedding.
    patch_size : int
        Spatial patch size for tokenisation.
    n_heads : int
        Number of self-attention heads.
    n_rope_features : int
        Dimensionality of RoPE frequency features per head.
    long_skips : bool
        Whether to use long (encoder→decoder) skip connections.
    add_bounds : bool
        If True, the input contains extra boundary-ring channels
        (boundaries + mask_bound appended to obs).
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        resolution_km: float = 50.,
        n_features: int = 256,
        n_blocks: int = 8,
        n_embedding: int = 128,
        patch_size: int = 4,
        n_heads: int = 8,
        n_rope_features: Optional[int] = None,
        long_skips: bool = True,
        add_bounds: bool = False,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.resolution_km = resolution_km
        self.add_bounds = add_bounds

        # RoPE requires 2 * n_rope_features <= n_features_head.
        # Auto-derive as n_features_head // 2 when not explicitly set.
        n_features_head = n_features // n_heads
        if n_rope_features is None:
            n_rope_features = n_features_head // 2

        self.transformer = Transformer(
            n_input=n_input,
            n_output=n_output,
            n_features=n_features,
            n_blocks=n_blocks,
            n_embedding=n_embedding,
            n_time_in=1,
            n_res_in=1,
            n_augment_in=3,
            n_features_head=n_features_head,
            n_heads=n_heads,
            n_rope_features=n_rope_features,
            mult=1,
            patch_size=patch_size,
            long_skips=long_skips,
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
        """
        Parameters
        ----------
        in_tensor : (B, n_input, H, W)
            Concatenation of [noisy_target, processed_obs, obs_mask_float,
            (boundaries, mask_bound if add_bounds)].
        pseudo_time : (B, 1)
            Flow-matching pseudo-time in [0, 1].
        mesh : (B, 2, H, W), optional
            Normalised spatial coordinates.  Created on-the-fly if None.
        mask : (B, 1, H, W), optional
            Spatial validity mask (1 = valid pixel).  All-ones if None.
        labels : (B, 3), optional
            Augmentation labels.  Zeros if None.
        resolution : (B, 1), optional
            Spatial resolution scalar.  Defaults to ``resolution_km``.

        Returns
        -------
        (B, n_output, H, W)  – predicted velocity field.
        """
        B, _, H, W = in_tensor.shape
        device, dtype = in_tensor.device, in_tensor.dtype

        if mesh is None:
            mesh = _make_grid_mesh(B, H, W, device, dtype)
        if mask is None:
            mask = torch.ones(B, 1, H, W, device=device, dtype=dtype)
        if labels is None:
            labels = torch.zeros(B, 3, device=device, dtype=dtype)
        if resolution is None:
            resolution = torch.full(
                (B, 1), self.resolution_km, device=device, dtype=dtype
            )

        # The tokenizer patchifies with a fixed patch_size, which requires H
        # and W to be exact multiples of it — not guaranteed for every domain
        # shape (e.g. the x50 grid is 294x304; 294 isn't a multiple of 4).
        # Pad up to the next multiple, run the transformer, then crop back.
        p = self.transformer.patch_size
        pad_h, pad_w = (-H) % p, (-W) % p
        if pad_h or pad_w:
            in_tensor = F.pad(in_tensor, (0, pad_w, 0, pad_h))
            mesh = F.pad(mesh, (0, pad_w, 0, pad_h))
            mask = F.pad(mask, (0, pad_w, 0, pad_h))

        out = self.transformer(
            in_tensor, mesh, mask, pseudo_time, labels, resolution
        )
        if pad_h or pad_w:
            out = out[..., :H, :W]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# FMSolver  (mirrors ConsistencyUNetSolver)
# ─────────────────────────────────────────────────────────────────────────────

class FMSolver(nn.Module):
    """CROSCIM-compatible Flow Matching solver.

    Wraps an ``FMTransformerWrapper`` and a ``FlowMatchingSampler`` so it can be
    used inside ``FMGradSolvers`` and called from the CROSCIM Lightning module.

    In *training* the forward is NOT used directly — the Lightning module drives
    the FM loss explicitly.  In *test / val* mode, ``forward(batch)`` runs the
    full ODE integration via ``sample_one``.

    Parameters
    ----------
    n_input_channels : int
        Total observation channels (n_obs_vars * n_time_steps).
    n_output_channels : int
        Target channels (n_target_vars * n_time_steps).
    resolution_km : float
        Nominal spatial resolution for this solver (used in resolution embedding).
    n_features : int
        Transformer hidden dimension.
    n_blocks : int
        Number of Transformer blocks.
    n_embedding : int
        Embedding dimension.
    patch_size : int
        Tokenisation patch size.
    n_heads : int
        Self-attention heads.
    n_rope_features : int
        RoPE frequency features per head.
    long_skips : bool
        Long encoder→decoder skip connections.
    n_steps : int
        Number of ODE integration steps at inference.
    schedule_scale : float
        Sigmoid time-schedule scale (0 = uniform).
    schedule_shift : float
        Sigmoid time-schedule shift.
    second_order : bool
        Use Heun (2nd-order) ODE integration instead of Euler.
    add_bounds : bool
        Enable boundary-ring conditioning channels.
    """

    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        resolution_km: float = 50.,
        n_features: int = 256,
        n_blocks: int = 8,
        n_embedding: int = 128,
        patch_size: int = 4,
        n_heads: int = 8,
        n_rope_features: Optional[int] = None,   # None → auto-derived in FMTransformerWrapper
        long_skips: bool = True,
        n_steps: int = 20,
        schedule_scale: float = 0.1,
        schedule_shift: float = 0.,
        second_order: bool = True,
        add_bounds: bool = False,
    ):
        super().__init__()

        self.n_input_channels  = n_input_channels
        self.n_output_channels = n_output_channels
        self.add_bounds        = add_bounds

        # ── Build input channel count for the Transformer ─────────────────
        # noisy_target + obs_clean + obs_mask
        n_transformer_in = n_output_channels + n_input_channels + n_input_channels
        if add_bounds:
            # + boundary_values + boundary_mask (both have n_output_channels)
            n_transformer_in += 2 * n_output_channels

        self.network = FMTransformerWrapper(
            n_input=n_transformer_in,
            n_output=n_output_channels,
            resolution_km=resolution_km,
            n_features=n_features,
            n_blocks=n_blocks,
            n_embedding=n_embedding,
            patch_size=patch_size,
            n_heads=n_heads,
            n_rope_features=n_rope_features,
            long_skips=long_skips,
            add_bounds=add_bounds,
        )

        self.sampler = FlowMatchingSampler(
            model=None,          # set in sample_one / compute_grad
            n_steps=n_steps,
            schedule_scale=schedule_scale,
            schedule_shift=schedule_shift,
            second_order=second_order,
            censoring=add_bounds,
        )

        # Boundary mask state (populated by setup_mask in Lightning module)
        self.border_h: int = 0
        self.border_w: int = 0
        self.mask_boundary: Optional[Tensor] = None

    # ── Boundary mask setup ───────────────────────────────────────────────────

    def setup_mask(
        self, patch_h: int, patch_w: int, border_h: int, border_w: int
    ):
        """Build and store the boundary ring mask.  Called from
        ``Lit4dVarNet_CROSCIM_FlowMatching.on_test_start``."""
        self.border_h = border_h
        self.border_w = border_w
        self.mask_boundary = make_boundary_mask(patch_h, patch_w, border_h, border_w)

    # ── Raw network access (for training / EMA) ───────────────────────────────

    def get_network(self) -> FMTransformerWrapper:
        return self.network

    # ── Build the encoded context tensor from observations ────────────────────

    @staticmethod
    def _encode_obs(
        y: Tensor,
        boundaries: Optional[Tensor] = None,
        mask_bound: Optional[Tensor] = None,
    ) -> Tensor:
        """Build the observation context tensor used as ``encoded`` by the
        sampler.  Handles NaN masking and optional boundary channels.

        Returns a (B, C_encoded, H, W) tensor free of NaN.
        """
        obs_mask  = (~torch.isnan(y)).float()
        obs_clean = torch.nan_to_num(y)
        parts = [obs_clean, obs_mask]
        if boundaries is not None and mask_bound is not None:
            parts += [boundaries, mask_bound]
        return torch.cat(parts, dim=1)

    # ── Core sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_one(
        self,
        y: Tensor,
        boundaries: Optional[Tensor] = None,
        mask_bound: Optional[Tensor] = None,
        latent_bounds: Optional[Tuple[Tensor, Tensor]] = None,
        mesh: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run ODE integration for a single batch.

        Parameters
        ----------
        y : (B, C_in, H, W)
            Observation tensor (may contain NaN).
        boundaries : (B, C_out, H, W), optional
            Boundary ring values (add_bounds mode).
        mask_bound : (B, C_out, H, W), optional
            Boundary availability mask.
        latent_bounds : tuple of two (C_out, 1, 1) tensors, optional
            Lower / upper bounds in latent space for censored sampling.
        mesh : (B, 2, H, W), optional
        mask : (B, 1, H, W), optional

        Returns
        -------
        (B, C_out, H, W)
        """
        B, _, H, W = y.shape
        device, dtype = y.device, y.dtype

        encoded = self._encode_obs(y, boundaries, mask_bound)

        # Initial noise state
        noise = torch.randn(
            B, self.n_output_channels, H, W, device=device, dtype=dtype
        )

        # Attach model to sampler and sample
        self.sampler.model = self.network
        result = self.sampler.sample(
            states=noise,
            encoded=encoded,
            latent_bounds=latent_bounds,
            mesh=mesh,
            mask=mask,
        )
        return result

    # ── CROSCIM-style forward (sBatch → prediction tensor) ───────────────────

    @torch.no_grad()
    def forward(self, batch) -> Tensor:
        """Run ODE integration from an sBatch(input, tgt).

        Delegates to ``sample_one(batch.input)``.
        """
        return self.sample_one(batch.input)


# ─────────────────────────────────────────────────────────────────────────────
# FMGradSolvers  (mirrors ConsistencyGradSolvers)
# ─────────────────────────────────────────────────────────────────────────────

class FMGradSolvers(nn.Module):
    """Drop-in replacement for ``ConsistencyGradSolvers`` that holds per-
    resolution ``FMSolver`` instances.

    Hydra instantiation example::

        solver:
          _target_: contrib.CROSCIM.solvers.flowmatching_solver.FMGradSolvers
          solvers:
            solver_x50:
              _target_: contrib.CROSCIM.solvers.flowmatching_solver.FMSolver
              n_input_channels: ...
              n_output_channels: ...
            solver_x10:
              _target_: contrib.CROSCIM.solvers.flowmatching_solver.FMSolver
              ...
    """

    def __init__(self, solvers: dict, **kwargs):
        super().__init__()
        self.solvers = nn.ModuleDict(solvers)

    def forward(self, batch, res: int = 1) -> Tensor:
        return self.solvers[f"solver_x{res}"](batch)
