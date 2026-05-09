"""
Lightning module for the GNN-based CROSCIM multi-resolution model.

Inherits the full training/validation/test pipeline from
:class:`Lit4dVarNet_CROSCIM_Supervised`, overriding only
:meth:`format_batch_for_solver` to produce a :class:`GraphSBatch` that
carries the spatial coordinates (lat, lon, land_mask) needed by the GNN.

Everything else — multi-resolution curriculum, loss terms, anomaly cascade,
test-step boundary conditioning, etc. — is unchanged.

Usage (Hydra config)
--------------------
model:
  _target_: contrib.CROSCIM.models.models_gnn.Lit4dVarNet_CROSCIM_GNN
  solver:
    _target_: contrib.CROSCIM.solvers.gnn_solver.MultiResGNNSolvers
    solvers:
      solver_x50:
        _target_: contrib.CROSCIM.solvers.gnn_solver.GNNSolver
        n_input_channels:  <computed>
        n_output_channels: <computed>
        hidden_dim: 128
        n_layers:   6
        n_heads:    4
        pos_dim:    16
        dropout:    0.1
      solver_x10:
        ...
"""

from __future__ import annotations

import torch
from torch import Tensor

from .models_supervised import Lit4dVarNet_CROSCIM_Supervised
from contrib.CROSCIM.solvers.gnn_solver import GraphSBatch


class Lit4dVarNet_CROSCIM_GNN(Lit4dVarNet_CROSCIM_Supervised):
    """
    GNN-backed CROSCIM multi-resolution Lightning model.

    Identical to :class:`Lit4dVarNet_CROSCIM_Supervised` except that
    :meth:`format_batch_for_solver` wraps the sBatch in a
    :class:`GraphSBatch` containing ``lat``, ``lon`` and ``land_mask``
    for the GNN's positional encoding and validity masking.

    Parameters
    ----------
    Same as :class:`Lit4dVarNet_CROSCIM_Supervised` — all extra GNN
    hyper-parameters live in the solver config.
    """

    # No new __init__ needed: all GNN hyper-parameters are inside the solver.

    def format_batch_for_solver(
        self,
        batch,
        include_masks: bool = False,
        res=None,
    ) -> GraphSBatch:
        """
        Build a :class:`GraphSBatch` from a TrainingItem batch.

        Calls the parent :meth:`format_batch_for_solver` to get the
        standard ``sBatch(input, tgt)`` tensors, then appends the spatial
        context (lat, lon, land_mask) extracted from the batch fields.

        Notes
        -----
        * ``lat`` / ``lon`` are stored as ``(B, 1, H, W)`` or ``(B, H, W)``
          in the batch; we normalise to ``(B, H, W)`` float.
        * ``land_mask`` is ``(B, T, H, W)`` or ``(B, 1, H, W)``; we
          collapse the time axis to get ``(B, 1, H, W)``.
        * If any of these fields is missing (older datasets), the
          ``GraphSBatch`` simply carries ``None`` and the GNN falls back to
          inferring the valid mask from NaN patterns in ``tgt``.
        """
        # ── 1. Standard sBatch (input, tgt tensors) ──────────────────
        sbatch = super().format_batch_for_solver(
            batch, include_masks=include_masks, res=res
        )
        device = sbatch.input.device

        # ── 2. Coordinates ────────────────────────────────────────────
        def _to_2d(tensor) -> Tensor | None:
            """Return (B, H, W) float on device, or None."""
            if tensor is None or not isinstance(tensor, Tensor):
                return None
            t = tensor.to(device=device, dtype=torch.float32)
            # (B, T, H, W) → take first time step
            if t.ndim == 4:
                t = t[:, 0]
            return t  # (B, H, W)

        lat = _to_2d(getattr(batch, "lat", None))
        lon = _to_2d(getattr(batch, "lon", None))

        # ── 3. Land mask ──────────────────────────────────────────────
        # land_mask in the batch: True/1 = valid ocean, 0/False = land
        lm_raw = getattr(batch, "land_mask", None)
        if isinstance(lm_raw, Tensor) and lm_raw.numel() > 0:
            lm = lm_raw.to(device=device, dtype=torch.float32)
            # Collapse time axis if present: any valid timestep → valid node
            if lm.ndim == 4:
                lm = lm.any(dim=1, keepdim=True).float()  # (B, 1, H, W)
            elif lm.ndim == 3:
                lm = lm.unsqueeze(1)                       # (B, 1, H, W)
        else:
            # Fallback: infer from NaN pattern in tgt (set by models_XXX masking)
            lm = sbatch.tgt.isfinite().any(dim=1, keepdim=True).float()

        return GraphSBatch(
            input     = sbatch.input,
            tgt       = sbatch.tgt,
            lat       = lat,
            lon       = lon,
            land_mask = lm,
        )
