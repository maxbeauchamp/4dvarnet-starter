"""
Swin U-Net backbone variant of the CROSCIM Flow Matching solver.

``FMSwinSolver`` subclasses ``FMSolver`` (flowmatching_solver.py) and
overrides ONLY network construction, swapping the ViT-style patch
``Transformer`` for ``SwinUNetBackbone`` (flow_swin_backbone.py). Everything
else -- ``sample_one``, boundary-ring conditioning (``setup_mask``,
``_encode_obs``), the ODE sampler -- is inherited unchanged, so
``Lit4dVarNet_CROSCIM_FlowMatching`` (models_flowmatching.py) works with this
solver with zero changes: it only ever calls ``solver.get_network()`` /
``solver.sample_one()``, agnostic to the backbone.

Hydra instantiation example::

    solver:
      _target_: contrib.CROSCIM.solvers.flowmatching_solver.FMGradSolvers
      solvers:
        solver_x50:
          _target_: contrib.CROSCIM.solvers.flowmatching_swin_solver.FMSwinSolver
          n_input_channels: ...
          n_output_channels: ...
          grid_size: [294, 304]
          ...
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch.nn as nn
from torch import Tensor

from .flow_swin_backbone import SwinUNetBackbone
from .flowmatching_solver import FMSolver
from .utils_flowmatching import FlowMatchingSampler


def _round_up_to_multiple(value: int, multiple: int) -> int:
    remainder = value % multiple
    return value if remainder == 0 else value + (multiple - remainder)


# ─────────────────────────────────────────────────────────────────────────────
# FMSwinWrapper -- adapts SwinUNetBackbone to the FMSolver calling convention
# ─────────────────────────────────────────────────────────────────────────────

class FMSwinWrapper(nn.Module):
    """Wraps ``SwinUNetBackbone`` to match the calling convention used by
    ``FlowMatchingSampler`` / ``FMSolver.sample_one``::

        model(cat(states, encoded), pseudo_time=t, mesh=..., mask=...,
              labels=..., resolution=...)

    ``mesh``/``mask``/``labels``/``resolution`` are accepted for signature
    compatibility but ignored: the Swin backbone builds its own RoPE
    positions internally from ``grid_size``, spatial validity is already
    conveyed through the ``obs_mask`` channel baked into ``conditions`` by
    ``FMSolver._encode_obs``, and ``labels``/``resolution`` are unused
    (each CROSCIM resolution gets its own solver/network instance, and
    ``labels`` is always zero in the existing pipeline).
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
        return self.backbone(noisy_state, conditions, pseudo_time)


# ─────────────────────────────────────────────────────────────────────────────
# FMSwinSolver -- FMSolver with the Swin backbone
# ─────────────────────────────────────────────────────────────────────────────

class FMSwinSolver(FMSolver):
    """``FMSolver`` with a Swin U-Net backbone instead of the ViT patch
    Transformer. Does not call ``FMSolver.__init__`` (the two backbones take
    different architecture kwargs) -- only network construction differs;
    ``sample_one``, ``setup_mask``, ``_encode_obs``, ``forward`` are
    inherited from ``FMSolver`` unchanged.
    """

    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
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
        n_steps: int = 20,
        schedule_scale: float = 0.1,
        schedule_shift: float = 0.,
        second_order: bool = True,
        add_bounds: bool = False,
    ):
        nn.Module.__init__(self)

        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.add_bounds = add_bounds

        # noisy_target + obs_clean + obs_mask [+ boundary_values + boundary_mask]
        n_swin_in = n_output_channels + n_input_channels + n_input_channels
        if add_bounds:
            n_swin_in += 2 * n_output_channels

        self.network = FMSwinWrapper(
            n_input=n_swin_in,
            n_output=n_output_channels,
            grid_size=grid_size,
            resolution_km=resolution_km,
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

        self.sampler = FlowMatchingSampler(
            model=None,
            n_steps=n_steps,
            schedule_scale=schedule_scale,
            schedule_shift=schedule_shift,
            second_order=second_order,
            censoring=add_bounds,
        )

        self.border_h: int = 0
        self.border_w: int = 0
        self.mask_boundary: Optional[Tensor] = None
