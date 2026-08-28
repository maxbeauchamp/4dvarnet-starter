"""
Flow Matching Lightning Module for CROSCIM multi-resolution framework.

Inherits from ``Lit4dVarNet_CROSCIM_Supervised`` and replaces the iterative
4DVarNet solver with a conditional Flow Matching (FM) training loop.

Training uses a linear flow interpolant (conditional flow matching):

    x_t = t · x_1 + (1 - t) · ε,   ε ~ N(0, I)
    v_target = x_1 - ε
    loss = NegLogPDF((v_target - v_pred) / exp(log_scale), log_scale)

where ``log_scale`` comes from a small ``LogScaleModel`` that conditions on
pseudo-time, resolution, and (optional) augmentation labels.

Key design choices
------------------
* Each resolution gets its own ``FMTransformerWrapper`` (student) + EMA copy
  via ``torch.optim.swa_utils.AveragedModel``.
* ``add_bounds=True`` enables censoring of the FM loss at lower / upper bounds
  (e.g. SIT ≥ 0) using ``neglogcdf``, mirroring gensim's censored FM loss.
  It also enables sequential boundary-ring conditioning at inference
  (ported from ``models_consistency.Lit4dVarNet_CROSCIM_Consistency``).
* Boundary conditioning utilities (``make_boundary_mask``, etc.) are imported
  from ``flowmatching_solver`` which re-exports them from ``consistency_solver``.
"""

from __future__ import annotations

import copy
import itertools
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from torch.optim.swa_utils import AveragedModel

from .models_supervised import Lit4dVarNet_CROSCIM_Supervised
from .stochastic_ensemble_test import StochasticEnsembleTestMixin, write_spread_netcdf
from contrib.CROSCIM.solvers.flowmatching_solver import (
    FMSolver,
    FMGradSolvers,
    make_boundary_mask,
    build_boundary_conditioning,
    random_boundary_dropout,
    raster_order_from_coords,
)

# ── local self-contained utilities (no external gensim dependency) ──────────
from contrib.CROSCIM.solvers.utils_flowmatching import (         # noqa: E402
    neglogpdf,
    neglogcdf,
    sample_uniform_time,
    masked_average,
    LogScaleModel,
)


# ─────────────────────────────────────────────────────────────────────────────
# EMA helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    """Exponential moving average update function for swa_utils.AveragedModel."""
    # NOTE: decay stored as attribute on the AveragedModel instance
    decay = getattr(averaged_model_parameter, "_ema_decay", 0.999)
    return decay * averaged_model_parameter + (1.0 - decay) * model_parameter


def _make_ema(module: nn.Module, decay: float) -> AveragedModel:
    """Build an EMA wrapper with a fixed decay rate."""
    ema = AveragedModel(module, avg_fn=lambda avg, cur, n: decay * avg + (1.0 - decay) * cur)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class Lit4dVarNet_CROSCIM_FlowMatching(StochasticEnsembleTestMixin, Lit4dVarNet_CROSCIM_Supervised):
    """Flow Matching variant of the CROSCIM multi-resolution Lightning module.

    Replaces the iterative 4DVarNet solver with a conditional FM training loop.
    Each resolution has its own student ``FMTransformerWrapper`` + EMA copy.

    Extra constructor arguments (on top of parent ``**kwargs``):
    -----------------------------------------------------------
    fm_config : dict
        Flow Matching hyper-parameters.  Recognised keys:

        ``lr`` (float, 1e-4)
            Peak learning rate.
        ``betas`` (list[float], [0.9, 0.995])
            Adam β parameters.
        ``lr_scheduler_start_factor`` (float, 1e-5)
            Linear warm-up start factor.
        ``lr_scheduler_iters`` (int, 20_000)
            Number of warm-up steps.
        ``ema_decay`` (float, 0.999)
            EMA decay rate (applied after each training step).
        ``gradient_clip_val`` (float, 0.5)
            Gradient clipping (applied inside ``training_step``).
        ``lower_bound`` (float or None)
            Physical lower bound of the target variable in **normalised** space.
            Used for censored FM loss when ``add_bounds=True``.
        ``upper_bound`` (float or None)
            Physical upper bound in normalised space.
        ``use_ema_at_inference`` (bool, True)
            Use the EMA network at validation/test time (previous, default
            behaviour). Set False to sample from the raw student network
            instead -- e.g. to diagnose whether the EMA is lagging behind a
            student that recently resumed training.
        ``tv_weight`` (dict[int, float], {})
            Per-resolution total-variation weight added to the training loss,
            e.g. ``{50: 0.01}`` to penalise blocky patch-boundary artefacts at
            x50 only. Computed on the one-step predicted field
            (``x_t + (1-t)*v_pred``), masked to pairs of finite target pixels.
            Missing resolutions default to 0 (no TV term, previous behaviour).

    add_bounds : bool
        Enable boundary-ring conditioning at inference + censored FM loss.
    """

    def __init__(
        self,
        fm_config: Optional[dict] = None,
        add_bounds: bool = False,
        n_test_members: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = fm_config or {}
        self.add_bounds = add_bounds
        self.n_test_members = n_test_members

        # ── FM hyper-parameters ───────────────────────────────────────────
        self._fm_lr                   = cfg.get("lr", 1e-4)
        self._fm_betas                = tuple(cfg.get("betas", (0.9, 0.995)))
        self._fm_lr_start_factor      = cfg.get("lr_scheduler_start_factor", 1e-5)
        self._fm_lr_iters             = cfg.get("lr_scheduler_iters", 20_000)
        self._fm_ema_decay            = cfg.get("ema_decay", 0.999)
        self._fm_lower_bound: Optional[float] = cfg.get("lower_bound", None)
        self._fm_upper_bound: Optional[float] = cfg.get("upper_bound", None)
        self._fm_tv_weight: Dict[int, float] = dict(cfg.get("tv_weight", {}))
        self._fm_use_ema_at_inference: bool = cfg.get("use_ema_at_inference", True)

        # ── Per-resolution EMA networks (student networks live in self.solver) ─
        self.ema_networks = nn.ModuleDict()
        for res in self.multires:
            key = f"solver_x{res}"
            student_net = self.solver.solvers[key].get_network()
            ema = _make_ema(student_net, self._fm_ema_decay)
            self.ema_networks[key] = ema

        # ── Per-resolution log-scale models ───────────────────────────────
        # Infer n_vars from the first solver's n_output_channels.
        # (All resolutions share the same number of output channels per step.)
        self.log_scale_models = nn.ModuleDict()
        for res in self.multires:
            key = f"solver_x{res}"
            n_out = self.solver.solvers[key].n_output_channels
            self.log_scale_models[key] = LogScaleModel(
                n_embedding=64,
                n_time_in=1,
                n_res_in=1,
                n_augment_in=3,
                n_vars=n_out,
            )

        self.automatic_optimization = True

        _sep = "=" * 60
        print(f"\n{_sep}")
        print("Lit4dVarNet_CROSCIM_FlowMatching initialised:")
        print(f"  Resolutions  : {self.multires}")
        print(f"  add_bounds   : {add_bounds}")
        print(f"  EMA decay    : {self._fm_ema_decay}")
        print(f"  LR           : {self._fm_lr}")
        print(f"  TV weight    : {self._fm_tv_weight or '(none)'}")
        print(f"  Use EMA @inf : {self._fm_use_ema_at_inference}")
        for res in self.multires:
            key = f"solver_x{res}"
            n_params = sum(
                p.numel() for p in self.solver.solvers[key].parameters()
            )
            print(f"  {key}: {n_params:,} params (student)")
        print(f"{_sep}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: build boundary kwargs for the FM loss call
    # ─────────────────────────────────────────────────────────────────────────

    def _build_boundary_kwargs(
        self, solver, x: Tensor, training: bool
    ) -> Dict[str, Tensor]:
        """Build boundary conditioning tensors if add_bounds is True."""
        if not self.add_bounds:
            return {}

        if solver.mask_boundary is None:
            ph, pw = x.shape[2], x.shape[3]
            bh = (ph - ph * 4 // 5) // 2
            bw = (pw - pw * 4 // 5) // 2
            solver.setup_mask(ph, pw, bh, bw)

        mask_hw = solver.mask_boundary.to(x.device)
        bound   = torch.nan_to_num(x) * mask_hw.unsqueeze(0).unsqueeze(0)
        mbound  = (
            mask_hw.unsqueeze(0).unsqueeze(0) * x.isfinite().float()
        )
        if training:
            bound, mbound = random_boundary_dropout(
                bound, mbound,
                solver.border_h, solver.border_w,
                p_drop_side=0.5,
            )
        return {"boundaries": bound, "mask_bound": mbound}

    def _get_latent_bounds(self) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """(lower, upper) for FlowMatchingSampler's inference-time hard clamp
        (``_bound_grad``) — same guard as the training-time censored loss in
        ``base_step``, so the physical bound is actually enforced at sampling
        time, not just encouraged through the loss."""
        if not self.add_bounds:
            return None
        if self._fm_lower_bound is None and self._fm_upper_bound is None:
            return None
        return (self._fm_lower_bound, self._fm_upper_bound)

    # ─────────────────────────────────────────────────────────────────────────
    # base_step — FM training / validation loss
    # ─────────────────────────────────────────────────────────────────────────

    def base_step(self, batch, res: int, phase: str = "", scale_channel=None):
        solver_key = f"solver_x{res}"
        sbatch     = self.format_batch_for_solver(
            batch, include_masks=self.include_masks, res=res, scale_channel=scale_channel
        )

        y = sbatch.input   # (B, C_in, H, W)  observations
        x = sbatch.tgt     # (B, C_out, H, W) clean targets

        finite_mask = torch.isfinite(x)
        x_clean     = torch.nan_to_num(x)

        solver = self.solver.solvers[solver_key]

        if self.training and phase == "train":
            # ── Flow Matching loss ────────────────────────────────────────
            network  = solver.get_network()
            B, C, H, W = x_clean.shape
            device, dtype = x_clean.device, x_clean.dtype

            # Sample equidistant pseudo-time in [0, 1]
            pseudo_time = sample_uniform_time(x_clean)      # (B, 1, 1, 1)
            t_flat      = pseudo_time.view(B, 1)            # (B, 1)

            # Linear interpolant: x_t = t * x_1 + (1-t) * noise
            noise      = torch.randn_like(x_clean)
            x_t        = pseudo_time * x_clean + (1.0 - pseudo_time) * noise
            v_target   = x_clean - noise                    # (B, C, H, W)

            # Build observation context
            boundary_kwargs = self._build_boundary_kwargs(solver, x, training=True)
            encoded = FMSolver._encode_obs(y, **boundary_kwargs)

            # Network input: cat(x_t, encoded)
            net_input = torch.cat([x_t, encoded], dim=1)

            # Resolution scalar
            resolution_km = solver.network.resolution_km
            resolution = torch.full(
                (B, 1), resolution_km, device=device, dtype=dtype
            )
            labels = torch.zeros(B, 3, device=device, dtype=dtype)

            # Predicted velocity
            v_pred = network(
                net_input, pseudo_time=t_flat,
                labels=labels, resolution=resolution
            )                                               # (B, C, H, W)

            # Log-scale model
            log_scale = self.log_scale_models[solver_key](
                t_flat, labels, resolution
            )                                               # (B, C)
            log_scale = log_scale.view(B, C, 1, 1).expand_as(v_pred)

            # Residual
            residual = (v_target - v_pred) / log_scale.exp()

            # NegLogPDF loss (only on finite pixels)
            loss_grid = neglogpdf(residual, log_scale)     # (B, C, H, W)

            # One-step predicted x_1 from the current interpolant, reused by
            # both the censored-loss bound check below and the TV term.
            x_pred = x_t + (1.0 - pseudo_time) * v_pred

            if self.add_bounds and (
                self._fm_lower_bound is not None or self._fm_upper_bound is not None
            ):
                # Censoring: add neglogcdf at the bounds
                if self._fm_lower_bound is not None:
                    lb = torch.full_like(x_pred, self._fm_lower_bound)
                    censor_lower = (x_pred <= lb).float()
                    loss_grid = (
                        loss_grid * (1.0 - censor_lower)
                        + censor_lower * neglogcdf(-residual)
                    )
                if self._fm_upper_bound is not None:
                    ub = torch.full_like(x_pred, self._fm_upper_bound)
                    censor_upper = (x_pred >= ub).float()
                    loss_grid = (
                        loss_grid * (1.0 - censor_upper)
                        + censor_upper * neglogcdf(residual)
                    )

            # Mask to finite target pixels
            loss = masked_average(
                loss_grid,
                finite_mask,
            )

            # Total-variation penalty on the predicted field, to discourage
            # blocky patch-boundary artefacts (opt-in per resolution via
            # fm_config.tv_weight, default 0 -- no effect on existing runs).
            tv_weight = self._fm_tv_weight.get(res, 0.0)
            if tv_weight > 0.0:
                tv_h_mask = finite_mask[:, :, 1:, :] & finite_mask[:, :, :-1, :]
                tv_w_mask = finite_mask[:, :, :, 1:] & finite_mask[:, :, :, :-1]
                tv_h = (x_pred[:, :, 1:, :] - x_pred[:, :, :-1, :]).abs()
                tv_w = (x_pred[:, :, :, 1:] - x_pred[:, :, :, :-1]).abs()
                tv_loss = masked_average(tv_h, tv_h_mask) + masked_average(tv_w, tv_w_mask)
                loss = loss + tv_weight * tv_loss
                if phase:
                    self.log(
                        f"{phase}_tv_loss", tv_loss,
                        on_step=True, on_epoch=True, sync_dist=True,
                    )

            if phase:
                self.log(
                    f"{phase}_loss", loss,
                    prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
                )
            if self.trainer.is_global_zero and (self.global_step % 50 == 0):
                print(
                    f"\n[Step {self.global_step:05d}] TRAIN | res=x{res} | "
                    f"loss={loss.item():.6f}"
                )

            # One-step prediction from current noise level (monitoring only).
            # Avoids running the full ODE integration at every training step.
            out_tensor = (x_t + (1.0 - pseudo_time) * v_pred).detach()

        else:
            # ── Validation / Test: EMA forward ───────────────────────────
            out_tensor = self._ema_forward(solver_key, sbatch)

            mask = torch.isfinite(x)
            loss = F.mse_loss(
                torch.where(mask, out_tensor, torch.zeros_like(out_tensor)),
                torch.where(mask, x,          torch.zeros_like(x)),
            )
            if phase:
                self.log(
                    f"{phase}_loss", loss,
                    prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
                )

        out = self.split_tensor_to_dict(out_tensor, res=res)
        return loss, out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ema_forward(self, solver_key: str, sbatch) -> Tensor:
        """Run the EMA network through the solver's sampler -- or the raw
        student network if ``self._fm_use_ema_at_inference`` is False (e.g.
        to check whether the EMA is lagging behind a student that just
        resumed training after being frozen for a while: the EMA keeps
        updating every batch even while its resolution is frozen, so it
        fully converges to that frozen/untrained state and then needs many
        steps to catch up once the student starts moving again)."""
        solver = self.solver.solvers[solver_key]

        # Build boundary kwargs (zeros / mask if add_bounds=True)
        boundary_kwargs = self._build_boundary_kwargs(solver, sbatch.tgt, training=False)

        if not self._fm_use_ema_at_inference:
            return solver.sample_one(
                sbatch.input, latent_bounds=self._get_latent_bounds(), **boundary_kwargs
            )

        ema_net = self.ema_networks[solver_key]

        # Temporarily swap the solver's network with the EMA version for sampling
        orig_net = solver.network
        solver.network = ema_net.module   # AveragedModel.module = averaged copy
        solver.sampler.model = solver.network
        try:
            out = solver.sample_one(
                sbatch.input, latent_bounds=self._get_latent_bounds(), **boundary_kwargs
            )
        finally:
            solver.network = orig_net
            solver.sampler.model = orig_net
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Override step: skip auxiliary losses (grad, prior, tv, context)
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, batch, res: int, phase: str = "", scale_channel=None):
        return self.base_step(batch, res=res, phase=phase, scale_channel=scale_channel)

    # ─────────────────────────────────────────────────────────────────────────
    # EMA update after each training batch
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for res in self.multires:
            key      = f"solver_x{res}"
            student  = self.solver.solvers[key].get_network()
            self.ema_networks[key].update_parameters(student)

    # ─────────────────────────────────────────────────────────────────────────
    # Optimizer
    # ─────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        params = []
        for res in self.multires:
            key = f"solver_x{res}"
            params.extend(
                filter(
                    lambda p: p.requires_grad,
                    self.solver.solvers[key].parameters(),
                )
            )
            params.extend(self.log_scale_models[key].parameters())

        opt = torch.optim.Adam(params, lr=self._fm_lr, betas=self._fm_betas)
        sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self._fm_lr_start_factor,
            total_iters=self._fm_lr_iters,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Forward override: capture boundary inputs at test time
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch, res: int = 1) -> Tensor:
        solver = self.solver.solvers[f"solver_x{res}"]
        if self.add_bounds and not self.training:
            res_key = f"patch_x{res}"
            if not hasattr(self, "_bound_inputs"):
                self._bound_inputs = {}
            if res_key not in self._bound_inputs:
                self._bound_inputs[res_key] = []
            # Observations don't vary across ensemble members (only the noise
            # draw inside sample_one does) — append once per batch, not once
            # per member, so _bound_inputs stays aligned with domain_masks.
            if getattr(self, "_ensemble_member_idx", 0) == 0:
                self._bound_inputs[res_key].append(batch.input.detach().cpu())

        if self.add_bounds:
            # No neighbouring-patch boundary info exists yet at this stage
            # (that only exists once _finalize_res's _apply_sequential_inference
            # reconstructs patches in raster order and this pass's output gets
            # superseded) — feed zero placeholders (mask_bound=0 => "no
            # boundary info") so the network still receives the channel count
            # it was built for, without fabricating values or leaking GT.
            n_out = solver.n_output_channels
            zeros = torch.zeros(
                batch.input.shape[0], n_out, *batch.input.shape[2:],
                device=batch.input.device, dtype=batch.input.dtype,
            )
            return solver.sample_one(
                batch.input, boundaries=zeros, mask_bound=zeros,
                latent_bounds=self._get_latent_bounds(),
            )
        return solver(batch)

    # ─────────────────────────────────────────────────────────────────────────
    # on_test_start: build boundary masks
    # ─────────────────────────────────────────────────────────────────────────

    def on_test_start(self):
        super().on_test_start()
        if not self.add_bounds:
            return
        for dataloader_idx, res in enumerate(self.multires):
            key = f"solver_x{res}"
            dl  = self.trainer.test_dataloaders[
                self.dataloader_keys[dataloader_idx]
            ]
            ds       = dl.dataset
            patch_h  = ds.patch_dims["yc"]
            patch_w  = ds.patch_dims["xc"]
            stride_h = ds.strides.get("yc", patch_h)
            stride_w = ds.strides.get("xc", patch_w)
            border_h = (patch_h - stride_h) // 2
            border_w = (patch_w - stride_w) // 2
            print(
                f"[add_bounds] {key}: patch=({patch_h},{patch_w}), "
                f"stride=({stride_h},{stride_w}), border=({border_h},{border_w})"
            )
            self.solver.solvers[key].setup_mask(patch_h, patch_w, border_h, border_w)

    # ─────────────────────────────────────────────────────────────────────────
    # test_step: swap to EMA network for inference
    # ─────────────────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        _dl_idx = 0 if dataloader_idx is None else dataloader_idx
        res     = self.multires[_dl_idx]
        res_key = f"patch_x{res}"

        if self.add_bounds and batch_idx == 0:
            if not hasattr(self, "_bound_inputs"):
                self._bound_inputs = {}
            if not hasattr(self, "_bound_domain_masks"):
                self._bound_domain_masks = {}
            if _dl_idx == 0:
                self._bound_inputs = {}
                self._bound_domain_masks = {}
            self._bound_inputs[res_key] = []
            self._bound_domain_masks[res_key] = []

        if self.add_bounds:
            _batch_dict = batch._asdict() if hasattr(batch, '_asdict') else vars(batch)
            _models_var = next(
                (k for k in _batch_dict
                 if k.startswith("models_")
                 and isinstance(_batch_dict[k], torch.Tensor)
                 and _batch_dict[k].numel() > 0),
                None
            )
            if _models_var is not None:
                _domain_invalid = ~_batch_dict[_models_var].isfinite().any(
                    dim=1, keepdim=True
                ).cpu()
            elif hasattr(batch, 'land_mask'):
                _domain_invalid = (batch.land_mask == 1.).cpu()
            else:
                _domain_invalid = None

            if _domain_invalid is not None:
                if res_key not in self._bound_domain_masks:
                    self._bound_domain_masks[res_key] = []
                for _b in range(_domain_invalid.shape[0]):
                    self._bound_domain_masks[res_key].append(_domain_invalid[_b])

        # Swap to EMA networks
        orig_nets = {}
        for solver_key in [f"solver_x{r}" for r in self.multires]:
            orig_nets[solver_key] = self.solver.solvers[solver_key].network
            ema_mod = self.ema_networks[solver_key].module
            self.solver.solvers[solver_key].network = ema_mod
            self.solver.solvers[solver_key].sampler.model = ema_mod
        try:
            result = super().test_step(batch, batch_idx, dataloader_idx)
        finally:
            for key, net in orig_nets.items():
                self.solver.solvers[key].network = net
                self.solver.solvers[key].sampler.model = net
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Sequential boundary-aware inference (mirrors models_consistency)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _apply_sequential_inference(self, res, inputs, coords, stacked, domain_masks=None):
        solver_key = f"solver_x{res}"
        solver     = self.solver.solvers[solver_key]

        if solver.mask_boundary is None:
            raise RuntimeError(
                f"setup_mask() not called for {solver_key}. "
                "Ensure on_test_start ran with add_bounds=True."
            )

        mask_cpu = solver.mask_boundary.cpu()
        border_h = solver.border_h
        border_w = solver.border_w
        device   = next(solver.network.parameters()).device
        n_tgt    = len(self._get_target_vars_for_resolution(res))
        c_out    = solver.n_output_channels

        order = raster_order_from_coords(coords)
        cache: dict = {}
        new_stacked = list(stacked)

        for b_idx, iy, ix in order:
            y_single = inputs[b_idx].unsqueeze(0).to(device)

            bound, mbound = build_boundary_conditioning(
                cache, iy, ix, mask_cpu, border_h, border_w, c_out=c_out
            )
            bound  = bound.unsqueeze(0).to(device)
            mbound = mbound.unsqueeze(0).to(device)

            pred = solver.sample_one(
                y_single, boundaries=bound, mask_bound=mbound,
                latent_bounds=self._get_latent_bounds(),
            )

            cache[(iy, ix)] = pred[0].cpu()

            s    = new_stacked[b_idx]
            T    = s.shape[1]
            H, W = s.shape[2], s.shape[3]
            pred_cpu  = pred[0].cpu().view(n_tgt, T, H, W)

            if domain_masks is not None and b_idx < len(domain_masks):
                _dinv = domain_masks[b_idx].expand(n_tgt, T, H, W)
            else:
                tgt_channels = s[n_tgt:]
                _dinv = ~tgt_channels.isfinite().any(dim=1, keepdim=True).expand_as(pred_cpu)
            pred_cpu = pred_cpu.masked_fill(_dinv, float('nan'))

            s_new     = s.clone()
            s_new[:n_tgt] = pred_cpu
            new_stacked[b_idx] = s_new

        return new_stacked

    def _finalize_res(self, dataloader_idx, idx_rec, write_netcdf=True):
        if not self.add_bounds:
            return super()._finalize_res(dataloader_idx, idx_rec, write_netcdf)

        import torch.distributed as dist

        res       = self.multires[dataloader_idx]
        res_key   = f"patch_x{res}"
        n_members = getattr(self, "n_test_members", 1)

        # Observations / domain mask are shared across members (only
        # test_data/test_times/test_coords become member-nested lists, via
        # StochasticEnsembleTestMixin.test_step, when n_members > 1).
        inputs       = list(itertools.chain(*self._bound_inputs.get(res_key, [])))
        domain_masks = list(getattr(self, '_bound_domain_masks', {}).get(res_key, []))

        results = []
        for m in range(n_members):
            if n_members > 1:
                times   = list(itertools.chain(*self.test_times[res_key][m]))
                coords  = list(itertools.chain(*self.test_coords[res_key][m]))
                stacked = list(itertools.chain(*self.test_data[res_key][m]))
            else:
                times   = list(itertools.chain(*self.test_times[res_key]))
                coords  = list(itertools.chain(*self.test_coords[res_key]))
                stacked = list(itertools.chain(*self.test_data[res_key]))

            cur_inputs, cur_times, cur_coords = inputs, times, coords
            cur_stacked, cur_domain_masks = stacked, domain_masks

            if self.trainer.world_size > 1:
                gathered = [None] * self.trainer.world_size
                dist.all_gather_object(
                    gathered,
                    {
                        "inputs":       [x.cpu() for x in cur_inputs],
                        "times":        [t.cpu() for t in cur_times],
                        "coords":       cur_coords,
                        "stacked":      [s.cpu() for s in cur_stacked],
                        "domain_masks": [dm.cpu() for dm in cur_domain_masks],
                    },
                )
                cur_inputs       = [x for g in gathered for x in g["inputs"]]
                cur_times        = [t for g in gathered for t in g["times"]]
                cur_coords       = [c for g in gathered for c in g["coords"]]
                cur_stacked      = [s for g in gathered for s in g["stacked"]]
                cur_domain_masks = [dm for g in gathered for dm in g["domain_masks"]]

            if self.trainer.world_size > 1 and not self.trainer.is_global_zero:
                result = None
            else:
                new_stacked = self._apply_sequential_inference(
                    res, cur_inputs, cur_coords, cur_stacked,
                    domain_masks=cur_domain_masks or None,
                )
                result = self.aggregate_batches(
                    idx_rec, new_stacked, cur_times, dataloader_idx,
                    metrics=False, write_netcdf=write_netcdf,
                    patch_coords=cur_coords,
                    member=(m if n_members > 1 else None),
                )
                print(result)

            if self.trainer.world_size > 1:
                container = [result]
                dist.broadcast_object_list(container, src=0)
                result = container[0]

            results.append(result)

        if self.trainer.world_size <= 1 or self.trainer.is_global_zero:
            write_spread_netcdf(self, res, n_members, write_netcdf=write_netcdf)

        self.aggregate_results[res_key] = results if n_members > 1 else results[0]
