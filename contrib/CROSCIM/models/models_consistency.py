"""
Consistency Model Lightning Module for CROSCIM multi-resolution framework.

Inherits from Lit4dVarNet_CROSCIM_Supervised and replaces the iterative
4DVarNet solver with a pairwise consistency training loop (student / teacher / EMA).

Key design:
- Each resolution gets its own student/teacher/ema_student triplet of ConsistencyUNet.
- training_step overrides the parent to use ConsistencyTrainingFewSteps_TimeEmbedding.
- test_step, reconstruct, aggregate_batches are inherited from the parent unchanged.
- At test time, inference uses iterative consistency sampling via ConsistencyUNetSolver.forward().
"""

import copy
import math
import itertools
from collections import namedtuple
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor

from .models_supervised import Lit4dVarNet_CROSCIM_Supervised
from contrib.CROSCIM.solvers.consistency_solver import (
    ConsistencyUNet,
    ConsistencyUNetConfig,
    ConsistencyUNetSolver,
    ConsistencyGradSolvers,
    consistency_forward_wrapper,
    compute_sigma,
    pad_dims_like,
    skip_scaling,
    output_scaling,
    make_boundary_mask,
    build_boundary_conditioning,
    raster_order_from_coords,
    random_boundary_dropout,
)

# Re-use the Karras schedule and EMA helpers
# These are small pure functions, so we inline them here to avoid
# a hard dependency on the devs repo.

def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 150,
) -> int:
    num_timesteps = (final_timesteps + 1) ** 2 - initial_timesteps ** 2
    num_timesteps = current_training_step * num_timesteps / total_training_steps
    num_timesteps = math.ceil(math.sqrt(num_timesteps + initial_timesteps ** 2) - 1)
    return num_timesteps + 1


def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device=None,
    as_time: bool = False,
) -> Tensor:
    rho_inv = 1.0 / rho
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min ** rho_inv + steps * (sigma_max ** rho_inv - sigma_min ** rho_inv)
    sigmas = sigmas ** rho
    return steps if as_time else sigmas


def ema_decay_rate_schedule(
    num_timesteps: int,
    initial_ema_decay_rate: float = 0.95,
    initial_timesteps: int = 2,
) -> float:
    return math.exp(
        (initial_timesteps * math.log(initial_ema_decay_rate)) / num_timesteps
    )


def _update_ema_weights(ema_iter, online_iter, decay: float):
    for ema_w, online_w in zip(ema_iter, online_iter):
        ema_w.data.lerp_(online_w.data, 1.0 - decay)


def update_ema_model_(ema_model: nn.Module, online_model: nn.Module, decay: float):
    _update_ema_weights(ema_model.parameters(), online_model.parameters(), decay)
    _update_ema_weights(ema_model.buffers(), online_model.buffers(), decay)
    return ema_model


# ──────────────────────────────────────────────────────────────────────
# Pairwise consistency training logic (from the notebook, adapted)
# ──────────────────────────────────────────────────────────────────────

class PairwiseConsistencyTraining:
    """Pairwise consistency training for few-step denoising with time embeddings.
    
    This is a stateless callable that can be shared across resolutions.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 1.0,
        initial_timesteps: int = 2,
        final_timesteps: int = 17,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps

    def __call__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        x: Tensor,
        y: Tensor,
        current_training_step: int,
        total_training_steps: int,
        **kwargs,
    ) -> dict:
        """Run one step of pairwise consistency training.
        
        Args:
            student_model: ConsistencyUNet being trained.
            teacher_model: EMA of student.
            x: Clean target data (B, C_out, H, W).
            y: Observations (B, C_in, H, W).
            current_training_step: Global step.
            total_training_steps: Total training steps.
        
        Returns:
            dict with keys: predicted, target, num_timesteps, steps
        """
        num_timesteps = timesteps_schedule(
            current_training_step, total_training_steps,
            self.initial_timesteps, self.final_timesteps,
        )
        num_timesteps = max(num_timesteps, 3)

        steps = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho,
            x.device, as_time=True,
        )
        noise = torch.randn_like(x)
        timestep_indices = torch.randint(0, num_timesteps - 2, (x.shape[0],), device=x.device)

        current_times = steps[timestep_indices]
        intermediate_times = steps[timestep_indices + 1]
        next_times = steps[timestep_indices + 2]

        # Student: denoise from intermediate → next
        sigma_inter = compute_sigma(intermediate_times, self.sigma_min, self.sigma_max)
        intermediate_noisy_x = x + pad_dims_like(sigma_inter, x) * noise

        predicted = consistency_forward_wrapper(
            student_model, intermediate_noisy_x, y,
            intermediate_times, next_times,
            self.sigma_data, self.sigma_min, self.sigma_max,
            **kwargs,
        )

        # Teacher: denoise from current → next (no grad)
        with torch.no_grad():
            sigma_curr = compute_sigma(current_times, self.sigma_min, self.sigma_max)
            current_noisy_x = x + pad_dims_like(sigma_curr, x) * noise

            target = consistency_forward_wrapper(
                teacher_model, current_noisy_x, y,
                current_times, next_times,
                self.sigma_data, self.sigma_min, self.sigma_max,
                **kwargs,
            )

        return {
            "predicted": predicted,
            "target": target,
            "num_timesteps": num_timesteps,
            "steps": steps,
        }


# ──────────────────────────────────────────────────────────────────────
# Lightning Module
# ──────────────────────────────────────────────────────────────────────

class Lit4dVarNet_CROSCIM_Consistency(Lit4dVarNet_CROSCIM_Supervised):
    """
    Consistency-model variant of the CROSCIM multi-resolution Lightning module.
    
    Replaces the iterative 4DVarNet solver with pairwise consistency training.
    Each resolution has its own student / teacher / ema_student UNet triplet.
    
    Constructor keyword arguments (on top of parent):
        consistency_config: dict with keys:
            sigma_min, sigma_max, rho, sigma_data,
            initial_timesteps, final_timesteps, total_training_steps,
            initial_ema_decay_rate, student_model_ema_decay_rate,
            lr, betas, lr_scheduler_start_factor, lr_scheduler_iters
    """

    def __init__(
        self,
        consistency_config: dict = None,
        add_bounds: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = consistency_config or {}

        self.add_bounds = add_bounds

        # Consistency hyper-parameters
        self.sigma_min = cfg.get("sigma_min", 0.002)
        self.sigma_max = cfg.get("sigma_max", 80.0)
        self.rho = cfg.get("rho", 7.0)
        self.sigma_data = cfg.get("sigma_data", 1.0)
        self.initial_timesteps = cfg.get("initial_timesteps", 2)
        self.final_timesteps = cfg.get("final_timesteps", 17)
        self.total_training_steps = cfg.get("total_training_steps", 10_000)
        self.initial_ema_decay_rate = cfg.get("initial_ema_decay_rate", 0.95)
        self.student_model_ema_decay_rate = cfg.get("student_model_ema_decay_rate", 0.99993)
        self._cm_lr = cfg.get("lr", 1e-4)
        self._cm_betas = tuple(cfg.get("betas", (0.9, 0.995)))
        self._cm_lr_scheduler_start_factor = cfg.get("lr_scheduler_start_factor", 1e-5)
        self._cm_lr_scheduler_iters = cfg.get("lr_scheduler_iters", 10_000)

        # ── sigma_max strategy ────────────────────────────────────────────────
        # Three modes, in priority order:
        #
        # 1. auto_sigma_max=True  →  calibrated from the first training batch
        #    by targeting a desired SNR = Var(x) / sigma_max² ≤ snr_target.
        #    Formula:  sigma_max = std(x) / sqrt(snr_target)
        #    Default snr_target=1e-2  →  sigma_max ≈ 10 * std(x).
        #
        # 2. sigma_max_per_res: {50: 80.0, 10: 10.0}  →  static override per res.
        #    Keys missing fall back to the global sigma_max.
        #
        # 3. sigma_max (global scalar, default 80.0)  →  applied to all resolutions.
        #
        # In auto mode, per-res overrides are still respected (auto only fills
        # resolutions NOT listed in sigma_max_per_res).
        self.auto_sigma_max = bool(cfg.get("auto_sigma_max", False))
        self.snr_target = float(cfg.get("snr_target", 1e-2))
        _per_res_raw = cfg.get("sigma_max_per_res", {})
        self.sigma_max_per_res: dict = {int(k): float(v) for k, v in _per_res_raw.items()}
        # Tracks which resolutions have already been auto-calibrated
        self._sigma_max_calibrated: set = set()

        # One PairwiseConsistencyTraining instance per resolution.
        # NOTE: PairwiseConsistencyTraining is NOT an nn.Module → plain dict.
        self.consistency_trainings: dict = {
            f"ct_x{res}": PairwiseConsistencyTraining(
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max_per_res.get(res, self.sigma_max),
                rho=self.rho,
                sigma_data=self.sigma_data,
                initial_timesteps=self.initial_timesteps,
                final_timesteps=self.final_timesteps,
            )
            for res in self.multires
        }
        # Back-compat alias (coarsest res)
        self.consistency_training = self.consistency_trainings[f"ct_x{self.multires[0]}"]

        # Build per-resolution teacher + ema_student from the student
        # The student UNets live inside self.solver.solvers["solver_xN"].unet
        self.teacher_solvers = nn.ModuleDict()
        self.ema_student_solvers = nn.ModuleDict()

        for res in self.multires:
            key = f"solver_x{res}"
            student_solver = self.solver.solvers[key]
            student_unet = student_solver.get_unet()

            # Teacher: copy of student, frozen
            teacher = copy.deepcopy(student_unet)
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()
            self.teacher_solvers[key] = teacher

            # EMA student: copy of student, frozen
            ema = copy.deepcopy(student_unet)
            for p in ema.parameters():
                p.requires_grad = False
            ema.eval()
            self.ema_student_solvers[key] = ema

        self.num_timesteps = self.initial_timesteps

        print(f"\n{'='*60}")
        print(f"Lit4dVarNet_CROSCIM_Consistency initialized:")
        print(f"  Resolutions: {self.multires}")
        _smax_info = ", ".join(
            (
                f"x{r}=auto(snr≤{self.snr_target})"
                if (self.auto_sigma_max and r not in self.sigma_max_per_res)
                else f"x{r}={self.sigma_max_per_res.get(r, self.sigma_max)}"
            )
            for r in self.multires
        )
        print(f"  Consistency config: sigma_min={self.sigma_min}, sigma_max: {_smax_info}")
        print(f"  final_timesteps={self.final_timesteps}, total_steps={self.total_training_steps}")
        for res in self.multires:
            key = f"solver_x{res}"
            n_params = sum(p.numel() for p in self.solver.solvers[key].parameters())
            print(f"  {key}: {n_params:,} params (student)")
        print(f"{'='*60}\n")

    # ── Override forward: capture inputs + call solver ────────────────

    def forward(self, batch, res=1):
        """At test time, stores observations into _bound_inputs if add_bounds=True."""
        if self.add_bounds and not self.training:
            res_key = f"patch_x{res}"
            if not hasattr(self, '_bound_inputs'):
                self._bound_inputs = {}
            if res_key not in self._bound_inputs:
                self._bound_inputs[res_key] = []
            self._bound_inputs[res_key].append(batch.input.detach().cpu())
        return self.solver.solvers[f"solver_x{res}"](batch)

    # ── Override on_test_start: set up boundary masks ─────────────────────

    def on_test_start(self):
        """Inherit parent setup (dataloader_keys) then build boundary masks if needed."""
        super().on_test_start()
        if not self.add_bounds:
            return
        for dataloader_idx, res in enumerate(self.multires):
            key = f"solver_x{res}"
            dl  = self.trainer.test_dataloaders[self.dataloader_keys[dataloader_idx]]
            ds  = dl.dataset
            patch_h  = ds.patch_dims['yc']
            patch_w  = ds.patch_dims['xc']
            stride_h = ds.strides.get('yc', patch_h)
            stride_w = ds.strides.get('xc', patch_w)
            border_h = (patch_h - stride_h) // 2
            border_w = (patch_w - stride_w) // 2
            print(f"[add_bounds] {key}: patch=({patch_h},{patch_w}), "
                  f"stride=({stride_h},{stride_w}), border=({border_h},{border_w})")
            self.solver.solvers[key].setup_mask(patch_h, patch_w, border_h, border_w)

    # ── Override base_step for consistency training ─────────────────────

    def base_step(self, batch, res, phase=""):
        """
        Replaces the parent's base_step with consistency training logic.
        
        The loss follows the original notebook pattern exactly:
          - consistency_training(student, teacher, x, y, step, total_steps)
          - loss = MSE(predicted_from_intermediate, target_from_current)
        
        No weighted-MSE, no interpolation/observation masks — the consistency
        loss operates on the *full* tensor (student prediction vs teacher
        target), which is the correct formulation for consistency models.
        
        Returns:
            (loss, out_dict) matching parent's signature so that
            multistep / step can work unchanged.
        """
        res_key = f"patch_x{res}"
        solver_key = f"solver_x{res}"

        # Format batch → sBatch(input, tgt)
        sbatch = self.format_batch_for_solver(batch, include_masks=self.include_masks, res=res)
        
        # Separate observations (y) and clean target (x)
        y = sbatch.input  # (B, C_in, H, W) — observations (may contain NaN)
        x = sbatch.tgt    # (B, C_out, H, W) — clean targets

        if self.training and phase == "train":
            # ── Auto-calibrate sigma_max from first batch (if enabled) ────────
            # Target SNR = Var(x) / sigma_max²  →  sigma_max = std(x) / sqrt(snr_target)
            # Only runs once per resolution; skipped if that res has a static override.
            if self.auto_sigma_max and res not in self._sigma_max_calibrated \
                    and res not in self.sigma_max_per_res:
                with torch.no_grad():
                    x_finite = x[x.isfinite()]
                    if x_finite.numel() > 1:
                        data_std = x_finite.std().item()
                        new_smax = data_std / math.sqrt(self.snr_target)
                        ct_key = f"ct_x{res}"
                        self.consistency_trainings[ct_key] = PairwiseConsistencyTraining(
                            sigma_min=self.sigma_min,
                            sigma_max=new_smax,
                            rho=self.rho,
                            sigma_data=self.sigma_data,
                            initial_timesteps=self.initial_timesteps,
                            final_timesteps=self.final_timesteps,
                        )
                        self._sigma_max_calibrated.add(res)
                        if self.trainer.is_global_zero:
                            print(
                                f"\n[sigma_max auto-calibration] res=x{res}: "
                                f"std(x)={data_std:.4f}, snr_target={self.snr_target} "
                                f"→ sigma_max={new_smax:.4f}"
                            )
                        self.log(f"sigma_max_x{res}", new_smax, on_step=True, on_epoch=False)

            # ── Consistency training (matches notebook training_step) ──
            student_unet = self.solver.solvers[solver_key].get_unet()
            teacher_unet = self.teacher_solvers[solver_key]

            # ── Build boundary conditioning from GT ring (add_bounds) ──────────
            kwargs = {}
            if self.add_bounds:
                solver  = self.solver.solvers[solver_key]
                # mask_boundary may be None during training (no test dataloaders yet).
                # Fall back to computing it from the train patch dims.
                if solver.mask_boundary is None:
                    ph = x.shape[2]; pw = x.shape[3]
                    bh = getattr(solver, 'border_h', 0) or (ph - ph * 4 // 5) // 2  # rough default
                    bw = getattr(solver, 'border_w', 0) or (pw - pw * 4 // 5) // 2
                    solver.setup_mask(ph, pw, bh, bw)
                mask_hw = solver.mask_boundary.to(x.device)               # (H, W)
                bound   = torch.nan_to_num(x) * mask_hw.unsqueeze(0).unsqueeze(0)  # (B, C, H, W)
                # Availability mask: ring pixels that are not NaN in the target
                mbound  = (mask_hw.unsqueeze(0).unsqueeze(0) *
                           x.isfinite().float())                           # (B, C, H, W)
                if self.training:
                    bound, mbound = random_boundary_dropout(
                        bound, mbound,
                        solver.border_h, solver.border_w,
                        p_drop_side=0.5,
                    )
                kwargs  = dict(boundaries=bound, mask_bound=mbound)

            ct = self.consistency_trainings[f"ct_x{res}"]
            output = ct(
                student_unet, teacher_unet,
                x.nan_to_num(), y,  # consistency training expects no NaNs in input
                self.global_step, self.total_training_steps,
                **kwargs,
            )
            self.num_timesteps = output["num_timesteps"]

            # Pure consistency loss: MSE(student prediction, teacher target), ice pixels only
            mask = torch.isfinite(x)
            if mask.any():
                loss = F.mse_loss(output["predicted"][mask], output["target"][mask])
            else:
                loss = (output["predicted"] * 0).sum()  # zero loss, keeps grad graph

            # Use student prediction as the output for downstream (multistep)
            out_tensor = self.solver.solvers[solver_key](sbatch)

            # Logging (same structure as notebook)
            if phase:
                self.log(f"{phase}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"num_timesteps_x{res}", float(self.num_timesteps), on_step=False, on_epoch=True)

            do_print = self.trainer.is_global_zero and (self.global_step % 50 == 0)
            if do_print:
                print(f"\n[Step {self.global_step:05d}] TRAIN | res=x{res} | "
                      f"num_timesteps={self.num_timesteps} | "
                      f"loss={loss.item():.6f}")

        else:
            # ── Validation / Test: consistency sampling with EMA student ──
            out_tensor = self.solver.solvers[solver_key](sbatch)
            
            # Reconstruction loss against ground truth (for monitoring only)
            mask = torch.isfinite(x)
            loss = F.mse_loss(torch.where(mask, out_tensor, torch.zeros_like(out_tensor)),
                              torch.where(mask, x, torch.zeros_like(x)))

            if phase:
                self.log(f"{phase}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # Split tensor → dict {pred_var: (B, T, H, W)} for multistep
        out = self.split_tensor_to_dict(out_tensor, res=res)

        return loss, out

    # ── Override step: skip auxiliary losses (grad, prior, tv, context) 

    def step(self, batch, res, phase=""):
        """For consistency training, the loss IS the consistency loss.
        No auxiliary losses (grad, prior, tv, context) — just the pure
        MSE(student_prediction, teacher_target) from base_step.
        """
        return self.base_step(batch, res=res, phase=phase)

    # ── EMA updates after each training batch ─────────────────────────

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher and EMA student after each training step."""
        # Teacher EMA decay rate (adapts with num_timesteps schedule)
        teacher_decay = ema_decay_rate_schedule(
            self.num_timesteps,
            self.initial_ema_decay_rate,
            self.initial_timesteps,
        )

        for res in self.multires:
            key = f"solver_x{res}"
            student_unet = self.solver.solvers[key].get_unet()

            # Update teacher
            update_ema_model_(self.teacher_solvers[key], student_unet, teacher_decay)
            # Update EMA student (fixed decay)
            update_ema_model_(self.ema_student_solvers[key], student_unet, self.student_model_ema_decay_rate)

        self.log("ema_decay_rate", teacher_decay, on_step=False, on_epoch=True)

    # ── Override configure_optimizers for consistency-specific optim ──

    def configure_optimizers(self):
        """Only optimize student UNet parameters."""
        # Dynamic total_training_steps: avoids num_timesteps saturating too early
        # (hardcoded 10k << actual steps when max_epochs * batches >> 10k)
        self.total_training_steps = self.trainer.estimated_stepping_batches
        print(f"[CM] total_training_steps set dynamically: {self.total_training_steps}")

        params = []
        for res in self.multires:
            key = f"solver_x{res}"
            student_unet = self.solver.solvers[key].get_unet()
            params.extend(filter(lambda p: p.requires_grad, student_unet.parameters()))

        opt = torch.optim.Adam(params, lr=self._cm_lr, betas=self._cm_betas)
        sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self._cm_lr_scheduler_start_factor,
            total_iters=self._cm_lr_scheduler_iters,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Override test_step to use EMA student for inference ───────────

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        """Same logic as parent test_step but uses the EMA student UNet.

        Also resets ``_bound_inputs`` per resolution when ``add_bounds=True``,
        so that ``_finalize_res`` can run sequential inference after gathering.
        """
        _dl_idx = 0 if dataloader_idx is None else dataloader_idx
        res     = self.multires[_dl_idx]
        res_key = f"patch_x{res}"

        # Reset input buffer at the start of each resolution's dataloader
        if self.add_bounds and batch_idx == 0:
            if not hasattr(self, '_bound_inputs'):
                self._bound_inputs = {}
            if not hasattr(self, '_bound_domain_masks'):
                self._bound_domain_masks = {}
            if _dl_idx == 0:
                self._bound_inputs = {}         # full reset at first dataloader
                self._bound_domain_masks = {}
            self._bound_inputs[res_key] = []
            self._bound_domain_masks[res_key] = []

        # Compute domain_invalid mask from the batch (same logic as parent test_step):
        # prefer models_XXX NaN pattern; fall back to land_mask.
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
                ).cpu()   # (B, 1, H, W)
            elif hasattr(batch, 'land_mask'):
                _domain_invalid = (batch.land_mask == 1.).cpu()   # (B, 1, H, W)
            else:
                _domain_invalid = None

            if _domain_invalid is not None:
                if res_key not in self._bound_domain_masks:
                    self._bound_domain_masks[res_key] = []
                for _b in range(_domain_invalid.shape[0]):
                    self._bound_domain_masks[res_key].append(_domain_invalid[_b])

        # Swap UNets to EMA student for inference
        original_unets = {}
        for res_key_inner in [f"solver_x{r}" for r in self.multires]:
            original_unets[res_key_inner] = self.solver.solvers[res_key_inner].unet
            self.solver.solvers[res_key_inner].unet = self.ema_student_solvers[res_key_inner]

        try:
            result = super().test_step(batch, batch_idx, dataloader_idx)
        finally:
            # Restore original UNets
            for key, unet in original_unets.items():
                self.solver.solvers[key].unet = unet

        return result

    # ── Sequential boundary-aware inference helpers ───────────────────

    @torch.no_grad()
    def _apply_sequential_inference(self, res, inputs, coords, stacked, domain_masks=None):
        """Run sequential boundary-conditioned inference; update prediction channels.

        Args:
            res          : resolution integer (e.g. 10)
            inputs       : list of ``(C_in, H, W)`` cpu tensors — raw observations per patch
            coords       : list of ``(yc_1d, xc_1d)`` tuples per patch
            stacked      : list of ``(V, T, H, W)`` cpu tensors (pred channels first, then tgt)
            domain_masks : optional list of ``(1, H, W)`` bool tensors — True where pixel is
                           invalid (land / outside model domain).  When provided, applied to
                           freshly-computed predictions to restore the domain mask that the
                           parent test_step normally applies.  Derived from the models_XXX NaN
                           pattern (or land_mask) at collection time.

        Returns:
            new_stacked : list with prediction channels replaced by boundary-conditioned output
        """
        solver_key = f"solver_x{res}"
        solver = self.solver.solvers[solver_key]

        if solver.mask_boundary is None:
            raise RuntimeError(
                f"setup_mask() not called for {solver_key}. "
                "Ensure on_test_start ran with add_bounds=True."
            )

        mask_cpu  = solver.mask_boundary.cpu()
        border_h  = solver.border_h
        border_w  = solver.border_w
        device    = next(solver.unet.parameters()).device
        n_tgt     = len(self._get_target_vars_for_resolution(res))
        c_out     = solver.n_output_channels   # = n_tgt * T

        order = raster_order_from_coords(coords)
        cache: dict = {}
        new_stacked = list(stacked)   # shallow copy; items replaced below

        for b_idx, iy, ix in order:
            y_single = inputs[b_idx].unsqueeze(0).to(device)   # (1, C_in, H, W)

            bound, mbound = build_boundary_conditioning(
                cache, iy, ix, mask_cpu, border_h, border_w,
                c_out=c_out,
            )
            bound  = bound.unsqueeze(0).to(device)   # (1, C_out, H, W)
            mbound = mbound.unsqueeze(0).to(device)

            pred = solver.sample_one(y_single, boundaries=bound, mask_bound=mbound)
            # pred: (1, C_out, H, W)

            # Store raw (C_out, H, W) in cache for neighbouring patches
            cache[(iy, ix)] = pred[0].cpu()

            # Replace prediction slice in stacked: first n_tgt entries along dim 0
            s = new_stacked[b_idx]          # (V, T, H, W)
            T = s.shape[1]
            H, W = s.shape[2], s.shape[3]
            pred_cpu = pred[0].cpu().view(n_tgt, T, H, W)   # (n_tgt, T, H, W)

            # ── Re-apply domain mask (land / model coverage) ──────────────
            # Use the pre-computed domain_invalid mask stored at collection time
            # (derived from models_XXX NaN pattern or land_mask — same logic as
            # parent test_step).  Fall back to tgt-channel NaN pattern only when
            # no mask was stored (e.g. legacy checkpoints).
            if domain_masks is not None and b_idx < len(domain_masks):
                # domain_masks[b_idx]: (1, H, W), True = invalid pixel
                _dinv = domain_masks[b_idx].expand(n_tgt, T, H, W)
            else:
                # Fallback: derive from tgt channels (less reliable but safe)
                tgt_channels = s[n_tgt:]   # (n_tgt, T, H, W)
                _dinv = ~tgt_channels.isfinite().any(dim=1, keepdim=True).expand_as(pred_cpu)
            pred_cpu = pred_cpu.masked_fill(_dinv, float('nan'))

            s_new = s.clone()
            s_new[:n_tgt] = pred_cpu
            new_stacked[b_idx] = s_new

        return new_stacked

    # ── Override _finalize_res: sequential inference when add_bounds ──

    def _finalize_res(self, dataloader_idx, idx_rec, write_netcdf=True):
        """Use sequential boundary-aware inference when ``add_bounds=True``,
        otherwise fall through to the parent's standard implementation.
        """
        if not self.add_bounds:
            return super()._finalize_res(dataloader_idx, idx_rec, write_netcdf)

        import torch.distributed as dist

        res     = self.multires[dataloader_idx]
        res_key = f"patch_x{res}"

        # ── Flatten locally accumulated lists ────────────────────────────────
        inputs       = list(itertools.chain(*self._bound_inputs.get(res_key, [])))
        # (each element of _bound_inputs is (B, C_in, H, W); chain iterates dim-0)
        times        = list(itertools.chain(*self.test_times[res_key]))
        coords       = list(itertools.chain(*self.test_coords[res_key]))
        stacked      = list(itertools.chain(*self.test_data[res_key]))
        domain_masks = list(getattr(self, '_bound_domain_masks', {}).get(res_key, []))
        # Each stacked item is now (V, T, H, W) after chain over the batch dim

        # ── Multi-GPU gathering ───────────────────────────────────────────────
        if self.trainer.world_size > 1:
            gathered = [None] * self.trainer.world_size
            dist.all_gather_object(
                gathered,
                {
                    'inputs':       [x.cpu() for x in inputs],
                    'times':        [t.cpu() for t in times],
                    'coords':       coords,
                    'stacked':      [s.cpu() for s in stacked],
                    'domain_masks': [m.cpu() for m in domain_masks],
                }
            )
            inputs       = [x for g in gathered for x in g['inputs']]
            times        = [t for g in gathered for t in g['times']]
            coords       = [c for g in gathered for c in g['coords']]
            stacked      = [s for g in gathered for s in g['stacked']]
            domain_masks = [m for g in gathered for m in g['domain_masks']]

        # ── Sequential inference on rank 0, broadcast to all ─────────────────
        if self.trainer.world_size > 1:
            if self.trainer.is_global_zero:
                new_stacked = self._apply_sequential_inference(
                    res, inputs, coords, stacked,
                    domain_masks=domain_masks or None,
                )
                result = self.aggregate_batches(
                    idx_rec, new_stacked, times, dataloader_idx,
                    metrics=False, write_netcdf=write_netcdf,
                    patch_coords=coords,
                )
                print(result)
            else:
                result = None
            container = [result]
            dist.broadcast_object_list(container, src=0)
            self.aggregate_results[res_key] = container[0]
        else:
            new_stacked = self._apply_sequential_inference(
                res, inputs, coords, stacked,
                domain_masks=domain_masks or None,
            )
            self.aggregate_results[res_key] = self.aggregate_batches(
                idx_rec, new_stacked, times, dataloader_idx,
                metrics=False, write_netcdf=write_netcdf,
                patch_coords=coords,
            )
            print(self.aggregate_results[res_key])

    # ── Utility: save/load EMA models ─────────────────────────────────
    def save_ema_models(self, base_path: str):
        """Save all EMA student models."""
        import os
        for res in self.multires:
            key = f"solver_x{res}"
            path = os.path.join(base_path, f"ema_{key}")
            # Wrap in ConsistencyUNet for save_pretrained
            ema_unet = self.ema_student_solvers[key]
            if hasattr(ema_unet, 'save_pretrained'):
                ema_unet.save_pretrained(path)
            else:
                os.makedirs(path, exist_ok=True)
                torch.save(ema_unet.state_dict(), os.path.join(path, "model.pt"))
        print(f"✅ EMA models saved to {base_path}")
