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
from .consistency_solver import (
    ConsistencyUNet,
    ConsistencyUNetConfig,
    ConsistencyUNetSolver,
    ConsistencyGradSolvers,
    consistency_forward_wrapper,
    compute_sigma,
    pad_dims_like,
    skip_scaling,
    output_scaling,
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        cfg = consistency_config or {}

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

        # Consistency training callable
        self.consistency_training = PairwiseConsistencyTraining(
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            sigma_data=self.sigma_data,
            initial_timesteps=self.initial_timesteps,
            final_timesteps=self.final_timesteps,
        )

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
        print(f"  Consistency config: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")
        print(f"  final_timesteps={self.final_timesteps}, total_steps={self.total_training_steps}")
        for res in self.multires:
            key = f"solver_x{res}"
            n_params = sum(p.numel() for p in self.solver.solvers[key].parameters())
            print(f"  {key}: {n_params:,} params (student)")
        print(f"{'='*60}\n")

    # ── Override forward: just call solver (for test / inference) ─────

    def forward(self, batch, res=1):
        """At test time, the solver does consistency sampling internally."""
        return self.solver.solvers[f"solver_x{res}"](batch)

    # ── Override base_step for consistency training ───────────────────

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
            # ── Consistency training (matches notebook training_step) ──
            student_unet = self.solver.solvers[solver_key].get_unet()
            teacher_unet = self.teacher_solvers[solver_key]

            output = self.consistency_training(
                student_unet, teacher_unet,
                x.nan_to_num(), y,  # consistency training expects no NaNs in input
                self.global_step, self.total_training_steps,
            )
            self.num_timesteps = output["num_timesteps"]

            # Pure consistency loss: MSE(student prediction, teacher target)
            mask = torch.isfinite(x)
            loss = F.mse_loss(torch.where(mask, output["predicted"], torch.zeros_like(output["predicted"])),
                              torch.where(mask, output["target"], torch.zeros_like(output["target"]))
                             )

            # Use student prediction as the output for downstream (multistep)
            out_tensor = self.solver.solvers[solver_key](sbatch)

            # Logging (same structure as notebook)
            if phase:
                self.log(f"{phase}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log("num_timesteps", float(self.num_timesteps), on_step=False, on_epoch=True)

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
        """Same logic as parent test_step but uses the EMA student UNet
        inside the ConsistencyUNetSolver for inference.
        
        We temporarily swap the solver's UNet to the EMA student before
        calling the parent's test_step, then swap back.
        """
        # Swap UNets to EMA student for inference
        original_unets = {}
        for res in self.multires:
            key = f"solver_x{res}"
            original_unets[key] = self.solver.solvers[key].unet
            self.solver.solvers[key].unet = self.ema_student_solvers[key]

        try:
            result = super().test_step(batch, batch_idx, dataloader_idx)
        finally:
            # Restore original UNets
            for key, unet in original_unets.items():
                self.solver.solvers[key].unet = unet

        return result

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
