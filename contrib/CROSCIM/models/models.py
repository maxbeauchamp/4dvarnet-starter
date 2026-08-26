from cmath import phase
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from datetime import datetime
from src.utils import get_last_time_wei, get_frcst_time_wei, get_linear_time_wei
from src.models import Lit4dVarNet
from contrib.CROSCIM.dataloaders.data import *
from dataclasses import dataclass
from collections import Counter
from scipy.interpolate import RegularGridInterpolator

# test push
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class sBatch:
    input: torch.Tensor
    tgt: torch.Tensor

def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # set to eval mode
    return model

class Lit4dVarNet_CROSCIM(Lit4dVarNet):

    def __init__(self,
            optim_weight,
            prior_weight,
            domain_limits,
            persist_rw=True, 
            frcst_lead=0,
            multires=[1], 
            tgt_vars=["tgt_sic","tgt_SIT"],
            satellite_vars=None,      # NEW: satellite variable config
            covariates=None,          # NEW: covariates config
            var_mapping=None,         # NEW: mapping for initialization
            norm_stats=None,          # Already exists
            norm_stats_covs=None,
            training_strategy='progressive',  # NEW PARAMETER
            include_masks=False,
            normalize_anomaly=True,  # instance-normalise anomaly before fine-res solver
            normalize_anomaly_patch_only=True,  # scale per-patch (batch sample) instead of pooled over the whole batch
            condition_on_scale=False,  # feed the coarse-field local scale as an extra input channel instead of hard-normalising the anomaly
            len_daw=None,  # optional override of the per-resolution crop_daw() window length ({res: n_timesteps}); defaults to the maxlen_daw/step-based schedule below
            save_obs_vars=False,  # also save raw satellite obs (asip_sic, cimr_*, cristal_*) in the test NetCDF, coarsest resolution only
            *args, **kwargs):

        # training_strategy options: 'simultaneous', 'progressive', 'hybrid'

        super().__init__(*args, **kwargs)

        # Store training strategy
        self.training_strategy = training_strategy
        self.normalize_anomaly = normalize_anomaly
        self.normalize_anomaly_patch_only = normalize_anomaly_patch_only
        self.condition_on_scale = condition_on_scale
        self.save_obs_vars = save_obs_vars

        # Store variable configuration
        self.satellite_vars = satellite_vars or DEFAULT_VAR_GROUPS
        self.covariates = covariates or DEFAULT_COVARIATES
        
        # Convert var_mapping to dict (handle OmegaConf)
        from omegaconf import OmegaConf
        if var_mapping is not None:
            if hasattr(var_mapping, '_metadata'):
                self.var_mapping = OmegaConf.to_container(var_mapping, resolve=True)
            else:
                self.var_mapping = dict(var_mapping)
        else:
            self.var_mapping = {}
        
        # Process tgt_vars: can be resolution-specific or global (handle OmegaConf)
        if tgt_vars is not None:
            if hasattr(tgt_vars, '_metadata'):
                self.tgt_vars_config = OmegaConf.to_container(tgt_vars, resolve=True)
            else:
                self.tgt_vars_config = tgt_vars
        else:
            self.tgt_vars_config = ["tgt_sic", "tgt_SIT"]
        
        # Detect if tgt_vars is resolution-specific
        if isinstance(self.tgt_vars_config, dict):
            # Resolution-specific: {"patch_x50": ["tgt_sic"], "patch_x10": ["tgt_sic", "tgt_SIT"]}
            self.tgt_vars = self._get_all_target_vars()
        else:
            # Global tgt_vars
            self.tgt_vars = self.tgt_vars_config
         
        # Construct input_vars list
        self.input_vars = []
        for source, vars in self.satellite_vars.items():
            for var in vars:
                self.input_vars.append(f"{source}_{var}")
        if self.covariates:
            self.input_vars.extend(self.covariates)

        self.include_masks = include_masks
        self.frcst_lead = frcst_lead
        self.domain_limits = domain_limits
        self.multires = multires
        self.maxlen_daw = 15
        #self.maxlen_daw = self.trainer.datamodule.test_dataloader()[f"patch_x{self.multires[0]}"].dataset.patch_dims["time"]
        if len_daw is not None:
            self.len_daw = dict(len_daw)
        else:
            n = len(self.multires)
            step = max(1, self.maxlen_daw // n)
            self.len_daw = {
                    r: max(1, self.maxlen_daw - i * step)
                    for i, r in enumerate(self.multires)
            }
        self._norm_stats = norm_stats
        self._norm_stats_cov = norm_stats_covs

        # Single function to process both weight dicts
        self.optim_weight = self._process_weights(optim_weight, prefix='_optim_weight')
        self.prior_weight = self._process_weights(prior_weight, prefix='_prior_weight')
        
        print(f"\n[Model Init] Instantiated weights:")
        for res_key in self.optim_weight.keys():
            weight = self.optim_weight[res_key]
            print(f"  {res_key}: shape={weight.shape}, device={weight.device}, dtype={weight.dtype}")

        # Dictionnaire d'équivalences : var canonique → liste d'alias
        self.equivalence_map = {
            "SIC": ["sic", "SIC", "sea_ice_concentration"],
            "SIT": ["sit", "SIT", "sea_ice_thickness"]
        }

        # Move all solvers to device once
        for res in self.multires:
            if f"solver_x{res}" in self.solver.solvers:
                self.solver.solvers[f"solver_x{res}"] = self.solver.solvers[f"solver_x{res}"].to(device)        

        # Create directory for debug plots
        self.debug_plot_dir = Path("debug_plots")
        self.debug_plot_dir.mkdir(exist_ok=True)
        self.plot_counter = 0  # Counter for unique filenames
        self.hook_backward = False  # Flag to control backward hook

        # Loss balancing configuration
        self.loss_target_ratios = {
            'base': 0.9,      # 80% of total loss
            'grad': 0.1,      # 10% of total loss
            'prior': 0.,     # 5% of total loss
            'tv': 0.,#0.10,       3356232 # 10% of total loss
            'context': 0.#0.05    # 5% of total loss
        }
        
        # Running averages for auto-balancing (EMA with alpha=0.1)
        self.register_buffer('loss_ema', torch.zeros(5))  # [base, grad, prior, tv, context]
        self.ema_alpha = 0.1
        self.loss_names = ['base', 'grad', 'prior', 'tv', 'context']
    
    def _get_all_target_vars(self):
        """Extract all unique target variables from resolution-specific config, preserving order."""
        if isinstance(self.tgt_vars_config, dict):
            # Use a list to preserve order (not a set!)
            all_vars = []
            seen = set()
            # Iterate through resolutions in a consistent order
            # Use the order from the dict itself (preserved in Python 3.7+)
            for res_key, res_vars in self.tgt_vars_config.items():
                for var in res_vars:
                    if var not in seen:
                        all_vars.append(var)
                        seen.add(var)
            print(f"\n[DEBUG _get_all_target_vars] Collected vars in order: {all_vars}")
            print(f"[DEBUG _get_all_target_vars] From tgt_vars_config: {self.tgt_vars_config}")
            return all_vars
        else:
            return self.tgt_vars_config
    
    def _get_target_vars_for_resolution(self, res):
        """
        Get target_vars for a specific resolution.
        
        Args:
            res: resolution factor (e.g., 50 for patch_x50)
            
        Returns:
            list: target variables for this resolution
        """
        if isinstance(self.tgt_vars_config, dict):
            res_key = f"patch_x{res}"
            result = self.tgt_vars_config.get(res_key, [])
            return result
        else:
            # Global tgt_vars apply to all resolutions
            return self.tgt_vars_config

    def _get_obs_var_names(self, dataloader_idx):
        """Raw satellite obs field names (e.g. 'asip_sic') to include in the
        test NetCDF, when save_obs_vars=True — coarsest resolution only
        (dataloader_idx == 0), for diagnostic visualisation of the raw inputs."""
        if not (self.save_obs_vars and dataloader_idx == 0):
            return []
        return [f"{source}_{var}" for source, vars_list in self.satellite_vars.items() for var in vars_list]

    def _process_weights(self, weight_dict, prefix='_weight'):
        """
        Process weight dict (handles callable, tensor, ndarray, DictConfig).
        Registers as buffer and returns processed dict.
        """
        processed = {}
        
        for res_key, weight_fn in weight_dict.items():
            # Convert to tensor
            weight_tensor = self._to_tensor(weight_fn)
            
            # Register as buffer (auto device management)
            buffer_name = f'{prefix}_{res_key.replace(".", "_").replace("-", "_")}'
            self.register_buffer(buffer_name, weight_tensor)
            processed[res_key] = getattr(self, buffer_name)
        
        return processed

    def _to_tensor(self, weight_fn):
        """
         Convert any weight type to tensor.
        Handles: torch.Tensor, np.ndarray, callable, DictConfig, or raw values.
        """
        # Already a tensor
        if isinstance(weight_fn, torch.Tensor):
            return weight_fn
        
        # Numpy array
        if isinstance(weight_fn, np.ndarray):
            return torch.from_numpy(weight_fn).float()
        
        # Callable (Hydra partial)
        if callable(weight_fn):
            result = weight_fn()
            return self._to_tensor(result)  # Recursive call
        
        # DictConfig with _target_ (needs instantiation)
        if hasattr(weight_fn, '_target_'):
            from hydra.utils import instantiate
            result = instantiate(weight_fn)
            return self._to_tensor(result)  # Recursive call
        
        # Fallback: try direct conversion
        return torch.tensor(weight_fn, dtype=torch.float32)
        
    def plot_batch_debug(self, sbatch, res, phase="train", batch_idx=0):
        """
        Plot input and target tensors for debugging.
        sbatch: sBatch with input and tgt tensors
        res: resolution
        phase: train/val
        batch_idx: batch index for filename
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Extract first sample from batch (B, C, H, W) -> (C, H, W)
        inp = sbatch.input[0].detach().cpu().numpy()  # (C, H, W)

        tgt = sbatch.tgt[0].detach().cpu().numpy()    # (C, H, W)
        
        # Determine time steps and variables
        n_channels_inp = inp.shape[0]
        n_channels_tgt = tgt.shape[0]
        
        # Assume 15 time steps
        n_time = 15
        n_vars_inp = n_channels_inp // n_time
        n_vars_tgt = n_channels_tgt // n_time
        
        # Reshape: (C, H, W) -> (V, T, H, W)
        inp_reshaped = inp.reshape(n_vars_inp, n_time, inp.shape[1], inp.shape[2])
        tgt_reshaped = tgt.reshape(n_vars_tgt, n_time, tgt.shape[1], tgt.shape[2])
        
        # === PLOT INPUT ===
        fig_inp, axes_inp = plt.subplots(n_vars_inp, n_time, 
                                         figsize=(n_time * 2, n_vars_inp * 2))
        if n_vars_inp == 1:
            axes_inp = axes_inp.reshape(1, -1)
        
        fig_inp.suptitle(f'INPUT - Res {res} - {phase} - Batch {batch_idx}', fontsize=16)
        
        for v in range(n_vars_inp):
            for t in range(n_time):
                ax = axes_inp[v, t]
                data = inp_reshaped[v, t]
                
                # Handle NaN values for visualization
                vmin, vmax = np.nanpercentile(data, [2, 98])
                
                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f'V{v} T{t}', fontsize=8)
                ax.axis('off')
                
                # Add colorbar for first column
                if t == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        filename_inp = self.debug_plot_dir / f'input_res{res}_{phase}_batch{self.plot_counter:04d}.png'
        plt.savefig(filename_inp, dpi=100, bbox_inches='tight')
        plt.close(fig_inp)
        
        # === PLOT TARGET ===
        fig_tgt, axes_tgt = plt.subplots(n_vars_tgt, n_time,
                                         figsize=(n_time * 2, n_vars_tgt * 2))
        if n_vars_tgt == 1:
            axes_tgt = axes_tgt.reshape(1, -1)
        
        fig_tgt.suptitle(f'TARGET - Res {res} - {phase} - Batch {batch_idx}', fontsize=16)
        
        for v in range(n_vars_tgt):
            for t in range(n_time):
                ax = axes_tgt[v, t]
                data = tgt_reshaped[v, t]
                
                vmin, vmax = np.nanpercentile(data, [2, 98])
                
                im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax.set_title(f'V{v} T{t}', fontsize=8)
                ax.axis('off')
                
                if t == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        filename_tgt = self.debug_plot_dir / f'target_res{res}_{phase}_batch{self.plot_counter:04d}.png'
        plt.savefig(filename_tgt, dpi=100, bbox_inches='tight')
        plt.close(fig_tgt)
        
        print(f"Saved debug plots: {filename_inp.name} and {filename_tgt.name}")

    def plot_input_target_mapping_debug(self, batch, res, phase="train", batch_idx=0):
        """
        Plot only input variables that are mapped to target variables.
        Shows the correspondence between source observations and targets.
        
        Args:
            batch: TrainingItem with all variables
            res: resolution
            phase: train/val/test
            batch_idx: batch index for filename
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Extract first sample from batch
        batch_dict = batch._asdict()
        
        # Determine which input variables are mapped to targets
        # var_mapping: {'tgt_sic': 'asip_sic', 'tgt_SIT': 'cimr_SIT'}
        input_target_pairs = []
        res_key = f"patch_x{res}"
        # Get resolution-specific mapping
        if isinstance(self.var_mapping, dict) and res_key in self.var_mapping:
            mapping = self.var_mapping[res_key]
        else:
            # Fallback to base var_mapping (for backward compatibility)
            mapping = self.var_mapping
        
        for tgt_var, src_var in mapping.items():
            if src_var in batch_dict and tgt_var in batch_dict:
                input_target_pairs.append((src_var, tgt_var))
        
        if not input_target_pairs:
            print(f" No input-target pairs found in batch for plotting")
            return
        
        n_pairs = len(input_target_pairs)
        
        # Extract data: (B, T, H, W) -> (T, H, W) for first sample
        plot_data = []
        for src_var, tgt_var in input_target_pairs:
            src_data = batch_dict[src_var][0].detach().cpu().numpy()  # (T, H, W)
            tgt_data = batch_dict[tgt_var][0].detach().cpu().numpy()  # (T, H, W)
            plot_data.append((src_var, src_data, tgt_var, tgt_data))
        
        n_time = plot_data[0][1].shape[0]  # Number of time steps
        
        # === CREATE FIGURE ===
        # Layout: n_pairs rows x n_time columns, but 2 subplots per cell (input + target)
        fig = plt.figure(figsize=(n_time * 3, n_pairs * 6))
        gs = fig.add_gridspec(n_pairs * 2, n_time, hspace=0.3, wspace=0.2)
        
        fig.suptitle(f'Input-Target Mapping - Res {res} - {phase} - Batch {batch_idx}', 
                    fontsize=16, y=0.995)
        
        for pair_idx, (src_var, src_data, tgt_var, tgt_data) in enumerate(plot_data):
            row_input = pair_idx * 2
            row_target = pair_idx * 2 + 1
            
            # Compute global vmin/vmax across all timesteps for consistent coloring
            src_vmin, src_vmax = np.nanpercentile(src_data, [2, 98])
            tgt_vmin, tgt_vmax = np.nanpercentile(tgt_data, [2, 98])
            
            for t in range(n_time):
                # === PLOT INPUT (SOURCE) ===
                ax_input = fig.add_subplot(gs[row_input, t])
                
                src_t = src_data[t]  # (H, W)
                
                im_input = ax_input.imshow(src_t, cmap='viridis', 
                                        vmin=src_vmin, vmax=src_vmax,
                                        interpolation='nearest')
                
                if t == 0:
                    ax_input.set_ylabel(f'{src_var}\n(Input)', fontsize=10, fontweight='bold')
                
                ax_input.set_title(f'T={t}', fontsize=9)
                ax_input.axis('off')
                
                # Add colorbar for first column
                if t == 0:
                    cbar = plt.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                
                # === PLOT TARGET ===
                ax_target = fig.add_subplot(gs[row_target, t])
                
                tgt_t = tgt_data[t]  # (H, W)
                
                im_target = ax_target.imshow(tgt_t, cmap='RdBu_r',
                                            vmin=tgt_vmin, vmax=tgt_vmax,
                                            interpolation='nearest')
                
                if t == 0:
                    ax_target.set_ylabel(f'{tgt_var}\n(Target)', fontsize=10, fontweight='bold')
                
                ax_target.axis('off')
                
                # Add colorbar for first column
                if t == 0:
                    cbar = plt.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                
                # === COMPUTE AND DISPLAY STATISTICS ===
                # Count valid observations
                src_valid = np.isfinite(src_t).sum()
                src_total = src_t.size
                tgt_valid = np.isfinite(tgt_t).sum()
                
                # Add text with statistics
                if t == n_time - 1:  # Add stats on last column
                    stats_text = (
                        f'Input valid: {src_valid}/{src_total} ({100*src_valid/src_total:.1f}%)\n'
                        f'Target valid: {tgt_valid}/{src_total} ({100*tgt_valid/src_total:.1f}%)'
                    )
                    ax_target.text(1.05, 0.5, stats_text, 
                                transform=ax_target.transAxes,
                                fontsize=8, verticalalignment='center',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save figure
        filename = self.debug_plot_dir / f'mapping_res{res}_{phase}_batch{self.plot_counter:04d}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f" Saved input-target mapping plot: {filename.name}")

    def plot_tv_loss_zones(self, pred, mask_interp, mask_obs, 
                        dilated_interp, dilated_obs, boundary_mask,
                        mask_yy, mask_xx, grad_yy, grad_xx):
        """
        Plot zones where TV loss is computed for debugging.
        
        Args:
            pred: (B, T, H, W) prediction
            mask_interp: original interpolation mask
            mask_obs: original observation mask
            dilated_interp: dilated interpolation mask
            dilated_obs: dilated observation mask
            boundary_mask: intersection of dilated masks
            mask_yy: vertical gradient mask (H-2, W)
            mask_xx: horizontal gradient mask (H, W-2)
            grad_yy: vertical second derivatives
            grad_xx: horizontal second derivatives
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        
        # Extract first sample and middle timestep for visualization
        B, T, H, W = pred.shape
        t_idx = T // 2
        
        # Move to CPU and convert to numpy
        pred_np = pred[0, t_idx].detach().cpu().numpy()
        mask_interp_np = mask_interp[0, t_idx].cpu().numpy()
        mask_obs_np = mask_obs[0, t_idx].cpu().numpy()
        dilated_interp_np = dilated_interp[0, t_idx].cpu().numpy()
        dilated_obs_np = dilated_obs[0, t_idx].cpu().numpy()
        boundary_mask_np = boundary_mask[0, t_idx].cpu().numpy()
        
        # Create composite mask for visualization
        # 0: neither, 1: interp only, 2: obs only, 3: boundary (where TV is computed)
        composite = np.zeros_like(mask_interp_np, dtype=int)
        composite[mask_interp_np] = 1
        composite[mask_obs_np] = 2
        composite[boundary_mask_np] = 3
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 1. Original prediction
        ax = axes[0, 0]
        vmin, vmax = np.nanpercentile(pred_np, [2, 98])
        im = ax.imshow(pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Prediction (t={t_idx})', fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 2. Original masks
        ax = axes[0, 1]
        mask_rgb = np.zeros((*mask_interp_np.shape, 3))
        mask_rgb[mask_interp_np] = [1, 0, 0]  # Red: interpolation
        mask_rgb[mask_obs_np] = [0, 0, 1]     # Blue: observations
        ax.imshow(mask_rgb)
        ax.set_title('Original Masks\n(Red=Interp, Blue=Obs)', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # 3. Dilated masks
        ax = axes[0, 2]
        dilated_rgb = np.zeros((*dilated_interp_np.shape, 3))
        dilated_rgb[dilated_interp_np] = [1, 0.5, 0.5]  # Light red
        dilated_rgb[dilated_obs_np] = [0.5, 0.5, 1]     # Light blue
        ax.imshow(dilated_rgb)
        ax.set_title('Dilated Masks\n(Expanded zones)', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # 4. Boundary mask (intersection)
        ax = axes[0, 3]
        cmap_boundary = plt.cm.colors.ListedColormap(['white', 'yellow'])
        ax.imshow(boundary_mask_np, cmap=cmap_boundary, vmin=0, vmax=1)
        ax.set_title(f'Boundary Mask\n({boundary_mask_np.sum()} pixels)', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # 5. Composite view (all zones)
        ax = axes[1, 0]
        cmap_composite = plt.cm.colors.ListedColormap(['white', 'red', 'blue', 'yellow'])
        bounds = [0, 1, 2, 3, 4]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap_composite.N)
        im = ax.imshow(composite, cmap=cmap_composite, norm=norm)
        ax.set_title('Composite View', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Custom legend
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Neither'),
            Patch(facecolor='red', label='Interpolation only'),
            Patch(facecolor='blue', label='Observation only'),
            Patch(facecolor='yellow', label='Boundary (TV computed here)')
        ]
        ax.legend(handles=legend_elements, loc='center', fontsize=8)
        
        # 6. Vertical gradient mask (mask_yy)
        ax = axes[1, 1]
        # Pad to original size for visualization
        mask_yy_padded = np.pad(mask_yy[0, t_idx].cpu().numpy(), 
                                ((1, 1), (0, 0)), mode='constant')
        cmap_grad = plt.cm.colors.ListedColormap(['white', 'green'])
        ax.imshow(mask_yy_padded, cmap=cmap_grad, vmin=0, vmax=1)
        ax.set_title(f'Vertical Gradient Mask\n({mask_yy[0, t_idx].sum().item():.0f} pixels)', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # 7. Horizontal gradient mask (mask_xx)
        ax = axes[1, 2]
        mask_xx_padded = np.pad(mask_xx[0, t_idx].cpu().numpy(),
                                ((0, 0), (1, 1)), mode='constant')
        ax.imshow(mask_xx_padded, cmap=cmap_grad, vmin=0, vmax=1)
        ax.set_title(f'Horizontal Gradient Mask\n({mask_xx[0, t_idx].sum().item():.0f} pixels)',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # 8. Combined gradient zones
        ax = axes[1, 3]
        combined_grad = np.zeros((H, W))
        # Pad masks to original size
        mask_yy_full = np.pad(mask_yy[0, t_idx].cpu().numpy(), ((1, 1), (0, 0)), mode='constant')
        mask_xx_full = np.pad(mask_xx[0, t_idx].cpu().numpy(), ((0, 0), (1, 1)), mode='constant')
        combined_grad[mask_yy_full] = 1
        combined_grad[mask_xx_full] = 1
        ax.imshow(combined_grad, cmap=cmap_grad, vmin=0, vmax=1)
        total_pixels = mask_yy[0, t_idx].sum().item() + mask_xx[0, t_idx].sum().item()
        ax.set_title(f'Combined Gradient Zones\n({total_pixels:.0f} total pixels)',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Global title with statistics
        n_interp = mask_interp_np.sum()
        n_obs = mask_obs_np.sum()
        n_boundary = boundary_mask_np.sum()
        n_total = H * W
        
        fig.suptitle(
            f'TV Loss Computation Zones (Step {self.global_step})\n'
            f'Interp: {n_interp}/{n_total} ({100*n_interp/n_total:.1f}%) | '
            f'Obs: {n_obs}/{n_total} ({100*n_obs/n_total:.1f}%) | '
            f'Boundary: {n_boundary}/{n_total} ({100*n_boundary/n_total:.1f}%)',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save figure
        filename = self.debug_plot_dir / f'tv_loss_zones_step{self.global_step:06d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f" Saved TV loss visualization: {filename.name}")

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)

    @property
    def norm_stats_covs(self):
        if self._norm_stats_covs is not None:
            return self._norm_stats_covs
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats_covs()
        return (0., 1.)

    def configure_optimizers(self):
        if self.opt_fn is not None:
            return self.opt_fn(self)
        else:
            params = []
            for model in self.solver.solvers.values():
                params += list(filter(lambda p: p.requires_grad, model.parameters()))
            opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
            return {
               "optimizer": opt,
               "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-6),
            }

    def crop_daw(self, item_dict, res):
        last = self.len_daw[res]
        for var in item_dict:
            data = item_dict[var]
            if isinstance(data, torch.Tensor) and data.ndim == 4 and data.shape[1] > 1:
                item_dict[var] = data[:,-last:,:,:]
            if var=="time" and data.ndim == 3 :
                item_dict[var] = data[:,:,-last:]
        return item_dict

    def modify_multires_batch(self, batch):
        """
        Applique un masquage temporel sur toutes les résolutions du batch multi-échelle.
        """
        # Variables satellites : seules celles-ci reçoivent le masquage frcst_lead
        satellite_var_names = {
            f"{src}_{var}"
            for src, vars in self.satellite_vars.items()
            for var in vars
        }

        for key, item in batch.items():
            if not key.startswith("patch_x"):
                continue  # sécurité pour ne traiter que les bons items

            item_dict = item._asdict()
            res = int(key[7:])
            item_dict = self.crop_daw(item_dict, res)
            new_item = {}

            for var in item_dict:
                data = item_dict[var]
                if isinstance(data, torch.Tensor) and data.ndim == 4 and data.shape[1] > 1:
                    # Masquage temporel : uniquement les variables satellites
                    if self.frcst_lead is not None and self.frcst_lead > 0 and var in satellite_var_names:
                        data[:, -self.frcst_lead:, :, :] = torch.nan
                    new_item[var] = data.to(device)
                else:
                    new_item[var] = data  # gardé tel quel (land_mask, latv, lonv...)
            # Reconstruction de l'item
            batch[key] = type(item)(**new_item)

        return batch

    def modify_batch(self, batch, res):
        """
        Applique un masquage temporel sur le batch
        """
        # Variables satellites : seules celles-ci reçoivent le masquage frcst_lead
        satellite_var_names = {
            f"{src}_{var}"
            for src, vars in self.satellite_vars.items()
            for var in vars
        }

        item_dict = batch._asdict()
        item_dict = self.crop_daw(item_dict, res)

        new_item = {}
        for var in item_dict:
            data = item_dict[var]
            if isinstance(data, torch.Tensor) and data.ndim == 4 and data.shape[1] > 1:
                # Masquage temporel : uniquement les variables satellites
                if self.frcst_lead is not None and self.frcst_lead > 0 and var in satellite_var_names:
                    data[:, -self.frcst_lead:, :, :] = torch.nan
                new_item[var] = data.to(device)
            else:
                new_item[var] = data  # gardé tel quel (land_mask, latv, lonv...)
        # Reconstruction de l'item
        batch = type(batch)(**new_item)
        return batch


    def format_batch_for_solver(self, batch, include_masks=False, res=None, scale_channel=None):
        """
        À partir d'un batch de type TrainingItem, retourne un dictionnaire avec :
        - 'input' : concaténation des input_vars (satellite + covariates)
        - 'tgt'   : concaténation des variables de tgt_vars

        Args:
            batch: TrainingItem namedtuple with all variables
            include_masks: bool, if True intercale data et mask pour chaque variable
                        [data_var1, mask_var1, data_var2, mask_var2, ...]
                        if False, juste les données comme avant
            res: resolution key (e.g., 50 for patch_x50) to get resolution-specific target vars
            scale_channel: optional (B, T, H, W) tensor from ``compute_scale_channel``,
                        appended as an extra input channel when ``condition_on_scale``
                        is used (see the multi-res anomaly loop in ``multistep``).

        Returns:
            sBatch with input and tgt tensors
        """
        # Get resolution-specific target vars
        if res is not None:
            tgt_vars = self._get_target_vars_for_resolution(res)
        else:
            tgt_vars = self.tgt_vars
        
        input_tensors = []
        
        if include_masks:
            # OPTION 2: Intercaler data et mask pour chaque variable
            for var in self.input_vars:
                if hasattr(batch, var):
                    data = getattr(batch, var)
                    # Skip empty tensors (variables not present at this resolution)
                    if torch.is_tensor(data) and data.numel() == 0:
                        continue
                    input_tensors.append(data)
                    
                    # Créer le masque de validité (1 = valide, 0 = NaN)
                    mask = data.isfinite().float()
                    input_tensors.append(mask)
                else:
                    print(f"  Warning: batch missing input variable '{var}'")
        else:
            # Mode original: seulement les données
            for var in self.input_vars:
                if hasattr(batch, var):
                    data = getattr(batch, var)
                    # Skip empty tensors (variables not present at this resolution)
                    if torch.is_tensor(data) and data.numel() == 0:
                        continue
                    input_tensors.append(data)
                else:
                    print(f"  Warning: batch missing input variable '{var}'")
        
        # Collecter les targets (inchangé)
        tgt_tensors = []
        for var in tgt_vars:
            if hasattr(batch, var):
                data = getattr(batch, var)
                # Skip empty tensors (variables not present at this resolution)
                if torch.is_tensor(data) and data.numel() == 0:
                    continue
                tgt_tensors.append(data)
            else:
                raise ValueError(f" Batch missing target variable '{var}'")
        
        #  DEBUG: Vérifier les dimensions (tous les 50 steps)
        if include_masks and self.global_step % 50 == 0 and self.trainer.is_global_zero:
            input_final = torch.cat(input_tensors, dim=1)
            n_vars = len(self.input_vars)
            n_channels = input_final.shape[1]
            n_time = n_channels // (2 * n_vars)  # 2 car data + mask
            
            print(f"\n[Step {self.global_step}] format_batch_for_solver (include_masks=True):")
            print(f"  Variables: {n_vars}")
            print(f"  Timesteps: {n_time}")
            print(f"  Total channels: {n_channels} (= {n_vars} × {n_time} × 2)")
            print(f"  Spatial: {input_final.shape[2]} × {input_final.shape[3]}")
            print(f"  Expected UNet n_channels: {n_channels}")

        if scale_channel is not None:
            input_tensors.append(scale_channel)

        return sBatch(
            input=torch.cat(input_tensors, dim=1).float(),
            tgt=torch.cat(tgt_tensors, dim=1).float()
        )

    def update_batch_as_anomaly(self, batch, out):
        """
        Met à jour les valeurs du batch (namedtuple) avec les anomalies prédites dans out,
        en tenant compte des équivalences de noms de variables.
        """
    
        batch_dict = batch._asdict()
        for pred_var, coarse_prediction in out.items():
            if coarse_prediction is None:
                continue
            # Skip empty tensors from coarse resolution
            if torch.is_tensor(coarse_prediction) and coarse_prediction.numel() == 0:
                continue
                
            # Extrait la variable canonique (ex: "pred_sic" → "sic")
            canon_var = pred_var.replace("pred_", "") if pred_var.startswith("pred_") else pred_var
            aliases = self.equivalence_map.get(canon_var, [canon_var])
            # Compute the anomaly
            for batch_var in batch_dict:
                batch_tensor = batch_dict[batch_var]
                # Skip if batch variable is empty (doesn't exist at this resolution)
                if torch.is_tensor(batch_tensor) and batch_tensor.numel() == 0:
                    continue
                    
                for alias in aliases:
                    if batch_var.lower().endswith(alias.lower()):
                        batch_dict[batch_var] = batch_dict[batch_var] - coarse_prediction
                        break  # évite de mettre à jour plusieurs fois le même batch_var
    
        return type(batch)(**batch_dict)


    def normalize_anomaly_batch(self, batch, eps: float = 1e-3):
        """
        Instance-normalise the anomaly fields in *batch* so that each target
        variable has std ≈ 1 across the valid (non-NaN) pixels.

        Controlled by ``self.normalize_anomaly_patch_only``:
          - True  (default): std computed per sample (dim 0) — each patch gets
            its own scale. Avoids mixing patches with very different natural
            anomaly amplitude (e.g. ice edge vs. central pack) into a single
            shared factor, which otherwise produces visible seams once
            patches are stitched back together.
          - False: std pooled over the whole batch (legacy behaviour, one
            shared scalar for every sample).

        This is called right after ``update_batch_as_anomaly`` when training /
        running inference on a fine resolution.  The returned ``scale_dict``
        must be passed to ``denormalize_anomaly_predictions`` so that the
        solver output can be rescaled before adding the coarse prediction back.

        Parameters
        ----------
        batch : namedtuple
            Batch after anomaly subtraction.
        eps : float
            Floor for the scale factor to avoid division by zero in nearly
            flat (e.g. ice-free summer) patches.

        Returns
        -------
        batch_norm : same type as *batch*, with target variables rescaled.
        scale_dict : dict  {batch_var_name: scale_tensor}, shape (B, 1, 1, ...)
            if ``normalize_anomaly_patch_only`` else a 0-d scalar tensor.
        """
        batch_dict = batch._asdict()
        scale_dict = {}

        # Only rescale target variables (tgt_*, models_*) — leave obs and coords unchanged
        target_prefixes = tuple(
            [f"{src}_" for src in self.satellite_vars]
            + ["tgt_", "models_"]
        )

        for var_name, tensor in batch_dict.items():
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            if not any(var_name.startswith(p) or var_name.lower().startswith("tgt_") for p in target_prefixes):
                # Also normalise variables whose name matches an alias in equivalence_map
                canon_match = any(
                    var_name.lower().endswith(alias.lower())
                    for aliases in self.equivalence_map.values()
                    for alias in aliases
                )
                if not canon_match:
                    continue
            if tensor.ndim < 2:
                continue

            if self.normalize_anomaly_patch_only:
                # Std over the valid (finite) pixels of each sample separately
                scale = tensor.new_ones(tensor.shape[0])
                for b in range(tensor.shape[0]):
                    valid = tensor[b][tensor[b].isfinite()]
                    if valid.numel() >= 2:
                        scale[b] = valid.std().clamp(min=eps)
                scale = scale.view(-1, *([1] * (tensor.ndim - 1)))
            else:
                # Std pooled over all valid (finite) pixels in the batch
                valid = tensor[tensor.isfinite()]
                if valid.numel() < 2:
                    scale = tensor.new_tensor(1.0)
                else:
                    scale = valid.std().clamp(min=eps)

            scale_dict[var_name] = scale
            batch_dict[var_name] = tensor / scale

        return type(batch)(**batch_dict), scale_dict

    def denormalize_anomaly_predictions(self, out: dict, scale_dict: dict) -> dict:
        """
        Reverse the instance-normalisation applied by ``normalize_anomaly_batch``.

        Maps scale_dict keys (batch variable names such as ``tgt_SIT``) to
        prediction keys (``pred_SIT``) via the same suffix logic used elsewhere,
        then multiplies each prediction tensor by the corresponding scale.

        Parameters
        ----------
        out : dict  {pred_var: (B, T, H, W)}
        scale_dict : dict  {batch_var_name: scale_tensor}, either a 0-d scalar
            or shape (B, 1, 1, ...) — both broadcast correctly against
            ``out[pred_key]`` below.

        Returns
        -------
        out : dict (modified in-place, also returned for convenience)
        """
        # Only the TARGET variable's scale must be reapplied. Several inputs share
        # the same suffix (e.g. models_SIT, cristal_SIT, cimr_SIT all → "SIT"); using
        # all of them multiplied pred_SIT by the *product* of their scales. Restrict
        # to target variables so each prediction is rescaled by a single factor.
        if isinstance(self.tgt_vars, dict):
            target_names = {v for lst in self.tgt_vars.values() for v in lst}
        else:
            target_names = set(self.tgt_vars)
        for batch_var, scale in scale_dict.items():
            if batch_var not in target_names:
                continue
            # derive canonical suffix
            if '_' in batch_var:
                suffix = batch_var.split('_', 1)[1]   # "SIT"
            else:
                suffix = batch_var
            pred_key = f"pred_{suffix}"
            if pred_key in out:
                out[pred_key] = out[pred_key] * scale.to(out[pred_key].device)
        return out

    def compute_scale_channel(self, coarse_field: dict, eps: float = 1e-3):
        """
        Per-patch local scale derived from the coarse-resolution prediction
        (not the true target), meant to be fed to the fine-res solver as an
        extra input channel (see ``condition_on_scale``) instead of hard
        dividing/re-multiplying the anomaly (``normalize_anomaly``). Using the
        coarse field keeps this available at genuine inference time too, when
        the true anomaly isn't known.

        Parameters
        ----------
        coarse_field : dict {pred_var: tensor (B, T, H, W)}
            The coarser-resolution output, already interpolated + cropped to
            the current resolution's grid/time window (``interpolate_torch``
            + ``crop_daw``), so T already matches the current resolution.
        eps : float
            Floor for the scale factor.

        Returns
        -------
        scale_channel : tensor (B, T, H, W), or None if coarse_field is empty.
            The per-patch scale, broadcast over time and space, ready to be
            concatenated as an extra input channel.
        """
        tensors = [t for t in coarse_field.values()
                   if isinstance(t, torch.Tensor) and t.numel() > 0]
        if not tensors:
            return None
        ref = tensors[0]
        B = ref.shape[0]
        scale = ref.new_ones(B)
        for b in range(B):
            valid = torch.cat([t[b][t[b].isfinite()] for t in tensors])
            if valid.numel() >= 2:
                scale[b] = valid.std().clamp(min=eps)
        return scale.view(B, 1, 1, 1).expand_as(ref)

    def interpolate_torch(self, coarse_data, xc_coarse, yc_coarse, xc_target, yc_target,
                        mode='bilinear', align_corners=True):
        """
        Interpolate coarse data to target grid using PyTorch grid_sample (batched).
        
        Args:
            coarse_data: (B, C, Hc, Wc) or dict of such tensors
            xc_coarse: (B, Wc) or (Wc,) - coarse x-coordinates
            yc_coarse: (B, Hc) or (Hc,) - coarse y-coordinates
            xc_target: (B, Wf) or (Wf,) - target x-coordinates
            yc_target: (B, Hf) or (Hf,) - target y-coordinates
            mode: 'bilinear' or 'nearest'
            align_corners: bool
        
        Returns:
            interpolated: (B, C, Hf, Wf) or dict of such tensors
        """
        
        def make_grid_batch(xc_coarse, yc_coarse, xc_target, yc_target):
            """
            Create normalized grid for torch grid_sample (batched version).
            """
            # Determine device from target coordinates (they should be on GPU)
            device = xc_target.device if isinstance(xc_target, torch.Tensor) else 'cpu'
            
            # Convert to tensors and move to device WITH EXPLICIT FLOAT32
            if not isinstance(xc_coarse, torch.Tensor):
                xc_coarse = torch.tensor(xc_coarse, dtype=torch.float32, device=device)
            else:
                xc_coarse = xc_coarse.to(device=device, dtype=torch.float32)
            
            if not isinstance(yc_coarse, torch.Tensor):
                yc_coarse = torch.tensor(yc_coarse, dtype=torch.float32, device=device)
            else:
                yc_coarse = yc_coarse.to(device=device, dtype=torch.float32)
            
            if not isinstance(xc_target, torch.Tensor):
                xc_target = torch.tensor(xc_target, dtype=torch.float32, device=device)
            else:
                xc_target = xc_target.to(device=device, dtype=torch.float32)
            
            if not isinstance(yc_target, torch.Tensor):
                yc_target = torch.tensor(yc_target, dtype=torch.float32, device=device)
            else:
                yc_target = yc_target.to(device=device, dtype=torch.float32)
            
            # Handle batched vs non-batched inputs
            if xc_coarse.ndim == 1:
                xc_coarse = xc_coarse.unsqueeze(0)  # (1, Wc)
            if yc_coarse.ndim == 1:
                yc_coarse = yc_coarse.unsqueeze(0)  # (1, Hc)
            if xc_target.ndim == 1:
                xc_target = xc_target.unsqueeze(0)  # (1, Wf)
            if yc_target.ndim == 1:
                yc_target = yc_target.unsqueeze(0)  # (1, Hf)
            
            B = xc_target.shape[0]
            
            # Expand to batch size if needed
            if xc_coarse.shape[0] == 1 and B > 1:
                xc_coarse = xc_coarse.expand(B, -1)
            if yc_coarse.shape[0] == 1 and B > 1:
                yc_coarse = yc_coarse.expand(B, -1)
            
            # Get bounds from coarse grid (per batch)
            x0 = xc_coarse[:, 0:1]  # (B, 1)
            x1 = xc_coarse[:, -1:]  # (B, 1)
            y0 = yc_coarse[:, 0:1]  # (B, 1)
            y1 = yc_coarse[:, -1:]  # (B, 1)
            
            dx = x1 - x0  # (B, 1)
            dy = y1 - y0  # (B, 1)
            
            # Normalize target coordinates to [-1, 1]
            xc_t = xc_target  # (B, Wf)
            yc_t = yc_target  # (B, Hf)
            
            # All tensors are now on the same device and dtype
            norm_x = 2.0 * (xc_t - x0) / dx - 1.0  # (B, Wf)
            norm_y = 2.0 * (yc_t - y0) / dy - 1.0  # (B, Hf)
            
            # Create meshgrid: (B, Hf, Wf)
            Hf = yc_t.shape[1]
            Wf = xc_t.shape[1]
            
            # Expand to meshgrid
            norm_x = norm_x.unsqueeze(1).expand(B, Hf, Wf)  # (B, Hf, Wf)
            norm_y = norm_y.unsqueeze(2).expand(B, Hf, Wf)  # (B, Hf, Wf)
            
            # Stack to (B, Hf, Wf, 2) - grid_sample expects (x, y) order
            grid = torch.stack([norm_x, norm_y], dim=-1)
            
            return grid
        
        # Determine device and dtype from coarse_data
        if isinstance(coarse_data, dict):
            sample_tensor = next(iter(coarse_data.values()))
        else:
            sample_tensor = coarse_data
        
        device = sample_tensor.device
        dtype = sample_tensor.dtype
        
        # Create grid (now all tensors will be on the same device and dtype)
        grid = make_grid_batch(xc_coarse, yc_coarse, xc_target, yc_target)  # (B, Hf, Wf, 2)
        
        # Ensure grid is on the same device AND dtype as data
        grid = grid.to(device=device, dtype=dtype)
        
        if isinstance(coarse_data, dict):
            interpolated = {}
            for key, data in coarse_data.items():
                # Ensure data is (B, C, H, W)
                if data.ndim == 3:
                    data = data.unsqueeze(1)  # Add channel dim
                
                # Ensure data is float32
                data = data.to(dtype=torch.float32)
                
                # Interpolate
                interpolated[key] = F.grid_sample(
                    data, grid, 
                    mode=mode, 
                    align_corners=align_corners,
                    padding_mode='border'
                )
            return interpolated
        else:
            # Ensure data is (B, C, H, W)
            if coarse_data.ndim == 3:
                coarse_data = coarse_data.unsqueeze(1)
            
            # Ensure coarse_data is on the correct device and dtype
            coarse_data = coarse_data.to(device=device, dtype=torch.float32)
            
            # Interpolate
            interpolated = F.grid_sample(
                coarse_data, grid,
                mode=mode,
                align_corners=align_corners,
                padding_mode='border'
            )
            
            return interpolated

    def split_tensor_to_dict(self, tensor, res=None):
        """
        Découpe un tenseur interpolé (B, C, H, W) en dictionnaire {var: (B, T, H, W)}.
        Args:
            tensor: torch.Tensor de shape (B, C=T*V, H, W)
            res: resolution key (e.g., 50 for patch_x50) to get resolution-specific target vars
        Returns:
            dict {pred_var_name: tensor de shape (B, T, H, W)}
        """
        # Get resolution-specific target vars
        if res is not None:
            tgt_vars = self._get_target_vars_for_resolution(res)
        else:
            tgt_vars = self.tgt_vars
        
        B, C, H, W = tensor.shape
        V = len(tgt_vars)

        time_steps = C // V
        assert C == time_steps * V, f"Expected C={time_steps}×{V}, but got {C}"
    
        tensor_reshaped = tensor.view(B, V, time_steps, H, W)  # (B, V, T, H, W)
        tensor_reshaped = tensor_reshaped.permute(0, 2, 1, 3, 4)  # (B, T, V, H, W)
    
        out_dict = {}
        for i, var in enumerate(tgt_vars):
            # Replace any prefix before '_' with 'pred_'
            if '_' in var:
                var_suffix = var.split('_', 1)[1]  # Get everything after first '_'
                pred_var_name = f'pred_{var_suffix}'
            else:
                pred_var_name = f'pred_{var}'
        
            out_dict[pred_var_name] = tensor_reshaped[:, :, i]  # (B, T, H, W)
    
        return out_dict

    def training_step(self, batch, batch_idx):
        # Ne pas enregistrer le hook ici - utilisez on_after_backward à la place
        return self.multistep(batch, "train")[0]

    def on_after_backward(self):
        """Called after every backward pass. Diagnose missing gradients."""
        if not self.trainer.is_global_zero:
            return

        if self.global_step % 20 == 0:
            print(f"\n[Step {self.global_step}] ── Gradient audit ──────────────────────")
            for res in self.multires:
                solver_key = f"solver_x{res}"
                solver = self.solver.solvers[solver_key]
                params = list(solver.named_parameters())
                with_grad    = [(n, p) for n, p in params if p.requires_grad and p.grad is not None]
                no_grad_req  = [(n, p) for n, p in params if not p.requires_grad]
                grad_missing = [(n, p) for n, p in params if p.requires_grad and p.grad is None]
                zero_grad    = [(n, p) for n, p in params if p.requires_grad and p.grad is not None and p.grad.norm().item() == 0]

                print(f"  solver_x{res}:")
                print(f"    requires_grad=True  & grad ok   : {len(with_grad)}")
                print(f"    requires_grad=True  & grad None : {len(grad_missing)}"
                      + (f"  ← PROBLEM" if grad_missing else ""))
                print(f"    requires_grad=True  & grad==0   : {len(zero_grad)}"
                      + (f"  ← suspicious" if zero_grad else ""))
                print(f"    requires_grad=False (frozen)    : {len(no_grad_req)}")

                # Detail the missing ones (first 5)
                for name, p in grad_missing[:5]:
                    print(f"      !! no grad: {name}  shape={tuple(p.shape)}")
                for name, p in zero_grad[:5]:
                    print(f"      !! zero grad: {name}  shape={tuple(p.shape)}")
            print(f"────────────────────────────────────────────────────────")
    
    def validation_step(self, batch, batch_idx):
        return self.multistep(batch, "val")[0]

    def forward(self, batch, res=1):
        model = self.solver.solvers[f"solver_x{res}"]
        return model(batch)

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        res_idx = min(epoch // (self.trainer.max_epochs // len(self.multires)), len(self.multires) - 1)
        train_res = self.multires[res_idx]

        # When the training resolution switches, the loss magnitude changes
        # (more timesteps / higher resolution → different loss scale).
        # Reset all ModelCheckpoint "best" scores so the new phase starts fresh
        # and doesn't inherit stale thresholds from the previous resolution.
        prev_res_idx = getattr(self, '_prev_res_idx', None)
        if prev_res_idx is not None and res_idx != prev_res_idx:
            from pytorch_lightning.callbacks import ModelCheckpoint
            for cb in self.trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    cb.best_model_score = None
                    cb.best_model_path = ""
                    cb.best_k_models = {}
                    cb.kth_best_model_path = ""
                    cb.kth_value = None
            if self.global_rank == 0:
                print(f"  ↺  Resolution switch x{self.multires[prev_res_idx]} → x{train_res}: "
                      f"ModelCheckpoint best scores reset.")

            # Restart the LR scheduler from its own beginning at each
            # resolution switch (progressive mode only) -- otherwise a
            # single scheduler keeps running on the global step/epoch count,
            # independently of when each resolution's own training actually
            # begins (e.g. it may already be at its plateau, or mid-cycle,
            # by the time a newly-unfrozen resolution starts learning).
            if getattr(self, 'training_strategy', None) == 'progressive':
                scheduler = self.lr_schedulers()
                if isinstance(scheduler, list):
                    scheduler = scheduler[0] if scheduler else None
                if scheduler is not None:
                    scheduler.last_epoch = -1
                    scheduler._step_count = 0
                    scheduler.step()
                    if self.global_rank == 0:
                        print(f"  ↺  LR scheduler restarted for x{train_res} phase.")
        self._prev_res_idx = res_idx

        if self.global_rank == 0:
            print(f"\n[Epoch {epoch}] Training resolution: {train_res}")

        #  LOG: Print learning rate at start of epoch
        if self.optimizers() is not None:
            optimizer = self.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                print(f"  Learning rate (group {i}): {lr:.6e}")
                self.log(f'lr_group_{i}', lr, on_step=False, on_epoch=True)
        
        print(f"{'='*60}")

        for res in self.multires:
            model = self.solver.solvers[f"solver_x{res}"]
            if res == train_res:
                model.train()
                for p in model.parameters():
                    p.requires_grad = True
                if self.trainer.is_global_zero:
                    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"  solver_x{res}: TRAINING mode - {n_trainable:,} trainable params")
            else:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                if self.trainer.is_global_zero:
                    print(f"  solver_x{res}: EVAL mode - gradients frozen")

    def _apply_constraints(self, out_dict, res):
        """
        Apply physical constraints to predictions based on norm_stats.
        Works on NORMALIZED data (as stored in the model).
        
        Args:
            out_dict: Dict of predictions {pred_var_name: tensor (B, T, H, W)}
        
        Returns:
            out_dict: Constrained predictions
        """
        constrained = {}

        res_key = f"patch_x{res}"
        # Get resolution-specific mapping
        if isinstance(self.var_mapping, dict) and res_key in self.var_mapping:
            mapping = self.var_mapping[res_key]
        else:
            # Fallback to base var_mapping (for backward compatibility)
            mapping = self.var_mapping
        
        for pred_var_name, pred in out_dict.items():
            if pred is None:
                constrained[pred_var_name] = None
                continue
            
            # Match pred_sic -> tgt_sic, pred_SIT -> tgt_SIT
            if 'pred' in pred_var_name:
                canonical_var = pred_var_name.replace('pred_', '')
            else:
                canonical_var = pred_var_name.replace('tgt_', '')
            matching_tgt_var = f'tgt_{canonical_var}'
            
            # Check if this target variable exists in var_mapping
            if matching_tgt_var not in mapping:
                # No mapping, no constraint
                constrained[pred_var_name] = pred
                continue
            
            # Get source variable from var_mapping
            # e.g., tgt_sic -> asip_sic
            source_var = mapping[matching_tgt_var]
            
            # Parse source_var (e.g., 'asip_sic' -> group='asip', var='sic')
            if '_' not in source_var:
                constrained[pred_var_name] = pred
                continue
            
            group, var = source_var.split('_', 1)
            
            # Get normalization stats
            if group not in self._norm_stats or var not in self._norm_stats[group]:
                constrained[pred_var_name] = pred
                continue
            
            stats = self._norm_stats[group][var]
            
            # Apply constraints based on normalization type
            if stats["type"] == "minmax":
                # For minmax normalization: (x - min) / (max - min)
                # Normalized data should be in [0, 1]
                min_val, max_val = 0.0, 1.0
                
                # Count violations before clamping
                below_min = (pred < min_val).sum().item()
                above_max = (pred > max_val).sum().item()
                total = pred.numel()
                
                # Clamp to [0, 1] for normalized data
                constrained[pred_var_name] = torch.clamp(pred, min_val, max_val)
            
            elif stats["type"] == "zscore":
                # For zscore normalization: (x - mean) / std
                # Normalized data should typically be in ~[-3, 3], we use ±5σ
                # In normalized space, this means [-5, 5]
                min_val, max_val = -5.0, 5.0
                
                # Count violations
                below_min = (pred < min_val).sum().item()
                above_max = (pred > max_val).sum().item()
                total = pred.numel()
                
                # Clamp to ±5 in normalized space
                constrained[pred_var_name] = torch.clamp(pred, min_val, max_val)
            
            else:
                # Unknown normalization type, no constraint
                constrained[pred_var_name] = pred
        
        return constrained
        
    def multistep(self, batch, phase=""):
        """
        Multi-resolution training with three strategies:
        1. Progressive: train one resolution at a time (curriculum learning)
        2. Simultaneous: train all resolutions together
        3. Hybrid: progressive then simultaneous
        
        Set via self.training_strategy (default: 'simultaneous')
        """
        batch = self.modify_multires_batch(batch)
        out = {}
        
        # STRATEGY SELECTION
        # Add this in __init__: self.training_strategy = "simultaneous"  # or "progressive" or "hybrid"
        strategy = getattr(self, 'training_strategy', 'simultaneous')
        
        if strategy == 'progressive':
            # Original curriculum learning approach
            epoch = self.current_epoch
            n_res = len(self.multires)
            total_epochs = self.trainer.max_epochs
            steps_per_res = max(1, total_epochs // n_res)
            res_index = min(epoch // steps_per_res, n_res - 1)
            train_resolutions = [self.multires[res_index]]
            
        elif strategy == 'hybrid':
            # Progressive at first, then simultaneous
            epoch = self.current_epoch
            n_res = len(self.multires)
            # Train progressively for first 40% of epochs, then all together
            warmup_epochs = int(self.trainer.max_epochs * 0.4)
            
            if epoch < warmup_epochs:
                steps_per_res = max(1, warmup_epochs // n_res)
                res_index = min(epoch // steps_per_res, n_res - 1)
                train_resolutions = [self.multires[res_index]]
            else:
                train_resolutions = self.multires  # All resolutions
                
        else:  # 'simultaneous'
            # Train all resolutions at once
            train_resolutions = self.multires

        # active_resolutions: résolutions pour lesquelles on logue les métriques
        # En progressive/hybrid partiel: seulement la résolution active
        # En simultané: toutes
        active_resolutions = train_resolutions

        total_loss = 0.
        
        for i, res in enumerate(self.multires):
            batch_res = batch[f"patch_x{res}"]
            should_train = (res in train_resolutions) and (phase == "train")
            # Loguer les métriques seulement pour la résolution active
            log_phase = phase if (res in active_resolutions) else ""

            if i == 0:
                # First resolution (coarsest)
                if should_train:
                    loss, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=phase)
                    total_loss += loss
                else:
                    # Inference only (validation/test or frozen resolution)
                    with torch.no_grad():
                        _, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=log_phase)
            else:
                # Finer resolutions
                coarser_res = self.multires[i-1]
                
                # Get coordinates
                xc_target = batch_res.xc
                yc_target = batch_res.yc
                xc_coarse = batch[f"patch_x{coarser_res}"].xc
                yc_coarse = batch[f"patch_x{coarser_res}"].yc
                
                if xc_coarse.ndim == 3:
                    xc_coarse = torch.squeeze(xc_coarse, dim=1)
                    yc_coarse = torch.squeeze(yc_coarse, dim=1)
                if xc_target.ndim == 3:
                    xc_target = torch.squeeze(xc_target, dim=1)
                    yc_target = torch.squeeze(yc_target, dim=1)
                
                # Detach or not based on training strategy
                if strategy == 'simultaneous':
                    # No detach: Gradients flow through all resolutions
                    out_coarse_for_interp = out[f"patch_x{coarser_res}"]
                else:
                    # Detach: Only current resolution is trained
                    out_coarse_for_interp = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v 
                        for k, v in out[f"patch_x{coarser_res}"].items()
                    }
                
                # Interpolate
                out[f"patch_x{coarser_res}_on_x{res}"] = self.interpolate_torch(
                    out_coarse_for_interp,
                    xc_coarse, yc_coarse,
                    xc_target, yc_target
                )
                out[f"patch_x{coarser_res}_on_x{res}"] = self.crop_daw(
                    out[f"patch_x{coarser_res}_on_x{res}"], res
                )
                
                # Update batch as anomaly
                batch_res = self.update_batch_as_anomaly(
                    batch_res, 
                    out[f"patch_x{coarser_res}_on_x{res}"]
                )

                # Instance-normalise the anomaly so the solver always sees std≈1
                if self.normalize_anomaly:
                    batch_res, anom_scale = self.normalize_anomaly_batch(batch_res)
                else:
                    anom_scale = {}

                # Alternative to normalize_anomaly: feed the coarse-field local
                # scale as an extra input channel instead of hard normalising.
                scale_channel = None
                if self.condition_on_scale:
                    scale_channel = self.compute_scale_channel(
                        out[f"patch_x{coarser_res}_on_x{res}"]
                    )

                # Train or inference
                if should_train:
                    loss, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=phase,
                                                            scale_channel=scale_channel)
                    total_loss += loss
                else:
                    with torch.no_grad():
                        _, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=log_phase,
                                                             scale_channel=scale_channel)

                # Denormalise predictions before adding back the coarse resolution
                if anom_scale:
                    out[f"patch_x{res}"] = self.denormalize_anomaly_predictions(
                        out[f"patch_x{res}"], anom_scale
                    )

                # Add coarse resolution back
                # Get resolution-specific target vars
                tgt_vars = self._get_target_vars_for_resolution(res)
                for var_name in tgt_vars:
                    if '_' in var_name:
                        var_suffix = var_name.split('_', 1)[1]
                        pred_var_name = f'pred_{var_suffix}'
                    else:
                        pred_var_name = f'pred_{var_name}'
                    
                    # Use non-inplace operation to avoid view issues
                    out[f"patch_x{res}"][pred_var_name] = (
                        out[f"patch_x{res}"][pred_var_name] + 
                        out[f"patch_x{coarser_res}_on_x{res}"][pred_var_name]
                    )
            
            # Apply constraints
            out[f"patch_x{res}"] = self._apply_constraints(out[f"patch_x{res}"], res)
        
        return total_loss, out

    def total_variation_loss_classic(self, pred, mask_interp, dilation_radius=2):
        """
        Compute Total Variation loss on interpolated pixels and their neighborhood.
        Encourages spatial smoothness, especially at the boundary between 
        interpolated and observed regions where artifacts often appear.
        
        Args:
            pred: (B, T, H, W) prediction
            mask_interp: (B, T, H, W) boolean mask of interpolated pixels
            dilation_radius: number of pixels to extend the mask (default: 2)
        
        Returns:
            tv_loss: scalar tensor
        """
        B, T, H, W = pred.shape
        
        # Dilate mask to include neighborhood around interpolated pixels
        # This captures the transition zone where artifacts are most visible
        if dilation_radius > 0:
            # Create dilation kernel
            kernel_size = 2 * dilation_radius + 1
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=pred.device)
            
            # Reshape mask for conv2d: (B, T, H, W) -> (B*T, 1, H, W)
            mask_flat = mask_interp.float().reshape(B * T, 1, H, W)
            
            # Apply dilation (max pooling with stride 1)
            dilated_mask = F.conv2d(
                mask_flat,
                kernel,
                padding=dilation_radius,
                stride=1
            )
            
            # Threshold to get binary mask (any neighbor was True)
            dilated_mask = (dilated_mask > 0).reshape(B, T, H, W)
            
            # Alternative: use morphological dilation (requires kornia)
            # from kornia.morphology import dilation
            # dilated_mask = dilation(mask_flat, kernel).reshape(B, T, H, W)
        else:
            dilated_mask = mask_interp
        
        # Compute spatial gradients
        diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])  # Vertical: (B, T, H-1, W)
        diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])  # Horizontal: (B, T, H, W-1)
        
        # Expand dilated mask to match gradient dimensions
        # For vertical gradients: combine adjacent rows
        mask_h = dilated_mask[:, :, 1:, :] | dilated_mask[:, :, :-1, :]  # (B, T, H-1, W)
        
        # For horizontal gradients: combine adjacent columns
        mask_w = dilated_mask[:, :, :, 1:] | dilated_mask[:, :, :, :-1]  # (B, T, H, W-1)
        
        # Compute TV loss only on masked regions
        n_valid_h = mask_h.sum()
        n_valid_w = mask_w.sum()
        
        if n_valid_h > 0:
            tv_h = (diff_h[mask_h]).mean()
        else:
            tv_h = torch.tensor(0.0, device=pred.device)
        
        if n_valid_w > 0:
            tv_w = (diff_w[mask_w]).mean()
        else:
            tv_w = torch.tensor(0.0, device=pred.device)
        
        tv_loss = tv_h + tv_w
        
        return tv_loss
    
    def total_variation_loss(self, pred, mask_interp, mask_obs, dilation_radius=2):
        """
        Compute Total Variation loss at the boundary between interpolated and observed pixels.
        Penalizes abrupt spatial gradients at the transition zone.
        
        Args:
            pred: (B, T, H, W) prediction
            mask_interp: (B, T, H, W) boolean mask of interpolated pixels
            mask_obs: (B, T, H, W) boolean mask of observed pixels
            dilation_radius: number of pixels to extend each mask (default: 2)
        
        Returns:
            tv_loss: scalar tensor
        """
        B, T, H, W = pred.shape
        
        #  1. Create boundary mask: intersection of dilated interpolation zone and dilated observation zone
        if dilation_radius > 0:
            kernel_size = 2 * dilation_radius + 1
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=pred.device)
            
            # Reshape masks for conv2d: (B, T, H, W) -> (B*T, 1, H, W)
            mask_interp_flat = mask_interp.float().reshape(B * T, 1, H, W)
            mask_obs_flat = mask_obs.float().reshape(B * T, 1, H, W)
            
            # Dilate both masks
            dilated_interp = F.conv2d(mask_interp_flat, kernel, padding=dilation_radius, stride=1)
            dilated_obs = F.conv2d(mask_obs_flat, kernel, padding=dilation_radius, stride=1)
            
            # Threshold to get binary masks
            dilated_interp = (dilated_interp > 0).reshape(B, T, H, W)
            dilated_obs = (dilated_obs > 0).reshape(B, T, H, W)
            
            # Boundary mask = intersection of both dilated zones
            boundary_mask = dilated_interp & dilated_obs
        else:
            # No dilation: direct intersection (rare case)
            boundary_mask = mask_interp & mask_obs
        
        # 2. Compute spatial gradients of prediction
        # Vertical gradient (using Sobel-like operator for robustness)
        grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # (B, T, H-1, W)
        # Horizontal gradient
        grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # (B, T, H, W-1)
        
        # 3. Compute second derivatives (measure of gradient variation)
        # Second derivative in y: d²f/dy² ≈ grad_y[i+1] - grad_y[i]
        grad_yy = torch.abs(grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :])  # (B, T, H-2, W)
        
        # Second derivative in x: d²f/dx²
        grad_xx = torch.abs(grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1])  # (B, T, H, W-2)
        
        # 4. Apply boundary mask to second derivatives
        # Mask must match dimensions of second derivatives
        # For grad_yy: need mask at (B, T, H-2, W)
        mask_yy = boundary_mask[:, :, 1:-1, :]  # Remove first and last rows
        
        # For grad_xx: need mask at (B, T, H, W-2)
        mask_xx = boundary_mask[:, :, :, 1:-1]  # Remove first and last columns
        
        #  PLOT TV LOSS COMPUTATION ZONES (every 100 steps)
        #if self.global_step % 100 == 0 and self.trainer.is_global_zero:
        #self.plot_tv_loss_zones(
        #        pred, mask_interp, mask_obs, 
        #        dilated_interp, dilated_obs, boundary_mask,
        #        mask_yy, mask_xx, grad_yy, grad_xx
        #    ) 
        # 5. Compute TV loss only on boundary
        n_valid_yy = mask_yy.sum()
        n_valid_xx = mask_xx.sum()
        
        if n_valid_yy > 0:
            tv_yy = (grad_yy[mask_yy]).mean()
        else:
            tv_yy = torch.tensor(0.0, device=pred.device)
        
        if n_valid_xx > 0:
            tv_xx = (grad_xx[mask_xx]).mean()
        else:
            tv_xx = torch.tensor(0.0, device=pred.device)
        
        tv_loss = tv_yy + tv_xx
        
        return tv_loss

    def spatial_context_loss(self, pred, target, input_obs, mask_interp, radius=3, weight=None):
        """
        For each interpolated pixel, ensure consistency with nearby observed pixels.
        
        Args:
            pred: (B, T, H, W) prediction
            target: (B, T, H, W) ground truth
            input_obs: (B, T, H, W) input observations (with NaN)
            mask_interp: (B, T, H, W) mask of interpolated pixels
            radius: neighborhood radius
            weight: (H, W) optional weight tensor for weighted MSE
        
        Returns:
            context_loss: scalar tensor
        """
        B, T, H, W = pred.shape
        
        # Ensure all tensors are float32
        pred = pred.float()
        target = target.float()
        input_obs = input_obs.float()
        
        # Create kernel for averaging neighborhood
        kernel = torch.ones(1, 1, 2*radius+1, 2*radius+1, device=pred.device, dtype=torch.float32) / ((2*radius+1)**2)
        
        # Reshape for conv2d
        pred_flat = pred.reshape(B*T, 1, H, W)
        input_flat = input_obs.reshape(B*T, 1, H, W)
        
        # Compute local averages of observations (ignoring NaN)
        input_valid = input_flat.nan_to_num(0.0)
        input_mask = input_flat.isfinite().float()
        
        # Weighted average of valid observations in neighborhood
        local_avg = F.conv2d(input_valid, kernel, padding=radius)
        local_count = F.conv2d(input_mask, kernel, padding=radius)
        local_avg = local_avg / (local_count + 1e-6)
        
        # Reshape back
        local_avg = local_avg.reshape(B, T, H, W)
        
        # Compute difference
        diff = pred - local_avg
        
        # Apply mask: only compute loss on interpolated pixels
        diff_masked = torch.where(mask_interp, diff, torch.tensor(float('nan'), device=pred.device))
        
        # Compute MSE with optional weighting
        valid_diff = diff_masked[mask_interp]
        if valid_diff.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        if weight is not None:
            context_loss = self.weighted_mse(diff_masked, weight)
        else:
            context_loss = (valid_diff ** 2).mean()

        return context_loss

    def compute_balanced_weights(self, loss_values):
        """
        Compute weights to balance losses according to target ratios.
        
        Args:
            loss_values: dict of {loss_name: scalar_value}
        
        Returns:
            weights: dict of {loss_name: weight}
        """
        # Convert to tensor
        losses = torch.stack([
            loss_values['base'],
            loss_values['grad'],
            loss_values['prior'],
            loss_values['tv'],
            loss_values['context']
        ])
        
        # Update EMA
        if self.training:
            if self.loss_ema.sum() == 0:  # First batch
                self.loss_ema = losses.detach()
            else:
                self.loss_ema = (1 - self.ema_alpha) * self.loss_ema + self.ema_alpha * losses.detach()
        
        #  Use the same EMA for both train and val (computed during training)
        # Compute target magnitudes based on ratios
        total_ema = self.loss_ema.sum()
        if total_ema == 0:  # Safety check (should not happen after first batch)
            total_ema = losses.sum().detach()
        
        target_magnitudes = torch.tensor([
            self.loss_target_ratios['base'],
            self.loss_target_ratios['grad'],
            self.loss_target_ratios['prior'],
            self.loss_target_ratios['tv'],
            self.loss_target_ratios['context']
        ], device=losses.device) * total_ema
        
        # Compute weights: target / current (with clipping to avoid instability)
        weights = target_magnitudes / (self.loss_ema + 1e-8)
        #weights = torch.clamp(weights, 0.00001, 10.0)  # Prevent extreme weights
        
        result = {
            'base': weights[0].item(),
            'grad': weights[1].item(),
            'prior': weights[2].item(),
            'tv': weights[3].item(),
            'context': weights[4].item()
        }
        # Persist for reuse during validation
        self._last_balanced_weights = result
        return result

    def step(self, batch, res, phase="", scale_channel=None):


        loss, out = self.base_step(batch, res=res, phase=phase, scale_channel=scale_channel)
        res_key = f"patch_x{res}"
        # Get resolution-specific mapping
        if isinstance(self.var_mapping, dict) and res_key in self.var_mapping:
            mapping = self.var_mapping[res_key]
        else:
            # Fallback to base var_mapping (for backward compatibility)
            mapping = self.var_mapping

        # Get resolution-specific target vars
        tgt_vars = self._get_target_vars_for_resolution(res)

        # Get source_vars for global_input_valid computation
        batch_dict = batch._asdict()
        global_input_valid = torch.ones_like(batch_dict[tgt_vars[0]], dtype=torch.bool)
        source_vars = list(mapping.values())
        
        for var in self.input_vars:
            if var in batch_dict and var not in source_vars:
                global_input_valid &= batch_dict[var].isfinite()
        
        for cov in self.covariates:
            if cov in batch_dict:
                global_input_valid &= batch_dict[cov].isfinite()

        total_grad_loss = 0.0
        total_prior_loss = 0.0
        total_tv_loss = 0.0
        total_context_loss = 0.0

        for var_name in tgt_vars:
            if not hasattr(batch, var_name):
                raise ValueError(f"Batch missing variable: {var_name}")
    
            target = getattr(batch, var_name)
            var_suffix = var_name.split('_', 1)[1]  # Get everything after first '_'
            pred_var_name = f'pred_{var_suffix}'
            pred = out[pred_var_name]
    
            # Create masks
            source_var = batch_dict[mapping[var_name]]
            mask_interp = (~source_var.isfinite()) & target.isfinite() #& global_input_valid
            mask_obs = source_var.isfinite() & target.isfinite() #& global_input_valid

            _zero = torch.tensor(0.0, device=pred.device, requires_grad=True)

            # Masks
            mask_grad = target.isfinite() #& global_input_valid

            # 1. Gradient loss — skip if no valid target pixels
            if mask_grad.any():
                tgt_sobel = kfilts.sobel(target)
                pred_sobel = kfilts.sobel(pred)
                grad_diff = pred_sobel - tgt_sobel
                grad_diff_masked = torch.where(
                    mask_grad,
                    grad_diff,
                    torch.tensor(float('nan'), device=pred.device)
                )
                grad_loss = self.weighted_mse(grad_diff_masked, self.optim_weight[res_key])
            else:
                grad_loss = _zero
            total_grad_loss += grad_loss

            #  2. Total Variation on interpolated regions — skip if no interp pixels
            mask_target = batch._asdict()[var_name].isfinite()
            if mask_interp.any():
                tv_loss = self.total_variation_loss(pred, mask_target, ~mask_target, dilation_radius=1)
            else:
                tv_loss = _zero
            total_tv_loss += tv_loss

            #  3. Spatial context with observations — skip if no interp pixels
            input_obs = batch._asdict()[mapping[var_name]]
            if mask_interp.any():
                context_loss = self.spatial_context_loss(
                    pred, target, input_obs, mask_interp,
                    radius=3,
                    weight=self.optim_weight[res_key]
                )
            else:
                context_loss = _zero
            total_context_loss += context_loss
    
        # 4. Prior / SRNN loss
        if hasattr(self.solver.solvers[f"solver_x{res}"], "prior_cost"):
            sbatch = self.format_batch_for_solver(batch, include_masks=self.include_masks, res=res,
                                                   scale_channel=scale_channel)
            model = self.solver.solvers[f"solver_x{res}"].to(device)
            prior = model.prior_cost.forward_ae(sbatch.input.nan_to_num())
            prior_diff = sbatch.tgt - prior
            prior_weight = self.prior_weight[res_key].to(device)
            prior_valid = prior_diff.isfinite() & ((torch.ones_like(prior_diff) * prior_weight[None, ...]) != 0.0)
            if prior_valid.any():
                total_prior_loss = self.weighted_mse(prior_diff, prior_weight)
                # Add small L2 regularization to ensure all params are used
                l2_reg = sum(p.pow(2.0).sum() for p in model.prior_cost.parameters())
                total_prior_loss = total_prior_loss + 1e-6 * l2_reg  # Tiny regularization
            else:
                total_prior_loss = _zero
        else:
            total_prior_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # COMPUTE BALANCED WEIGHTS
        loss_values = {
            'base': loss,
            'grad': total_grad_loss,
            'prior': total_prior_loss,
            'tv': total_tv_loss,
            'context': total_context_loss
        }
        
        if self.training:
            weights = self.compute_balanced_weights(loss_values)
        else:
            # Reuse the last balanced weights from training (fall back to uniform if not yet available)
            weights = getattr(self, '_last_balanced_weights',
                              {'base': 1.0, 'grad': 1.0, 'prior': 1.0, 'tv': 1.0, 'context': 1.0})
        
        # COMBINED LOSS with auto-balanced weights
        training_loss = (
            weights['base'] * loss +
            weights['grad'] * total_grad_loss +
            weights['prior'] * total_prior_loss +
            weights['tv'] * total_tv_loss +
            weights['context'] * total_context_loss
        )

        # ── Auxiliary losses summary (every batch, rank 0) ─────────────────
        if self.trainer.is_global_zero:
            def _fmt(v, w):
                val = v.item() if hasattr(v, 'item') else float(v)
                contrib = w * val
                flag = " ❌ NaN"  if not np.isfinite(val)  else \
                       " ⚠️ >10"   if val > 10.             else \
                       " ⚠️ >1"    if val > 1.              else ""
                return f"{val:.6f} × {w:.3f} = {contrib:.6f}{flag}"

            print(f"  ── Auxiliary losses (x{res}) ──")
            print(f"    base    : {_fmt(loss,                weights['base'])}")
            print(f"    grad    : {_fmt(total_grad_loss,     weights['grad'])}")
            print(f"    prior   : {_fmt(total_prior_loss,    weights['prior'])}")
            print(f"    tv      : {_fmt(total_tv_loss,       weights['tv'])}")
            print(f"    context : {_fmt(total_context_loss,  weights['context'])}")
            tl = training_loss.item() if hasattr(training_loss, 'item') else float(training_loss)
            flag_total = " ❌ NaN" if not np.isfinite(tl) else " ⚠️ >10" if tl > 10. else ""
            print(f"    TOTAL   : {tl:.6f}{flag_total}")

        # Log individual losses AND weights (seulement si phase est défini)
        if phase:
            self.log(f"{phase}_loss",         loss,               prog_bar=True,  on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{phase}_gloss",        total_grad_loss,    prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{phase}_prior_loss",   total_prior_loss,   prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{phase}_tv_loss",      total_tv_loss,      prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{phase}_context_loss", total_context_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{phase}_total_loss",   training_loss,      prog_bar=True,  on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{phase}_weight_base",  weights['base'],    prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{phase}_weight_grad",  weights['grad'],    prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{phase}_weight_tv",    weights['tv'],      prog_bar=False, on_step=False, on_epoch=True)

        return training_loss, out

    def base_step(self, batch, res, phase="", scale_channel=None):
        """
        Compute loss over selected target variables in a multi-variate model.
        Args:
            batch: a NamedTuple with target fields matching tgt_vars.
            phase: string for logging ("train", "val", etc.)
            scale_channel: optional (B, T, H, W) tensor, see ``compute_scale_channel``.
        Returns:
           loss: total loss
           out: model output tensor
        """
        res_key = f"patch_x{res}"
        # Get resolution-specific mapping
        if isinstance(self.var_mapping, dict) and res_key in self.var_mapping:
            mapping = self.var_mapping[res_key]
        else:
            # Fallback to base var_mapping (for backward compatibility)
            mapping = self.var_mapping

        sbatch = self.format_batch_for_solver(batch, include_masks=self.include_masks, res=res,
                                               scale_channel=scale_channel)
        self.plot_counter += 1  
        # === PLOT DEBUG (every N batches) ===
        if (self.plot_counter % 100 == 0) and (res==50):  # Plot every 10 batches
            try:
                self.plot_batch_debug(sbatch, res, phase, batch_idx=self.plot_counter)
                self.plot_input_target_mapping_debug(batch, res, phase, batch_idx=self.plot_counter)
            except Exception as e:
                print(f"Warning: Failed to create debug plot: {e}")

        out = self(batch=sbatch, res=res)  # out is a tensor 
        out = self.split_tensor_to_dict(out, res=res)
        res_key = f"patch_x{res}"

        # Get resolution-specific target vars
        tgt_vars = self._get_target_vars_for_resolution(res)

        # Create global mask for all input variables
        batch_dict = batch._asdict()
        # Start with all True (all pixels valid initially)
        global_input_valid = torch.ones_like(batch_dict[tgt_vars[0]], dtype=torch.bool)
        # Get list of source variables from var_mapping
        source_vars = list(mapping.values())  
        # For each input variable, check if finite
        for var in self.input_vars:
            if var in batch_dict and var not in source_vars:
                global_input_valid &= batch_dict[var].isfinite()
        # Also check covariates
        for cov in self.covariates:
            if cov in batch_dict:
                global_input_valid &= batch_dict[cov].isfinite()

        total_loss = 0.0
        do_print = self.trainer.is_global_zero and (self.global_step % 50 == 0)

        if do_print:
            print(f"\n{'='*70}")
            print(f"[Step {self.global_step:05d}] {phase.upper():5s} | res=x{res}")
            print(f"{'='*70}")

        for i, var_name in enumerate(tgt_vars):
            if not hasattr(batch, var_name):
                raise ValueError(f"Batch does not contain variable '{var_name}'")
            target = getattr(batch, var_name)  # (B, T, Y, X)
            # Convert var_name to pred_var_name
            if '_' in var_name:
                var_suffix = var_name.split('_', 1)[1]
                pred_var_name = f'pred_{var_suffix}'
            else:
                pred_var_name = f'pred_{var_name}'
            pred = out[pred_var_name]  # (B, T, Y, X)

            # Mask 1: Interpolation pixels (input NaN, target valid)
            mask  = ~batch._asdict()[mapping[var_name]].isfinite() & target.isfinite() #& global_input_valid
            # Mask 2: Observation pixels (both input and target valid)
            mask2 =  batch._asdict()[mapping[var_name]].isfinite() & target.isfinite() #& global_input_valid

            n_mask  = mask.sum().item()
            n_mask2 = mask2.sum().item()
            n_total = target.numel()
            pct_mask  = 100.0 * n_mask  / n_total if n_total > 0 else 0.0
            pct_mask2 = 100.0 * n_mask2 / n_total if n_total > 0 else 0.0

            # Compute losses — return 0 when the mask is empty (e.g. full satellite coverage)
            _zero = torch.tensor(0.0, device=pred.device, requires_grad=True)
            if n_mask > 0:
                loss = self.weighted_mse(
                    torch.where(mask,  pred, torch.tensor(float('nan'), device=pred.device)) - target,
                    self.optim_weight[res_key]
                )
            else:
                loss = _zero

            if n_mask2 > 0:
                loss2 = self.weighted_mse(
                    torch.where(mask2, pred, torch.tensor(float('nan'), device=pred.device)) - target,
                    self.optim_weight[res_key]
                )
            else:
                loss2 = _zero

            # ── per-variable diagnostics ──────────────────────────────────
            if do_print:
                target_valid = target[target.isfinite()]
                pred_finite   = pred[pred.isfinite()]

                t_min  = target_valid.min().item() if target_valid.numel() > 0 else float('nan')
                t_max  = target_valid.max().item() if target_valid.numel() > 0 else float('nan')
                t_mean = target_valid.mean().item() if target_valid.numel() > 0 else float('nan')
                p_min  = pred_finite.min().item()   if pred_finite.numel()  > 0 else float('nan')
                p_max  = pred_finite.max().item()   if pred_finite.numel()  > 0 else float('nan')
                p_mean = pred_finite.mean().item()  if pred_finite.numel()  > 0 else float('nan')
                p_nan_pct = 100.0 * (~pred.isfinite()).sum().item() / pred.numel()

                flag_mask  = " ❌ NO INTERP PIXELS"  if n_mask  == 0 else \
                             " ⚠️  <1% interp"        if pct_mask  < 1.0 else ""
                flag_mask2 = " ❌ NO OBS PIXELS"     if n_mask2 == 0 else \
                             " ⚠️  <1% obs"           if pct_mask2 < 1.0 else ""
                flag_loss  = " ❌ NaN loss"           if not loss.isfinite()  else \
                             " ⚠️  loss>10"            if loss.item()  > 10.  else \
                             " ⚠️  loss>1"             if loss.item()  > 1.   else ""
                flag_loss2 = " ❌ NaN loss"           if not loss2.isfinite() else \
                             " ⚠️  loss>10"            if loss2.item() > 10.  else \
                             " ⚠️  loss>1"             if loss2.item() > 1.   else ""
                flag_pred  = " ❌ NaN pred"           if p_nan_pct > 50.     else \
                             " ⚠️  NaN in pred"        if p_nan_pct > 0.      else ""
                flag_range = " ⚠️  pred out of target range" \
                             if pred_finite.numel() > 0 and target_valid.numel() > 0 and \
                                (p_min < t_min - abs(t_min) or p_max > t_max + abs(t_max)) else ""

                print(f"  ── {var_name} ──")
                print(f"    Target : [{t_min:+.4f}, {t_max:+.4f}]  mean={t_mean:+.4f}  "
                      f"valid={100.*(target_valid.numel()/n_total):.1f}%")
                print(f"    Pred   : [{p_min:+.4f}, {p_max:+.4f}]  mean={p_mean:+.4f}  "
                      f"NaN={p_nan_pct:.1f}%{flag_pred}{flag_range}")
                print(f"    Interp : {n_mask:7d}/{n_total:7d} ({pct_mask:5.2f}%){flag_mask}")
                print(f"    Obs    : {n_mask2:7d}/{n_total:7d} ({pct_mask2:5.2f}%){flag_mask2}")
                print(f"    Loss_interp = {loss.item():.6f}{flag_loss}")
                print(f"    Loss_obs    = {loss2.item():.6f}{flag_loss2}")

            # Log to tensorboard/wandb (seulement si phase est défini)
            if phase:
                self.log(f"{phase}_loss_interp_{var_name}", loss,  on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"{phase}_loss_obs_{var_name}",    loss2, on_step=True, on_epoch=True, sync_dist=True)
                self.log(f"{phase}_mask_pct_interp_{var_name}", pct_mask,  on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{phase}_mask_pct_obs_{var_name}",    pct_mask2, on_step=False, on_epoch=True, sync_dist=True)
            total_loss += loss + loss2

        if do_print:
            print(f"  ── TOTAL base_loss = {total_loss.item():.6f} {'❌ NaN' if not total_loss.isfinite() else ''}")
            print(f"{'='*70}")

        return total_loss, out

    def reconstruct(self, dl, items, daw, time, weight=None, patch_coords=None):
        """
        takes as input a list of tensor of dimensions (V, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims
        items: list of torch tensor corresponding to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        patch_coords: optional list of (yc_1d, xc_1d) numpy arrays, one per item.
                      When provided the items are placed by their actual spatial
                      coordinates instead of by their sequential index in the
                      dataset – required for correct multi-GPU reconstruction.
        overlapping patches will be averaged with weighting 
        """

        if weight is None:
            weight = np.ones(list(dl.dataset.patch_dims.values()))
        weight = torch.tensor(weight)

        nvars = items[0].shape[0]

        result_tensor = torch.full((nvars, 1, dl.dataset.da_dims['yc'], dl.dataset.da_dims['xc']),
                                   float('nan'))
        count_tensor = torch.zeros((nvars, 1, dl.dataset.da_dims['yc'], dl.dataset.da_dims['xc']))

        if patch_coords is not None:
            # Multi-GPU path: use explicit coordinates stored alongside each prediction
            coord_iter = patch_coords
        else:
            # Single-GPU legacy path: derive coordinates from the dataset by index
            ds_coords = dl.dataset.get_coords()[(daw * len(items)):((daw + 1) * len(items))]
            coord_iter = [(c.yc.values, c.xc.values) for c in ds_coords]

        for idx, item in enumerate(items):
            yc_patch, xc_patch = coord_iter[idx]
            iy = [np.where(dl.dataset.yc == y)[0][0] for y in yc_patch]
            ix = [np.where(dl.dataset.xc == x)[0][0] for x in xc_patch]
            result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] = torch.where(torch.isnan(result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]),
                                                                              0.,
                                                                              result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1])
            result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += torch.squeeze(item * weight)
            count_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += weight

        result_tensor /= np.maximum(count_tensor, 1e-6)
        result_da = xr.DataArray(
            result_tensor,
            #dims=[f'v{i}' for i in range(nvars - len(coords[0].dims))] + ["time", "yc", "xc"],
            dims = ["v0", "time", "yc", "xc"],
            coords={
                "time": [time],
                "xc": dl.dataset.xc,
                "yc": dl.dataset.yc,
                "lon": (["yc","xc"],dl.dataset.lon),
                "lat": (["yc","xc"],dl.dataset.lat)
            }
        )
        return result_da

    def aggregate_batches_one_domain(self, idx_daw, idx_rec,
                                     test_data, 
                                     dataloader_idx=None,
                                     use_datamodule=False,
                                     patch_coords=None):

        dl = self.trainer.test_dataloaders[self.dataloader_keys[dataloader_idx]]
        
        res = self.multires[dataloader_idx]
        last = self.len_daw[res]
        res_key = f"patch_x{res}"
        
        # Get resolution-specific target vars and build variable names
        tgt_vars = self._get_target_vars_for_resolution(res)
        pred_vars = []
        for var in tgt_vars:
            if "_" in var:
                suffix = var.split("_", 1)[-1]
                pred_vars.append(f"pred_{suffix}")
            else:
                pred_vars.append(f"pred_{var}")
        var_names = pred_vars + tgt_vars + self._get_obs_var_names(dataloader_idx)

        netcdf_final = []
                                       
        for i in idx_rec:
            time = dl.dataset.times[-last:][idx_daw+i]
            print("Reconstructing LEADTIME "+str(i))
            if isinstance(dl,list):
                dl = dl[0]
            nbatch = len(test_data)
            if use_datamodule:
                rec_da = dl.dataset.reconstruct(
                            [ test_data[j][:,[i],:,:].cpu() for j in range(nbatch) ],
                            idx_daw, time,
                            self.rec_weight[res_key].cpu().numpy()[[i],:,:]
                    )
            else:
                rec_da = self.reconstruct(dl,
                            [ test_data[j][:,[i],:,:].cpu() for j in range(nbatch) ],
                            idx_daw, time,
                            self.rec_weight[res_key].cpu().numpy()[[i],:,:],
                            patch_coords=patch_coords
                    )
            
            # Instead of using generic v0 dimension, create Dataset directly with variable names
            # rec_da has shape (nvars, time, yc, xc) where nvars = len(var_names)
            data_vars = {}
            for idx, var_name in enumerate(var_names):
                data_vars[var_name] = (("time", "yc", "xc"), rec_da.data[idx])
            
            test_data_ldt = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": rec_da.coords["time"],
                    "yc": rec_da.coords["yc"],
                    "xc": rec_da.coords["xc"],
                    "lon": rec_da.coords["lon"],
                    "lat": rec_da.coords["lat"]
                }
            )
            
            # crop (if necessary) 
            test_data_ldt = test_data_ldt.sel(**(self.domain_limits or {}))
            # stack each time 
            netcdf_final.append(test_data_ldt)

        # merge all time steps for final NetCDFs
        return xr.concat(netcdf_final, dim="time").sortby("time")

    def aggregate_batches(self, idx_rec,
                        test_data, test_times,
                        dataloader_idx=None,
                        metrics=False,
                        write_netcdf=False,
                        use_datamodule=False,
                        patch_coords=None,
                        member=None):

        res = self.multires[dataloader_idx]
        res_key = f"patch_x{res}"

        # Get resolution-specific mapping
        if isinstance(self.var_mapping, dict) and res_key in self.var_mapping:
            mapping = self.var_mapping[res_key]
        else:
            # Fallback to base var_mapping (for backward compatibility)
            mapping = self.var_mapping

        # Get resolution-specific target vars
        tgt_vars = self._get_target_vars_for_resolution(res)

        # On convertit chaque time en tuple Python (hashable)
        time_groups = [tuple(d.cpu().tolist()) for d in test_times]
        # On mappe chaque tuple vers un identifiant unique
        unique_times = {}
        daws = []
        for t in time_groups:
            if t not in unique_times:
                unique_times[t] = len(unique_times)  # new ID
            daws.append(unique_times[t])
        daws = torch.tensor(daws)

        netcdf_final = []
        
        def unnormalize(varname, data):
            """
            Unnormalize using stats from the target variable.
            For pred_SIC and models_SIC, use the same stats (from models_SIC).
            
            Args:
                varname: variable name (e.g., 'pred_SIC', 'models_SIC', 'tgt_sic')
                data: normalized data tensor
            """
            # Find the corresponding target variable
            # If it's a pred_* variable, find its target in tgt_vars
            if varname.startswith('pred_'):
                suffix = varname.split('_', 1)[1]  # 'pred_SIC' -> 'SIC'
                # Find matching target variable with this suffix
                target_var = None
                for tvar in tgt_vars:
                    if tvar.endswith('_' + suffix) or tvar == suffix:
                        target_var = tvar
                        break
                if target_var is None:
                    raise ValueError(f"No target variable found for prediction '{varname}'")
            else:
                # It's already a target variable
                target_var = varname
            
            # Get stats from target variable
            if target_var.startswith('models_') and hasattr(self, 'norm_stats_models'):
                # models_SIT -> use norm_stats_models['SIT']
                var_suffix = target_var.split('_', 1)[1]
                stats = self.norm_stats_models[var_suffix]
            elif target_var.startswith('tgt_') or target_var in mapping:
                # tgt_XXX or other mapped variable -> use source stats from var_mapping
                source_var = mapping.get(target_var)
                if source_var is None:
                    raise ValueError(f"No mapping found for target variable '{target_var}'")
                
                if '_' in source_var:
                    group, var = source_var.split('_', 1)
                else:
                    raise ValueError(f"Invalid source variable format: '{source_var}'")
                
                stats = self.norm_stats[group][var]
            else:
                raise ValueError(f"Cannot determine stats for target variable '{target_var}'")
            
            # Apply denormalization
            if stats["type"] == "zscore":
                return data * stats["std"] + stats["mean"]
            elif stats["type"] == "minmax":
                return data * (stats["max"] - stats["min"]) + stats["min"]
            else:
                raise ValueError(f"Unknown normalization type for {target_var}")

        for idx_daw in torch.unique(daws):
            sel_daw = torch.where(daws==idx_daw)[0]
            test_data_sel = [test_data[i] for i in sel_daw.tolist()]
            coords_sel = [patch_coords[i] for i in sel_daw.tolist()] if patch_coords is not None else None
            test_data_uniq = self.aggregate_batches_one_domain(idx_daw, idx_rec,
                                                            test_data_sel,
                                                            dataloader_idx,
                                                            use_datamodule,
                                                            patch_coords=coords_sel)
            # Prepare unnormalization for metrics and storage
            test_data_unnorm = test_data_uniq.copy(deep=False)
            
            for var in tgt_vars:
                # Get the variable suffix (e.g., 'tgt_sic' -> 'sic')
                if '_' in var:
                    var_suffix = var.split('_', 1)[1]
                else:
                    var_suffix = var
                # Unnormalize using var_mapping
                test_data_unnorm = test_data_unnorm.update({
                    f"pred_{var_suffix}": (("time", "yc", "xc"),
                                        unnormalize(var, test_data_uniq[f"pred_{var_suffix}"].data))
                })
                test_data_unnorm = test_data_unnorm.update({
                    var: (("time", "yc", "xc"),
                                        unnormalize(var, test_data_uniq[var].data))
                })

            for field in self._get_obs_var_names(dataloader_idx):
                source, obs_var = field.split('_', 1)
                stats = self.norm_stats[source][obs_var]
                data = test_data_uniq[field].data
                if stats["type"] == "zscore":
                    data = data * stats["std"] + stats["mean"]
                elif stats["type"] == "minmax":
                    data = data * (stats["max"] - stats["min"]) + stats["min"]
                test_data_unnorm = test_data_unnorm.update({field: (("time", "yc", "xc"), data)})

            if metrics:
                metric_data = test_data_unnorm.pipe(self.pre_metric_fn),
                metrics = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                print(metrics.to_frame(name="Metrics").to_markdown())
            # save NetCDFs
            time = [ datetime.datetime.strptime(str(t)[:10], "%Y-%m-%d").strftime("%Y%m%d") for t in test_data_unnorm.time.data ]
            member_suffix = f'_member{member}' if member is not None else ''
            file = f'test_data_{time[0]}_{time[-1]}_patch_x{res}{member_suffix}.nc'
            if self.logger and write_netcdf:
                 test_data_unnorm.attrs['croscim_model_class'] = type(self).__name__
                 test_data_unnorm.to_netcdf(Path(self.logger.log_dir) / file)
                 print(Path(self.trainer.log_dir) / file)
                 if metrics:
                     self.logger.log_metrics(metrics.to_dict())
            # stack each daw
            netcdf_final.append(test_data_uniq)

        # merge all time steps in a dictionary
        return { f"daw_{i}": nc for i, nc in enumerate(netcdf_final) }
        #return xr.concat(netcdf_final, dim="daw").assign_coords(daw=torch.unique(daws)).sortby("daw")

    def convert_xr_to_batch(self, coarse, batch, spatial_sel=False, res=None):
        """
        Convert an xarray.Dataset (coarse) to a dictionary of PyTorch tensors,
        matching the batch's temporal indices.
        Args:
            coarse (xr.Dataset): xarray with dims (time, yc, xc)
            batch (dict): Dictionary with keys 'time', 'yc', 'xc' etc., 
                          values are tensors of shape (B, T, H, W)
            spatial_sel (bool): Whether to perform spatial selection
            res: resolution key (e.g., 50 for patch_x50) to get resolution-specific target vars
        Returns:
            coarse_dict: dict with same keys as coarse.data_vars, each of shape (B, T, H, W)
        """
        # Get resolution-specific target vars
        if res is not None:
            tgt_vars = self._get_target_vars_for_resolution(res)
        else:
            tgt_vars = self.tgt_vars
        
        times = batch.time.cpu().numpy().astype('datetime64[s]').astype('datetime64[ns]')
        times = times.astype('datetime64[D]').astype('datetime64[ns]')
        nbatch = len(batch.time)  # batch.time: shape (B, T)
        T, H, W = batch.time.shape[1], batch.yc.shape[1], batch.xc.shape[1]
        coarse_dict = {}

        # Pour chaque variable du Dataset
        # Include both tgt_vars and their pred_* equivalents
        pred_vars = []
        for var in tgt_vars:
            if '_' in var:
                var_suffix = var.split('_', 1)[1]
                pred_vars.append(f'pred_{var_suffix}')
            else:
                pred_vars.append(f'pred_{var}')  
        for var in tgt_vars + pred_vars + ["time", "yc", "xc"]:
            B_array = []
            for i in range(nbatch):
                times_i = np.squeeze(times[i])  # (T,)
                xcs_i = batch.xc[i].cpu().numpy()      # (W,)
                ycs_i = batch.yc[i].cpu().numpy()      # (H,)
                
                # temporal selection
                matching_keys = [
                    key
                    for key, ds in coarse.items()
                    if set(ds.time.values) == set(times_i)
                ]
                
                if not matching_keys:
                    print(f"ERROR: No matching time found in coarse dataset!")
                    print(f"  Looking for times: {times_i}")
                    print(f"  Available keys in coarse: {list(coarse.keys())}")
                    if coarse:
                        first_key = list(coarse.keys())[0]
                        print(f"  Example times in coarse['{first_key}']: {coarse[first_key].time.values}")
                    raise ValueError(f"No dataset in coarse matches times {times_i}")
                
                matching_key = matching_keys[0]
                sel_time = coarse[matching_key]
                # spatial selection
                if spatial_sel:
                    xc_is_descending = sel_time.xc[0] > sel_time.xc[-1]
                    yc_is_descending = sel_time.yc[0] > sel_time.yc[-1]
                    xc_start, xc_end = sorted([xcs_i.min(), xcs_i.max()],
                                          reverse=xc_is_descending)
                    yc_start, yc_end = sorted([ycs_i.min(), ycs_i.max()],
                                          reverse=yc_is_descending)
                    sel_patch = sel_time.sel(
                                  xc=slice(xc_start, xc_end),
                                  yc=slice(yc_start, yc_end)
                                        )
                else:
                    sel_patch = sel_time
                
                # Find variable in dataset by suffix matching
                # If looking for 'tgt_SIC', also accept 'models_SIC', 'pred_SIC', etc.
                var_name_in_ds = None
                if var in ["time", "yc", "xc", "lon", "lat"]:
                    # Coordinates - use exact name
                    var_name_in_ds = var if var in sel_patch else None
                elif var in sel_patch:
                    # Exact match
                    var_name_in_ds = var
                else:
                    # Try suffix matching (e.g., tgt_SIC matches models_SIC)
                    if '_' in var:
                        suffix = var.split('_', 1)[1]  # Extract SIC from tgt_SIC
                        for ds_var in sel_patch.data_vars:
                            if '_' in ds_var and ds_var.split('_', 1)[1] == suffix:
                                var_name_in_ds = ds_var
                                break
                
                if var_name_in_ds is None:
                    # Variable not found - skip it (will be None in output)
                    continue
                
                # Convertir en numpy
                arr = sel_patch[var_name_in_ds].values  # (T, H, W)
                if var == "time":
                    arr = arr.astype('datetime64[ns]').astype('int64')
                if var in ["time", "yc", "xc"]:
                    arr = np.expand_dims(arr,axis=0)
                B_array.append(torch.from_numpy(arr).float())
            # Stack B (B, T, H, W)
            coarse_dict[var] = torch.stack(B_array, dim=0)
            # Récupérer tous les champs
            # fields = batch._fields
            # Construire un nouveau dict avec les valeurs de coarse_dict
            # ou None par défaut si clé manquante
            # complete_dict = {field: coarse_dict.get(field, None) for field in fields}
        # return type(batch)(**complete_dict)
            fields = self.tgt_vars + pred_vars + ["time", "yc", "xc"]
            complete_dict = {field: coarse_dict.get(field, None) for field in fields}
        return complete_dict

    def on_test_start(self):
        # Handle both dict (multiple dataloaders) and single DataLoader
        tdl = self.trainer.test_dataloaders
        if isinstance(tdl, dict):
            self.dataloader_keys = list(tdl.keys())
            self.num_test_batches = {
                i: len(dl)
                for i, dl in enumerate(tdl.values())
            }
        else:
            # Single DataLoader — wrap it
            self.dataloader_keys = [0]
            self.num_test_batches = {0: len(tdl)}

    def is_last_batch(self, batch_idx, dataloader_idx):
        total_batches = self.num_test_batches[dataloader_idx]
        return batch_idx == total_batches - 1

    def _is_simple_datamodule(self):
        """Return True if the datamodule is a *_simplify variant."""
        dm = getattr(self, '_datamodule', None) or getattr(self.trainer, 'datamodule', None)
        return dm is not None and 'simplif' in type(dm).__name__.lower()

    def _simple_test_step(self, batch, batch_idx, dataloader_idx=0):
        """Lightweight test step for *_simplify datamodules.

        Just does a forward pass per resolution and computes RMSE / μ-score
        (pred vs tgt) on normalised data.  Results are printed and logged.
        No reconstruction, no xarray, no multi-daw aggregation.
        """
        if dataloader_idx is None:
            dataloader_idx = 0
        res = self.multires[dataloader_idx]

        # Simple datamodule returns a dict {patch_xN: NamedTuple}; extract the
        # resolution-specific item before calling methods that expect a NamedTuple.
        if isinstance(batch, dict):
            res_key = f"patch_x{res}"
            batch = batch.get(res_key, next(iter(batch.values())))

        batch = self.modify_batch(batch, res)
        sbatch = self.format_batch_for_solver(batch, include_masks=self.include_masks, res=res)
        out = self(batch=sbatch, res=res)
        out = self.split_tensor_to_dict(out, res=res)

        tgt_vars = self._get_target_vars_for_resolution(res)
        metrics = {}
        for var in tgt_vars:
            var_suffix = var.split('_', 1)[1] if '_' in var else var
            pred = out[f'pred_{var_suffix}']
            tgt  = getattr(batch, var)
            valid = tgt.isfinite() & pred.isfinite()
            if valid.any():
                diff   = (pred[valid] - tgt[valid])
                rmse   = diff.pow(2).mean().sqrt().item()
                mu     = 1.0 - rmse / (tgt[valid].std().item() + 1e-8)
                metrics[f'test_rmse_{var}_x{res}'] = rmse
                metrics[f'test_mu_{var}_x{res}']   = mu
            else:
                metrics[f'test_rmse_{var}_x{res}'] = float('nan')
                metrics[f'test_mu_{var}_x{res}']   = float('nan')

        for k, v in metrics.items():
            self.log(k, v, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        if self.trainer.is_global_zero and batch_idx % 20 == 0:
            print(f"\n[simple_test_step] batch={batch_idx} res=x{res}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        # ── Simple datamodule path ────────────────────────────────────────
        if self._is_simple_datamodule():
            return self._simple_test_step(batch, batch_idx, dataloader_idx)

        # ── Full multi-resolution path (default) ──────────────────────────
        # Fix for single resolution
        if dataloader_idx is None:
            dataloader_idx = 0
        res = self.multires[dataloader_idx]
        res_key = f"patch_x{res}"
        last = self.len_daw[res]

        print(f"Dataloader_{dataloader_idx}, Batch_{batch_idx}, res_{res}")
        if (dataloader_idx == 0) and (batch_idx == 0) :
            self.test_data = {}
            self.test_times = {}
            self.test_coords = {}
            self.aggregate_results = {}

        if batch_idx == 0:
            self.test_data[res_key] = []
            self.test_times[res_key] = []
            self.test_coords[res_key] = []
            
        batch = self.modify_batch(batch, res)
        # Determine device from batch
        device = batch.tgt_sic.device if hasattr(batch, 'tgt_sic') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # anomaly conversion
        if dataloader_idx > 0:
            coarser_res = self.multires[dataloader_idx-1]
            # project coarser_res batch on res batch
            xc_target = torch.squeeze(batch.xc, dim=1)
            yc_target = torch.squeeze(batch.yc, dim=1)
            # identify batch daw / coarse daw equivalence for selection
            coarse = self.aggregate_results[f"patch_x{coarser_res}"]
            coarse = {
               k: v.isel(time=np.arange(self.len_daw[coarser_res]-last,
                                        self.len_daw[coarser_res]))
               for k, v in coarse.items()
            }
            coarse = self.convert_xr_to_batch(coarse, batch, res=res)       
            # Move all coarse tensors to device and filter out None values
            coarse = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in coarse.items()
                if v is not None  # Skip None values
            }
            xc_coarse = torch.squeeze(coarse["xc"], dim=1)
            yc_coarse = torch.squeeze(coarse["yc"], dim=1)
            itrp_coarse = self.interpolate_torch(coarse,#._asdict(),
                                                 xc_coarse, yc_coarse,
                                                 xc_target, yc_target)
            #itrp_coarse = self.crop_daw(itrp_coarse,res)
            # modify batch to work on anomaly compared to coarser resolution
            batch = self.update_batch_as_anomaly(batch,
                                                 {k: v for k, v in itrp_coarse.items() if k.startswith('pred_')}
                            )
            # Save original (pre-normalisation) anomaly targets so that tgt_norm
            # can reconstruct the true full-field ground truth later.
            tgt_vars_for_res = self._get_target_vars_for_resolution(res)
            orig_tgt = {var: getattr(batch, var).clone() for var in tgt_vars_for_res}
            # Instance-normalise the anomaly so the solver always sees std≈1
            if self.normalize_anomaly:
                batch, anom_scale = self.normalize_anomaly_batch(batch)
            else:
                anom_scale = {}

            scale_channel = None
            if self.condition_on_scale:
                scale_channel = self.compute_scale_channel(
                    {k: v for k, v in itrp_coarse.items() if k.startswith('pred_')}
                )
        else:
            anom_scale = {}
            orig_tgt = None
            scale_channel = None

        sbatch = self.format_batch_for_solver(batch, include_masks=self.include_masks, res=res,
                                               scale_channel=scale_channel)

        out = self(batch=sbatch, res=res)
        out = self.split_tensor_to_dict(out, res=res)

        # Denormalise predictions before adding back the coarse resolution
        if anom_scale:
            out = self.denormalize_anomaly_predictions(out, anom_scale)

        # Get resolution-specific target vars
        tgt_vars = self._get_target_vars_for_resolution(res)
        
        # add coarser resolution to output
        if dataloader_idx > 0:
            out = {k: out[k] + itrp_coarse[k] for k in out}

        # Masque de domaine : si une variable models_XXX est présente dans le batch,
        # on utilise ses NaN comme masque (NaN → pixel invalide/hors domaine).
        # Sinon on replie sur le land_mask classique (land_mask==1 → invalide).
        batch_dict = batch._asdict()
        models_mask_var = next(
            (k for k in batch_dict if k.startswith("models_")
             and isinstance(batch_dict[k], torch.Tensor)
             and batch_dict[k].numel() > 0),
            None
        )
        if models_mask_var is not None:
            # True où le pixel est invalide (au moins un NaN sur l'axe temporel)
            domain_invalid = ~batch_dict[models_mask_var].isfinite().any(dim=1, keepdim=True)
        else:
            domain_invalid = (batch.land_mask == 1.)

        for var in out:
            out[var] = torch.where(domain_invalid, torch.tensor(float('nan'), device=out[var].device, dtype=out[var].dtype), out[var])

        # Stockage des sorties et des cibles
        # Unnormalization is done in aggregate
        out_norm, tgt_norm = {}, {}
        for i, var in enumerate(tgt_vars):
            # Convert var_name to pred_var_name
            if '_' in var:
                var_suffix = var.split('_', 1)[1]
                pred_var_name = f'pred_{var_suffix}'
            else:
                pred_var_name = f'pred_{var}'
            pred = out[pred_var_name]
            out_norm[pred_var_name] = pred
            # Use the pre-normalisation anomaly so that adding back the coarse
            # field yields the correct full-field ground truth.
            if orig_tgt is not None:
                tgt_norm[var] = orig_tgt[var]
            else:
                tgt_norm[var] = getattr(batch, var)
            if dataloader_idx > 0:
                tgt_norm[var] = tgt_norm[var] + itrp_coarse[pred_var_name]
        
        # apply constraints
        out_norm = self._apply_constraints(out_norm, res)
        #tgt_norm = self._apply_constraints(tgt_norm, res)

        obs_norm = {field: getattr(batch, field) for field in self._get_obs_var_names(dataloader_idx)}

        combined = list(out_norm.values()) + list(tgt_norm.values()) + list(obs_norm.values())
        stacked = torch.stack(combined, dim=1)

        # stacked has shape (B,V,T,H,W) with V the number of variables
        self.test_data[res_key].append(stacked)
        self.test_times[res_key].append(torch.squeeze(batch.time, dim=1))

        # Store patch spatial coordinates per item in the batch so that
        # reconstruction can use explicit coords and is order-independent
        # (needed for multi-GPU where patches arrive out of global order).
        # batch.yc / batch.xc: (B, 1, H) / (B, 1, W)  →  squeeze to (H,) / (W,)
        batch_size = stacked.shape[0]
        patch_coords_batch = []
        for b in range(batch_size):
            yc_b = batch.yc[b].squeeze().detach().cpu().numpy()
            xc_b = batch.xc[b].squeeze().detach().cpu().numpy()
            patch_coords_batch.append((yc_b, xc_b))
        self.test_coords[res_key].append(patch_coords_batch)

        # If last batch for this dataloader, aggregate immediately so that
        # finer-resolution dataloaders can access aggregate_results[res_key].
        # The helper handles multi-GPU gathering before reconstruction.
        if self.is_last_batch(batch_idx, dataloader_idx):
            idx_rec = np.arange(batch.time.shape[-1])
            self._finalize_res(dataloader_idx, idx_rec, write_netcdf=True)

        batch, out = None, None

    @property
    def test_quantities(self):
        # Create pred_suffix list, then keep original tgt_vars names
        pred_vars = []
        for var in self.tgt_vars:
            if "_" in var:
                suffix = var.split("_", 1)[-1]
                pred_vars.append(f"pred_{suffix}")
            else:
                pred_vars.append(f"pred_{var}")
        result = pred_vars + self.tgt_vars
        #print(f"\n[DEBUG test_quantities] self.tgt_vars: {self.tgt_vars}")
        #print(f"[DEBUG test_quantities] result: {result}")
        return result

    def _finalize_res(self, dataloader_idx, idx_rec, write_netcdf=True):
        """
        Flatten locally accumulated patches for one resolution, optionally
        gather them across all GPUs (DDP), then reconstruct on rank 0.

        This is called from ``test_step`` at ``is_last_batch`` so that
        ``aggregate_results[res_key]`` is available before the next
        (finer) resolution's dataloader begins.
        """
        res     = self.multires[dataloader_idx]
        res_key = f"patch_x{res}"

        # ── Flatten per-batch lists ───────────────────────────────────────────
        data   = list(itertools.chain(*self.test_data[res_key]))
        times  = list(itertools.chain(*self.test_times[res_key]))
        coords = list(itertools.chain(*self.test_coords[res_key]))

        # ── Multi-GPU gathering (collective → all ranks must enter) ───────────
        if self.trainer.world_size > 1:
            import torch.distributed as dist
            gathered = [None] * self.trainer.world_size
            dist.all_gather_object(
                gathered,
                {
                    'data':   data,
                    'times':  [t.cpu() for t in times],
                    'coords': coords,
                }
            )
            data   = [item for g in gathered for item in g['data']]
            times  = [t    for g in gathered for t    in g['times']]
            coords = [c    for g in gathered for c    in g['coords']]

        # ── Reconstruction on rank 0 only, then broadcast to all ranks ────────
        # Running aggregate_batches on every rank causes hangs (xarray/joblib
        # worker contention).  Instead rank 0 reconstructs and broadcasts the
        # resulting xr.Dataset dict to the other ranks so that every rank has
        # aggregate_results[res_key] populated (required by the finer-res
        # test_step: coarse = self.aggregate_results[...]).
        if self.trainer.world_size > 1:
            import torch.distributed as dist
            if self.trainer.is_global_zero:
                result = self.aggregate_batches(
                    idx_rec,
                    data,
                    times,
                    dataloader_idx,
                    metrics=False,
                    write_netcdf=write_netcdf,
                    patch_coords=coords,
                )
                print(result)
            else:
                result = None
            # broadcast_object_list is a collective: all ranks must call it
            container = [result]
            dist.broadcast_object_list(container, src=0)
            self.aggregate_results[res_key] = container[0]
        else:
            # Single-GPU: straightforward
            self.aggregate_results[res_key] = self.aggregate_batches(
                idx_rec,
                data,
                times,
                dataloader_idx,
                metrics=False,
                write_netcdf=write_netcdf,
                patch_coords=coords,
            )
            print(self.aggregate_results[res_key])

    def on_test_epoch_end(self):
        # Aggregation is done inside test_step via _finalize_res.
        pass

    def on_load_checkpoint(self, checkpoint):
        """
        Handle weight tensor size mismatches between training and inference.
        
        This is crucial when using different patch_dims for test vs train:
        - Training: patch_dims = {time: 15, yc: 256, xc: 256}
        - Test: patch_dims = {time: 15, yc: 294, xc: 304}
        
        The weight tensors (_rec_weight_*, _optim_weight_*, _prior_weight_*) are
        initialized based on patch_dims, so they differ between train and test.
        
        Solution: Replace checkpoint weights with current model's weights.
        """
        print("\n" + "="*60)
        print("Loading checkpoint with weight adaptation...")
        print("="*60)
        
        current_state = self.state_dict()
        checkpoint_state = checkpoint["state_dict"]
        
        # Keys that need size adaptation
        weight_prefixes = ["_rec_weight", "_optim_weight", "_prior_weight", "_sr_weight"]
        
        adapted_keys = []
        for key in current_state.keys():
            if any(key.startswith(prefix) for prefix in weight_prefixes):
                checkpoint_shape = checkpoint_state.get(key, torch.empty(0)).shape
                current_shape = current_state[key].shape
                
                if checkpoint_shape != current_shape:
                    print(f"  ⚠️  Size mismatch for '{key}':")
                    print(f"      Checkpoint: {checkpoint_shape}")
                    print(f"      Current:    {current_shape}")
                    print(f"      → Using current model's weight")
                    
                    #  Replace with current model's weight
                    checkpoint_state[key] = current_state[key]
                    adapted_keys.append(key)
        
        if adapted_keys:
            print(f"\n   Adapted {len(adapted_keys)} weight tensors:")
            for key in adapted_keys:
                print(f"      - {key}")
        else:
            print("   No weight adaptation needed (shapes match)")
        
        print("="*60 + "\n")
        
        #  Update checkpoint with adapted weights
        checkpoint["state_dict"] = checkpoint_state
