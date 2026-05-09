import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr

class GradSolver(nn.Module):
    def __init__(self, 
                 prior_cost, 
                 obs_cost, 
                 grad_mod, 
                 n_step,
                 input_vars,
                 target_vars,
                 var_mapping,
                 n_time,
                 lr_grad=0.2, 
                 include_masks=False,
                 **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod
        self.n_step = n_step
        self.lr_grad = lr_grad
        self.include_masks = include_masks
        
        # Store variable configuration
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.var_mapping = var_mapping
        self.n_time = n_time
        
        # Validate mapping
        for tgt_var in target_vars:
            if tgt_var not in var_mapping:
                raise ValueError(f"Target variable '{tgt_var}' not found in var_mapping")
            if var_mapping[tgt_var] not in input_vars:
                raise ValueError(f"Mapped input variable '{var_mapping[tgt_var]}' not found in input_vars")
        
        # Compute channel dimensions
        self.n_input_vars = len(input_vars)
        self.n_target_vars = len(target_vars)
        
        # Channel dimensions
        if include_masks:
            # INPUT: Each variable has data + mask intercalés
            self.channels_per_var_input = n_time * 2  # [t0_data, t0_mask, t1_data, t1_mask, ...]
            self.dim_input = self.n_input_vars * n_time * 2
            
            # OUTPUT: Only data channels (no masks)
            self.channels_per_var_output = n_time  # [t0_data, t1_data, t2_data, ...]
            self.dim_target = self.n_target_vars * n_time
        else:
            self.channels_per_var_input = n_time
            self.dim_input = self.n_input_vars * n_time
            self.channels_per_var_output = n_time
            self.dim_target = self.n_target_vars * n_time
        
        # Identify auxiliary variables
        self.auxiliary_vars = [v for v in input_vars if v not in var_mapping.values()]
        
        # Compute target DATA indices (only DATA channels, not masks)
        self.target_data_indices = []
        for tgt_var in target_vars:
            inp_var = var_mapping[tgt_var]
            var_idx = input_vars.index(inp_var)
            
            if include_masks:
                # Extract only DATA channels (even indices)
                start_idx = var_idx * n_time * 2
                for t in range(n_time):
                    self.target_data_indices.append(start_idx + t * 2)  # Even indices
            else:
                start_idx = var_idx * n_time
                self.target_data_indices.extend(range(start_idx, start_idx + n_time))
        
        # Compute target WITH MASK indices (for grad_mod context)
        self.target_with_mask_indices = []
        if include_masks:
            for tgt_var in target_vars:
                inp_var = var_mapping[tgt_var]
                var_idx = input_vars.index(inp_var)
                start_idx = var_idx * n_time * 2
                end_idx = start_idx + n_time * 2
                self.target_with_mask_indices.extend(range(start_idx, end_idx))
        else:
            self.target_with_mask_indices = self.target_data_indices
        
        self._grad_norm = None
        
        print(f"GradSolver initialized:")
        print(f"  Input vars: {input_vars}")
        print(f"  Target vars: {target_vars}")
        print(f"  Mapping: {var_mapping}")
        print(f"  Include masks: {include_masks}")
        print(f"  Input channels per var: {self.channels_per_var_input}")
        print(f"  Output channels per var: {self.channels_per_var_output}")
        print(f"  Dimensions: input={self.dim_input}, target={self.dim_target}")
        print(f"  Target DATA indices: {self.target_data_indices}")
        if include_masks:
            print(f"  Target WITH MASK indices: {self.target_with_mask_indices}")
    
    def split_by_variables(self, tensor, var_names, is_input=True):
        """
        Split a tensor (B, C, H, W) into dict by variable names.
        
        Args:
            tensor: (B, C, H, W)
            var_names: List of variable names
            is_input: If True, expects input format (with masks if include_masks=True)
                     If False, expects output format (data only, no masks)
            
        Returns:
            dict {var_name: (B, N_channels, H, W)}
        """
        B, C, H, W = tensor.shape
        n_vars = len(var_names)
        
        if is_input:
            channels_per_var = self.channels_per_var_input
        else:
            channels_per_var = self.channels_per_var_output
        
        expected_channels = n_vars * channels_per_var
        
        assert C == expected_channels, \
            f"Expected C={expected_channels} ({n_vars} vars × {channels_per_var} ch/var), got {C}"
        
        # Reshape: (B, N_vars * N_ch, H, W) -> (B, N_vars, N_ch, H, W)
        tensor_reshaped = tensor.view(B, n_vars, channels_per_var, H, W)
        
        # Split by variable
        var_dict = {}
        for i, var_name in enumerate(var_names):
            var_dict[var_name] = tensor_reshaped[:, i]  # (B, N_ch, H, W)
        
        return var_dict
    
    def merge_variables(self, var_dict, var_names, is_input=True, requires_grad=True):
        """
        Merge dict of variables into single tensor.
        
        Args:
            var_dict: dict {var_name: (B, N_channels, H, W)}
            var_names: List of variable names in desired order
            is_input: If True, expects input format (with masks)
                     If False, expects output format (data only)
            requires_grad: Whether to keep gradients
            
        Returns:
            tensor: (B, N_vars * N_channels, H, W)
        """
        var_tensors = [var_dict[var_name] for var_name in var_names]
        stacked = torch.stack(var_tensors, dim=1)  # (B, N_vars, N_channels, H, W)
        
        B, N_vars, N_channels, H, W = stacked.shape
        merged = stacked.view(B, N_vars * N_channels, H, W)
        
        if not requires_grad:
            merged = merged.detach()
        
        return merged

    def init_state(self, batch, x_init=None, random=False):
        """
        Initialize state as dict of variables.
        Input variables have shape (B, channels_per_var_input, H, W) - WITH masks
        Target variables have shape (B, channels_per_var_output, H, W) - WITHOUT masks
        """
        if x_init is not None:
            return x_init

        # Split input (WITH masks if include_masks=True)
        input_dict = self.split_by_variables(
            batch.input.nan_to_num(), 
            self.input_vars,
            is_input=True
        )

        state_dict = {}
        
        # Initialize input variables (WITH masks)
        for inp_var in self.input_vars:
            is_target_source = inp_var in self.var_mapping.values()
            
            if is_target_source:
                if random:
                    B, C, H, W = input_dict[inp_var].shape
                    device = batch.input.device
                    
                    if self.include_masks:
                        # Generate random data, keep original masks
                        random_data = torch.randn(B, C, H, W, device=device)
                        random_input = input_dict[inp_var].clone()
                        random_input[:, ::2] = random_data[:, ::2]  # Replace data channels only
                        state_dict[inp_var] = random_input.requires_grad_(True)
                    else:
                        random_input = torch.randn(B, C, H, W, device=device)
                        state_dict[inp_var] = random_input.requires_grad_(True)
                else:
                    state_dict[inp_var] = input_dict[inp_var].clone().detach().requires_grad_(True)
            else:
                # Auxiliary variable (covariates)
                state_dict[inp_var] = input_dict[inp_var].clone().detach().requires_grad_(True)

        # Target variables: Extract DATA only (no masks)
        for tgt_var in self.target_vars:
            inp_var = self.var_mapping[tgt_var]
            inp_data_with_mask = input_dict[inp_var]  # (B, 2*n_time, H, W) if masks
            
            if self.include_masks:
                # Extract only data channels (even indices: 0, 2, 4, ...)
                data_only = inp_data_with_mask[:, ::2]  # (B, n_time, H, W)
            else:
                data_only = inp_data_with_mask
            
            state_dict[tgt_var] = data_only.clone().detach()  # (B, n_time, H, W)
        
        return state_dict

    def solver_step(self, state_dict, batch, step):
        """
        Solver step that updates only target-mapped input variables.
        
        Flow:
        1. Merge all input vars (WITH masks) -> full state
        2. Extract target state DATA only -> for optimization
        3. Extract target state WITH masks -> for grad_mod context
        4. Compute costs and gradients on DATA only
        5. Update DATA only
        6. Reconstruct [data, mask] format
        """
        target_source_vars = [self.var_mapping[tgt] for tgt in self.target_vars]
        
        # 1. Merge ALL input variables (WITH masks if include_masks=True)
        input_state = self.merge_variables(
            {k: state_dict[k] for k in self.input_vars},
            self.input_vars,
            is_input=True
        )  # (B, dim_input, H, W)
        
        # 2. Extract target state DATA only (for optimization)
        target_state_data = input_state[:, self.target_data_indices, :, :]
        # (B, dim_target, H, W) - data only

        # 3. Get observation variables (DATA only)
        obs = self.merge_variables(
            {k: state_dict[k] for k in self.target_vars},
            self.target_vars,
            is_input=False,
            requires_grad=False
        )  # (B, dim_target, H, W)

        # 4. Compute costs
        # prior_cost: uses full input_state (with masks) and target_state_data (no masks)
        prior_cost = self.prior_cost(input_state, target_state_data)
        obs_cost = self.obs_cost(target_state_data, obs)
        var_cost = prior_cost + obs_cost
        
        # 5. Compute gradient (on DATA only)
        grad = torch.autograd.grad(var_cost, target_state_data, create_graph=True)[0]
        
        # 6. Extract target WITH masks for grad_mod context
        if self.include_masks:
            target_context = input_state[:, self.target_with_mask_indices, :, :]
        else:
            target_context = target_state_data
        
        # 7. Apply gradient model (context with masks, grad on data only)
        gmod = self.grad_mod(target_context, grad)
        
        # 8. Update (DATA only)
        state_update = (
            1 / (step + 1) * gmod
            + self.lr_grad * (step + 1) / self.n_step * grad
        )
        
        new_target_state_data = target_state_data - state_update
        
        # 9. Split back into variables (DATA only)
        updated_dict = self.split_by_variables(
            new_target_state_data, 
            target_source_vars,
            is_input=False
        )
        
        # 10. Build new state dict
        new_state_dict = {}
        
        # Update target-mapped input variables: reconstruct [data, mask] format
        for inp_var in target_source_vars:
            updated_data = updated_dict[inp_var]  # (B, n_time, H, W)
            
            if self.include_masks:
                # Reconstruct [data, mask] intercalé
                old_var_with_mask = state_dict[inp_var]  # (B, 2*n_time, H, W)
                new_var_with_mask = old_var_with_mask.clone()
                new_var_with_mask[:, ::2] = updated_data  # Update data channels only
                # Masks (odd indices) remain unchanged
                new_state_dict[inp_var] = new_var_with_mask.requires_grad_(True)
            else:
                new_state_dict[inp_var] = updated_data.requires_grad_(True)

        # Keep other input variables unchanged (covariates, etc.)
        for inp_var in self.input_vars:
            if inp_var not in new_state_dict:
                new_state_dict[inp_var] = state_dict[inp_var]
        
        # Keep target ground truth unchanged
        for tgt_var in self.target_vars:
            new_state_dict[tgt_var] = state_dict[tgt_var]
        
        return new_state_dict
    
    def forward(self, batch):
        """Forward pass through the solver."""
        with torch.set_grad_enabled(True):
            state_dict = self.init_state(batch)
            
            target_source_vars = [self.var_mapping[tgt] for tgt in self.target_vars]
            
            # Initialize grad_mod with target context (WITH masks if available)
            if self.include_masks:
                target_init_list = []
                for var in target_source_vars:
                    var_with_mask = state_dict[var]  # (B, 2*n_time, H, W)
                    target_init_list.append(var_with_mask)
                target_init = torch.cat(target_init_list, dim=1)
            else:
                target_init_list = []
                for var in target_source_vars:
                    var_data = state_dict[var]  # (B, n_time, H, W)
                    target_init_list.append(var_data)
                target_init = torch.cat(target_init_list, dim=1)
            
            self.grad_mod.reset_state(target_init)
            
            # Iterative optimization
            for step in range(self.n_step):
                state_dict = self.solver_step(state_dict, batch, step=step)
                if not self.training:
                    for inp_var in target_source_vars:
                        state_dict[inp_var] = state_dict[inp_var].detach().requires_grad_(True)
        
        # Return optimized predictions (DATA only, no masks)
        if self.include_masks:
            output_list = []
            for var in target_source_vars:
                var_with_mask = state_dict[var]  # (B, 2*n_time, H, W)
                data_only = var_with_mask[:, ::2]  # (B, n_time, H, W)
                output_list.append(data_only)
            output = torch.cat(output_list, dim=1)
        else:
            output = self.merge_variables(
                {k: state_dict[k] for k in target_source_vars},
                target_source_vars,
                is_input=False
            )
        
        return output


class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
        """
        Args:
            dim_in: Input dimension WITH masks if solver uses masks
                   (N_target_vars * 2 * n_time if include_masks=True, else N_target_vars * n_time)
            dim_out: Output dimension WITHOUT masks (N_target_vars * n_time)
            dim_hidden: Hidden dimension
        """
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        # Process gradient (data only)
        self.gates = torch.nn.Conv2d(
            dim_out + dim_hidden,  # grad_data_only + hidden
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        
        # Context projection (for masks if available)
        self.context_proj = torch.nn.Conv2d(
            dim_in,  # Can include masks
            dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def reset_state(self, inp):
        """
        Args:
            inp: Initial state, can be with or without masks
        """
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, target_with_masks, grad_data_only):
        """
        Args:
            target_with_masks: (B, dim_in, H, W) - context (may include masks)
            grad_data_only: (B, dim_out, H, W) - gradient on data only
            
        Returns:
            (B, dim_out, H, W) - gradient modification on data only
        """
        x = grad_data_only
        
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x = x / self._grad_norm
        
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
        
        # Process context (with masks if available)
        context = self.context_proj(self.down(target_with_masks))
        
        # Combine gradient with hidden state
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        # Update cell with context
        cell = (remember_gate * cell) + (in_gate * cell_gate) + context
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        
        # Output: data only (no masks)
        out = self.conv_out(hidden)
        out = self.up(out)
        
        return out


class GradSolvers(nn.Module):
    def __init__(self, solvers, **kwargs):
        super().__init__()
        self.solvers = nn.ModuleDict(solvers)

    def forward(self, batch, res=1):
        return self.solvers[f"solver_x{res}"](batch)


class BaseObsCost(nn.Module):
    def __init__(self, w=1, use_target=True) -> None:
        """
        Args:
            w: Weight for the observation cost
            use_target: If True, compare state with batch.tgt (target variables)
                       If False, compare with batch.input (input variables)
        """
        super().__init__()
        self.w = w
        self.use_target = use_target

    def forward(self, state, obs):
        """
        Both state and obs are DATA only (no masks)
        
        Args:
            state: (B, N_target_vars * N_time, H, W) - predicted target variables
            obs: (B, N_target_vars * N_time, H, W) - ground truth observations  
        
        Returns:
            Observation cost (scalar)
        """
        msk = obs.isfinite()
        return self.w * F.mse_loss(state[msk], obs[msk])


class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, kernel_size=3, downsamp=None, 
                 bilin_quad=True):
        """
        Args:
            dim_in: Full input dimension (all vars, WITH masks if solver uses masks)
            dim_out: Target output dimension (target vars, WITHOUT masks)
            dim_hidden: Hidden dimension
            kernel_size: Convolutional kernel size
            downsamp: Downsampling factor (None for no downsampling)
            bilin_quad: Use quadratic (True) or bilinear (False) nonlinearity
        """
        super().__init__()
        self.bilin_quad = bilin_quad
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        # Input: full state (all vars with masks if available)
        self.conv_in = nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        
        self.conv_hidden = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.bilin_1 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_21 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_22 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # Output: target data only (no masks)
        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward_ae(self, state_full):
        """
        Args:
            state_full: (B, dim_in, H, W) - all vars with masks if available
            
        Returns:
            (B, dim_out, H, W) - reconstructed target data (no masks)
        """
        x = self.down(state_full)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state_full, target_state_data):
        """
        Args:
            state_full: (B, dim_in, H, W) - all vars with masks if available
            target_state_data: (B, dim_out, H, W) - target data only (no masks)
            
        Returns:
            Prior cost (scalar)
        """
        # Reconstruct target data from full state
        reconstructed = self.forward_ae(state_full)  # (B, dim_out, H, W)
        
        return F.mse_loss(target_state_data, reconstructed)