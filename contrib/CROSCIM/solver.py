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
                 input_grad_update,
                 input_vars,           # e.g., ['asip_sic', 'cimr_sic', 'cimr_SIT', 'aux_var1', ...]
                 target_vars,          # e.g., ['tgt_sic', 'tgt_SIT']
                 var_mapping,          # e.g., {'tgt_sic': 'asip_sic', 'tgt_SIT': 'cimr_SIT'}
                 n_time,               # Number of time steps per variable
                 lr_grad=0.2, 
                 **kwargs):
        """
        GradSolver that handles different input/output variables with explicit mapping.
        
        Args:
            input_vars: List of input variable names (e.g., ['asip_sic', 'cimr_sic', 'cimr_SIT', ...])
            target_vars: List of target variable names (e.g., ['tgt_sic', 'tgt_SIT'])
            var_mapping: Dict mapping target vars to input vars 
                        (e.g., {'tgt_sic': 'asip_sic', 'tgt_SIT': 'cimr_SIT'})
            n_time: Number of time steps per variable
        """
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod
        self.n_step = n_step
        self.lr_grad = lr_grad
        
        # Store variable configuration
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.var_mapping = var_mapping  # Explicit mapping: target -> input
        self.n_time = n_time
        self.input_grad_update = input_grad_update
        
        # Validate mapping
        for tgt_var in target_vars:
            if tgt_var not in var_mapping:
                raise ValueError(f"Target variable '{tgt_var}' not found in var_mapping")
            if var_mapping[tgt_var] not in input_vars:
                raise ValueError(f"Mapped input variable '{var_mapping[tgt_var]}' not found in input_vars")
        
        # Compute channel dimensions
        self.n_input_vars = len(input_vars)
        self.n_target_vars = len(target_vars)
        self.dim_input = self.n_input_vars * n_time
        self.dim_target = self.n_target_vars * n_time
        
        # Identify auxiliary variables (input vars not mapped to any target)
        self.auxiliary_vars = [v for v in input_vars if v not in var_mapping.values()]
        
        # Compute target indices for prior_cost
        # These are the channel indices in the input_state that correspond to target variables
        target_indices = []
        for tgt_var in target_vars:
            inp_var = var_mapping[tgt_var]
            var_idx = input_vars.index(inp_var)
            # Each variable occupies n_time channels
            start_idx = var_idx * n_time
            end_idx = start_idx + n_time
            target_indices.extend(range(start_idx, end_idx))
        
        self.target_indices = target_indices
        
        # Set target_indices in prior_cost if it supports it
        if hasattr(self.prior_cost, 'target_indices'):
            self.prior_cost.target_indices = target_indices
            print(f"Set target_indices in prior_cost: {target_indices}")
        
        self._grad_norm = None
        
        print(f"GradSolver initialized:")
        print(f"  Input vars: {input_vars}")
        print(f"  Target vars: {target_vars}")
        print(f"  Mapping: {var_mapping}")
        print(f"  Auxiliary vars: {self.auxiliary_vars}")
        print(f"  Dimensions: input={self.dim_input}, target={self.dim_target}")
        print(f"  Target channel indices in input_state: {target_indices}")
    
    def split_by_variables(self, tensor, var_names):
        """
        Split a tensor (B, C, H, W) into dict by variable names.
        
        Args:
            tensor: (B, N_vars * N_time, H, W)
            var_names: List of variable names
            
        Returns:
            dict {var_name: (B, N_time, H, W)}
        """
        B, C, H, W = tensor.shape
        n_vars = len(var_names)
        
        assert C == n_vars * self.n_time, \
            f"Expected C={n_vars * self.n_time} ({n_vars} vars × {self.n_time} time), got {C}"
        
        # Reshape: (B, N_vars * N_time, H, W) -> (B, N_vars, N_time, H, W)
        tensor_reshaped = tensor.view(B, n_vars, self.n_time, H, W)
        
        # Split by variable
        var_dict = {}
        for i, var_name in enumerate(var_names):
            var_dict[var_name] = tensor_reshaped[:, i]  # (B, N_time, H, W)
        
        return var_dict
    
    def merge_variables(self, var_dict, var_names, requires_grad=True):
        """
        Merge dict of variables into single tensor.
        
        Args:
            var_dict: dict {var_name: (B, N_time, H, W)}
            var_names: List of variable names in desired order
            
        Returns:
            tensor: (B, N_vars * N_time, H, W)
        """
        # Stack variables: list of (B, N_time, H, W) -> (B, N_vars, N_time, H, W)
        var_tensors = [var_dict[var_name] for var_name in var_names]
        stacked = torch.stack(var_tensors, dim=1)
        
        # Reshape: (B, N_vars, N_time, H, W) -> (B, N_vars * N_time, H, W)
        B, N_vars, N_time, H, W = stacked.shape
        merged = stacked.view(B, N_vars * N_time, H, W)
        
        # Only detach if requires_grad=False
        # Don't use .requires_grad_(True) as it creates a new leaf
        if not requires_grad:
            merged = merged.detach()
        
        return merged

    def init_state(self, batch, x_init=None, random=False):
        """
        Initialize state as dict of variables.
        Target variables are initialized from their corresponding input variables.
        """
        if x_init is not None:
            return x_init

        # Split input into variables
        input_dict = self.split_by_variables(
            batch.input.nan_to_num(), 
            self.input_vars
        )

        state_dict = {}
        
        # Initialize ALL input variables from batch.input
        # The ones mapped to targets will be optimized, others are auxiliary
        for inp_var in self.input_vars:
            # Check if this input variable is mapped to a target
            is_target_source = inp_var in self.var_mapping.values()
            if is_target_source:
                # This variable will be optimized (e.g., 'asip_sic', 'cimr_SIT')
                if random:
                    B, C, H, W = input_dict[inp_var].shape
                    device = batch.input.device
                    random_input = torch.randn(B, C, H, W, device=device)
                    state_dict[inp_var] = random_input.requires_grad_(True)
                else:
                    state_dict[inp_var] = input_dict[inp_var].clone().detach().requires_grad_(True)
            else:
                # This is an auxiliary variable (e.g., 'cimr_SIC', 'msl', 't2m')
                state_dict[inp_var] = input_dict[inp_var].clone().detach().requires_grad_(True)

        # Keep target variables (ground truth, read-only, used for obs_cost)
        for tgt_var in self.target_vars:
            state_dict[tgt_var] = input_dict[self.var_mapping[tgt_var]].clone().detach()
        
        #print(f"\nInitialized state:")
        #print(f"  Variables with grad: {[k for k, v in state_dict.items() if v.requires_grad]}")
        #print(f"  Variables without grad: {[k for k, v in state_dict.items() if not v.requires_grad]}")
        
        return state_dict

    def solver_step(self, state_dict, batch, step, alpha_step=1.):
        """
        Solver step that updates only target-mapped input variables.
        
        Args:
            state_dict: dict {var_name: (B, N_time, H, W)}
        """



        # Get target-mapped input variables (the ones being optimized)
        # e.g., 'asip_sic', 'cimr_SIT'
        target_source_vars = [self.var_mapping[tgt] for tgt in self.target_vars]
        
        # Merge ALL input variables for prior cost context
        input_state = self.merge_variables(
            {k: state_dict[k] for k in self.input_vars},
            self.input_vars
        )  # (B, N_input_vars * N_time, H, W)
        
        target_state = input_state[:, self.target_indices, :, :]

        # Get observation variables (ground truth) for obs cost
        obs = self.merge_variables(
            {k: state_dict[k] for k in self.target_vars},
            self.target_vars,
            requires_grad=False
        )  # (B, N_target_vars * N_time, H, W)


        if( isinstance(step, float) ):
            device = batch.input.device
            t = torch.tensor([step], device=device).repeat(obs.shape[0])
        else:
            t = step

        if 'subgrad' in self.input_grad_update :
            gobs = (obs-target_state).nan_to_num()

            gprior = target_state - self.prior_cost.forward_ae(input_state)
            grad = torch.concatenate((gobs,gprior),dim=1)

            if 'state' in self.input_grad_update :
                grad = torch.concatenate((grad,target_state),dim=1)


        elif 'gradsplit' in self.input_grad_update :
            prior_cost = self.prior_cost(input_state, target_state)
            obs_cost = self.obs_cost(target_state, obs)
            # Compute full gradient
            grad_prior = torch.autograd.grad(prior_cost, target_state, create_graph=True)[0]
            grad_obs = torch.autograd.grad(obs_cost, target_state, create_graph=True)[0]
            grad = torch.concatenate((grad_prior, grad_obs),dim=1)
            if 'state' in self.input_grad_update :
                grad = torch.concatenate((grad,target_state),dim=1)   


        elif 'grad' in self.input_grad_update :
            prior_cost = self.prior_cost(input_state, target_state)
            obs_cost = self.obs_cost(target_state, obs)
            var_cost = prior_cost + obs_cost
            # Compute full gradient
            grad = torch.autograd.grad(var_cost, target_state, create_graph=True)[0]

            if 'state' in self.input_grad_update :
                grad = torch.concatenate(( grad,target_state),dim=1)



        elif  self.input_grad_update == 'obs+state' :
            grad = torch.concatenate((input_state,obs),dim=1)

        elif  self.input_grad_update == 'obs' :
            grad = input_state


        
        # Check gradient
        nan_count = (~grad.isfinite()).sum().item()
        total_count = target_state.numel()
        nan_pct = 100 * nan_count / total_count
        zero_count = (grad == 0).sum().item()
        zero_pct = 100 * zero_count / total_count
        
        #print(f"  FINAL grad: NaN%={nan_pct:.2f}%, Zero%={zero_pct:.2f}%")

        # Apply gradient model 
        gmod = self.grad_mod(grad, timesteps=t, extra=[])
        
        # Compute state 
        # update
        state_update = alpha_step * gmod
        if ( 'grad' in self.input_grad_update ) and ( self.lr_grad > 0. ) : 
            state_update += self.lr_grad * (step + 1) / self.n_step * grad[:,:target_state.shape[1],:,:]

        
        # Update target-mapped variables
        new_target_state = target_state - state_update
        
        # Split back into variables
        updated_dict = self.split_by_variables(new_target_state, target_source_vars)
        
        # Build new state dict
        new_state_dict = {}
        
        # 1. Update target-mapped input variables (e.g., 'asip_sic', 'cimr_SIT')
        for inp_var in target_source_vars:
            new_state_dict[inp_var] = updated_dict[inp_var]
            # Re-enable gradients for next iteration
            #if self.training:
            new_state_dict[inp_var].requires_grad_(True)

        # 2. Keep other input variables unchanged (auxiliary variables)
        for inp_var in self.input_vars:
            if inp_var not in new_state_dict:
                new_state_dict[inp_var] = state_dict[inp_var]
        
        # 3. Keep target ground truth unchanged
        for tgt_var in self.target_vars:
            new_state_dict[tgt_var] = state_dict[tgt_var]
        
        return new_state_dict

    def forward(self, batch):
        """
        Forward pass through the solver.
        Returns only target variables as tensor.
        """
        with torch.set_grad_enabled(True):
            state_dict = self.init_state(batch)
            
            # Get target-mapped input variables for initialization
            target_source_vars = [self.var_mapping[tgt] for tgt in self.target_vars]
            
            # Initialize grad_mod with target dimensions
            target_init = self.merge_variables(
                {k: v for k, v in state_dict.items() if k in target_source_vars},
                target_source_vars
            )
            self.grad_mod.reset_state(target_init)
            
            # Iterative optimization
            for step in range(self.n_step):
                alpha_step = 1. / self.n_step
                state_dict = self.solver_step(state_dict, batch, step= step / self.n_step, alpha_step=alpha_step)
                if not self.training:
                    # Detach and re-enable gradients for target-mapped variables
                    for inp_var in target_source_vars:
                        state_dict[inp_var] = state_dict[inp_var].detach().requires_grad_(True)
        
        # Return target-mapped input variables (the optimized predictions)
        # These correspond to the target variables
        output = self.merge_variables(
            {k: state_dict[k] for k in target_source_vars},
            target_source_vars
        )
    
        return output
        #return target_init

class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
        super().__init__()
        self.dim_hidden = dim_hidden

        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
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
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def predict(self, x, timesteps=None, extra=None):
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x = x / self._grad_norm
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)
        out = self.up(out)
        return out


    def forward(self, x, timesteps=None, extra=None):
        #x = batch.input
        #x = x.nan_to_num()

        if timesteps is None:
            timesteps = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
        if extra is None:
            extra = []

        out = self.predict(x, timesteps, extra)


        #if self.dims+2 > len(batch.input.shape):
        #    out = out.view(out.shape[0], out.shape[2], out.shape[3], out.shape[4] ) # add channel dim if missing

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
            use_target: If True, compare state with batch.tgt (target variables)
                       If False, compare with batch.input (input variables)
        """
        super().__init__()
        self.w = w
        self.use_target = use_target

    def forward(self, state, obs):
        """
        state: (B, N_target_vars * N_time, H, W) - predicted target variables
        obs: (B, N_target_vars * N_time, H, W) - ground truth observations  
        """
        msk = obs.isfinite()
        return self.w * F.mse_loss(state[msk], obs[msk])

class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, kernel_size=3, downsamp=None, 
                 bilin_quad=True, target_indices=None):
        """
        Args:
            dim_in: Input dimension (N_input_vars * N_time)
            dim_hidden: Hidden dimension
            dim_out: Output dimension (N_target_vars * N_time)
            target_indices: Indices of target variables in the input state
                           e.g., if input_vars = ['asip_sic', 'cimr_SIC', 'cimr_SIT', 'msl']
                           and target_vars = ['tgt_sic', 'tgt_SIT'] (mapped to 'asip_sic', 'cimr_SIT')
                           then target_indices would select channels corresponding to 
                           'asip_sic' (0:n_time) and 'cimr_SIT' (2*n_time:3*n_time)
        """
        super().__init__()
        self.bilin_quad = bilin_quad
        self.target_indices = target_indices  # will be set by GradSolver
        
        self.conv_in = nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.gn = torch.nn.GroupNorm(
            num_groups=1, 
            num_channels=dim_hidden  
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

        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

        self.bilin_21 = nn.utils.spectral_norm(self.bilin_21)
        self.bilin_22 = nn.utils.spectral_norm(self.bilin_22)

    '''
    def forward_ae(self, x):
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))
        x = self.gn(x)

        #nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        nonlin = nonlin = self.bilin_21(x) * self.bilin_22(x)
        nonlin = torch.tanh(nonlin) * 5.0 
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = torch.tanh(x) * 5.0
        x = self.up(x)
        return x
    '''
    def forward_ae(self, x):
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state, target_state):
        """
        Args:
            state: (B, N_input_vars * N_time, H, W) - full input state
            
        Returns:
            Prior cost comparing reconstructed targets with actual targets from state
        """
        # Reconstruct target variables from full state
        reconstructed = self.forward_ae(state)  # (B, N_target_vars * N_time, H, W)
        
        return F.mse_loss(target_state, reconstructed)
    





class AEPriorCostTwoScale(torch.nn.Module):
    """
    A prior cost model using bilinear autoencoders.

    Attributes:
        bilin_quad (bool): Whether to use bilinear quadratic terms.
        conv_in (nn.Conv2d): Convolutional layer for input.
        conv_hidden (nn.Conv2d): Convolutional layer for hidden states.
        bilin_1 (nn.Conv2d): Bilinear layer 1.
        bilin_21 (nn.Conv2d): Bilinear layer 2 (part 1).
        bilin_22 (nn.Conv2d): Bilinear layer 2 (part 2).
        conv_out (nn.Conv2d): Convolutional layer for output.
        down (nn.Module): Downsampling layer.
        up (nn.Module): Upsampling layer.
    """

    def __init__(self, dim_in, dim_out, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True, bias=True):
        """
        Initialize the BilinAEPriorCost module.

        Args:
            dim_in (int): Number of input dimensions.
            dim_hidden (int): Number of hidden dimensions.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            downsamp (int, optional): Downsampling factor. Defaults to None.
            bilin_quad (bool, optional): Whether to use bilinear quadratic terms. Defaults to True.
        """
        super().__init__()
        self.bilin_quad = bilin_quad
        self.conv_in = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.conv_hidden = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_out, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )


        self.conv_in_lr = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.conv_hidden_lr = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.conv_out_lr = torch.nn.Conv2d(
            dim_hidden, dim_out, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )


        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else torch.nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )

    def forward_ae(self, x):
        """
        Perform the forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the autoencoder.
        """

        # coarse-scale processing
        x_ = self.down(x)
        x_ = self.conv_in_lr(x_)
        x_ = self.conv_hidden_lr(torch.nn.functional.relu(x_))
        x_ = self.conv_out_lr(torch.nn.functional.relu(x_))
        dx = self.up(x_)

        # fine-scale processing
        x = self.conv_in(x)
        x = self.conv_hidden(torch.nn.functional.relu(x))
        x = self.conv_out(torch.nn.functional.relu(x))

        return x + dx

    def forward(self, state, target_state):
        """
        Compute the prior cost using the autoencoder.

        Args:
            state (torch.Tensor): The current state tensor.

        Returns:
            torch.Tensor: The computed prior cost.
        """
        reconstructed = self.forward_ae(state)
        return F.mse_loss(target_state, reconstructed)





class GenericAEPriorCost(torch.nn.Module):
    """
    A prior cost model using bilinear autoencoders.

    Attributes:
        bilin_quad (bool): Whether to use bilinear quadratic terms.
        conv_in (nn.Conv2d): Convolutional layer for input.
        conv_hidden (nn.Conv2d): Convolutional layer for hidden states.
        bilin_1 (nn.Conv2d): Bilinear layer 1.
        bilin_21 (nn.Conv2d): Bilinear layer 2 (part 1).
        bilin_22 (nn.Conv2d): Bilinear layer 2 (part 2).
        conv_out (nn.Conv2d): Convolutional layer for output.
        down (nn.Module): Downsampling layer.
        up (nn.Module): Upsampling layer.
    """

    def __init__(self, model_ae):
        """
        Initialize the BilinAEPriorCost module.

        Args:
            dim_in (int): Number of input dimensions.
            dim_hidden (int): Number of hidden dimensions.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            downsamp (int, optional): Downsampling factor. Defaults to None.
            bilin_quad (bool, optional): Whether to use bilinear quadratic terms. Defaults to True.
        """
        super().__init__()

        self.model_ae = model_ae 

    def forward_ae(self, x):
        """
        Perform the forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the autoencoder.
        """
        return self.model_ae(x)

    def forward(self, state, target_state):
        """
        Compute the prior cost using the autoencoder.

        Args:
            state (torch.Tensor): The current state tensor.

        Returns:
            torch.Tensor: The computed prior cost.
        """
        reconstructed = self.forward_ae(state)
        return torch.nn.functional.mse_loss(target_state, reconstructed)



class GradModelWithCondition(torch.nn.Module):
    """
    A generic conditional model for gradient modulation.

    Attributes:
        grad_model : grad update model
    """

    def __init__(self, grad_model=False, dropout=0.,use_grad_norm=True):
        """
        Initialize the ConvLstmGradModel.

        Args:
            grad_model : grad update model
        """
        super().__init__()
        self.grad_model = grad_model
        self.dropout = torch.nn.Dropout(dropout)
        self.use_grad_norm = use_grad_norm


    def reset_state(self, inp):
        """
        Reset the internal state of the LSTM.

        Args:
            inp (torch.Tensor): Input tensor to determine state size.
        """
        self._grad_norm = None


    def forward(self, x, timesteps=None, extra=[]):
        """
        Perform the forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()


        #print('self._grad_norm in GradModelWithCondition:', self._grad_norm, flush=True)
        x = x / self._grad_norm

        x = self.dropout(x)



        out = self.grad_model.predict(x, timesteps=timesteps, extra=extra)

        return out
