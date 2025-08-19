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
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, use_cov=False, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod
        self.n_step = n_step
        self.lr_grad = lr_grad
        self._grad_norm = None
        self.use_cov = use_cov

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init
        #return batch.input.nan_to_num().detach().requires_grad_(True)
        return batch.coarse.nan_to_num().detach().requires_grad_(True)

    def solver_step(self, state, batch, prior, step):
        var_cost = self.prior_cost(state, prior) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)
            if not self.use_cov:
                prior = self.prior_cost.forward_ae(batch.coarse.nan_to_num())
            else:
                prior = self.prior_cost.forward_ae(torch.cat([
                                        batch.coarse.nan_to_num(),
                                        #batch.lonv.nan_to_num()[:,[0],:,:],
                                        #batch.latv.nan_to_num()[:,[0],:,:],
                                        #batch.land_mask.nan_to_num()[:,[0],:,:],
                                        batch.t2m.nan_to_num(),
                                        batch.istl1.nan_to_num(),
                                        batch.sst.nan_to_num(),
                                        batch.skt.nan_to_num()],dim=1))
            #prior = batch.coarse.nan_to_num()
            for step in range(self.n_step):
                state = self.solver_step(state, batch, prior, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)
            state = torch.clip(state,min=0.,max=1.)
        return state, prior

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

    def forward(self, x):
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x =  x / self._grad_norm
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

class CondConvLstmGradModel(ConvLstmGradModel):

    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None, n_covs=0):
        super().__init__(dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None)
        self.n_covs = n_covs
        self.gates = torch.nn.Conv2d(
            dim_in*(self.n_covs+1) + dim_hidden, # + 3,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

class CondGradSolver(GradSolver):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, **kwargs):
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, **kwargs)

    def solver_step(self, state, batch, prior, step):
        var_cost = self.prior_cost(state, prior) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        gmod = self.grad_mod(torch.cat([grad,
                                        #batch.lonv.nan_to_num()[:,[0],:,:],
                                        #batch.latv.nan_to_num()[:,[0],:,:],
                                        #batch.land_mask.nan_to_num()[:,[0],:,:],
                                        batch.t2m.nan_to_num(),
                                        batch.istl1.nan_to_num(),
                                        batch.sst.nan_to_num(),
                                        batch.skt.nan_to_num()],dim=1))
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)
            prior = self.prior_cost.forward_ae(torch.cat([
                                        batch.coarse.nan_to_num(),
                                        #batch.lonv.nan_to_num()[:,[0],:,:],
                                        #batch.latv.nan_to_num()[:,[0],:,:],
                                        #batch.land_mask.nan_to_num()[:,[0],:,:],
                                        batch.t2m.nan_to_num(),
                                        batch.istl1.nan_to_num(),
                                        batch.sst.nan_to_num(),
                                        batch.skt.nan_to_num()],dim=1))
            for step in range(self.n_step):
                state = self.solver_step(state, batch, prior, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)
            state = torch.clip(state,min=0.,max=1.)
        return state, prior

class BaseObsCost(nn.Module):
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w=w

    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return self.w * F.mse_loss(state[msk], batch.input.nan_to_num()[msk])

class SRNNPriorCost(nn.Module):
    def __init__(self, srnn):
        super().__init__()
        self.srnn = srnn

    def forward_ae(self, x):
        res = self.srnn(x)
        res = torch.clip(res,min=0.,max=1.)
        return res

    def forward(self, state, prior):
         return F.mse_loss(state, prior)
