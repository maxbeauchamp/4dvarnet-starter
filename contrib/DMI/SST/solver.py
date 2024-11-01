import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from contrib.DMI.SST.VAE import *

class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad

        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init
        return batch.input.nan_to_num().detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
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

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            #if not self.training:
            #    state = self.prior_cost.forward_ae(state)
            #state = self.prior_cost.forward_ae(state)
        return state


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


class BaseObsCost(nn.Module):
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w=w

    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return self.w * F.mse_loss(state[msk], batch.input.nan_to_num()[msk])

class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True, nt=None):
        super().__init__()
        self.nt = nt
        self.bilin_quad = bilin_quad
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

        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

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

    def forward(self, state, exclude_params=False):
        if not exclude_params:
            return F.mse_loss(state, self.forward_ae(state))
        else:
            return F.mse_loss(state[:,:self.nt,:,:], self.forward_ae(state)[:,:self.nt,:,:])

class GradSolver_wgeo(GradSolver):

    def init_state(self, batch, x_init=None):
        x_init = super().init_state(batch, x_init)
        coords_cov = torch.stack((batch.latv[:,0].nan_to_num(), 
                                  batch.lonv[:,0].nan_to_num(), 
                                  batch.land_mask[:,0].nan_to_num(),
                                  batch.topo[:,0].nan_to_num(), 
                                  batch.fg_std[:,0].nan_to_num()), dim=1).to(x_init.device)
        return (x_init, coords_cov)

    def solver_step(self, state, batch, step, x_prior=None):
        if isinstance(self.prior_cost,BilinAEPriorCost_wgeo):
            var_cost = self.prior_cost(state) + self.obs_cost(state[0], batch)
        else:
            var_cost = self.prior_cost(state, batch, x_prior) + self.obs_cost(state[0], batch)
        x, coords_cov = state
        grad = torch.autograd.grad(var_cost, x, create_graph=True)[0]

        x_update = (
            1 / (step + 1) * self.grad_mod(grad)
            + self.lr_grad * (step + 1) / self.n_step * grad
        )
        state = (x - x_update, coords_cov)
        return state

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            if not isinstance(self.prior_cost,BilinAEPriorCost_wgeo):
                # create a prior with VAE strategy
                x_prior = self.prior_cost.forward_ae(state, batch)
                state = self.init_state(batch, x_init=x_prior)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step, x_prior=x_prior)
                if not self.training:
                    state = [s.detach().requires_grad_(True) for s in state]

            #if not self.training:
            #    state = [self.prior_cost.forward_ae(state,batch), state[1]]
        return state[0]

class BilinAEPriorCost_wgeo(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True, nt=None):
        super().__init__()
        self.nt = nt
        self.bilin_quad = bilin_quad
        self.conv_in = nn.Conv2d(
            dim_in+5, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
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

        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.down2 = nn.MaxPool2d(downsamp) if downsamp is not None else nn.Identity()

        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward_ae(self, state):
        x = self.down(state[0])
        coords_cov = self.down2(state[1])
        x = self.conv_in(torch.cat((x,coords_cov),dim=1))
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state):
        return F.mse_loss(state[0], self.forward_ae(state))


class VAEPriorCost_wgeo(nn.Module):
    def __init__(self, dim_in, dim_out, path_weights, norm_stats_anom, norm_stats_state):
        super().__init__()
        self.vae = VAE(in_channels=dim_in, out_channels=dim_out)
        # load weights of the VAE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path_weights, map_location=device)
        self.vae.load_state_dict(ckpt)
        self.m_anom, self.s_anom = norm_stats_anom
        self.m_state, self.s_state = norm_stats_state

    def norm(self, x, m, s, forward=True):
        if forward:
            return (x-m)/s
        else:
            return (x*s)+m

    def forward_ae(self, state, batch):
        msk = batch.input.isfinite()
        anom = batch.input.nan_to_num()
        coarse = batch.coarse.nan_to_num()
        y = self.norm(self.norm(anom, self.m_anom, self.s_anom, forward=False)+coarse,self.m_state, self.s_state)
        y = self.norm(self.norm(anom, self.m_anom, self.s_anom, forward=False),self.m_state, self.s_state)
        inp = torch.cat((y, state[1]),dim=1)
        # VAE acts on SST not anomalies
        res = self.norm(self.norm(self.vae(inp), self.m_state, self.s_state,forward=False) - coarse, self.m_anom, self.s_anom)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3,1)
        ax[0].pcolormesh(y[0,3,:,:].detach().cpu())
        ax[1].pcolormesh(res[0,3,:,:].detach().cpu())
        ax[2].pcolormesh((self.vae(inp))[0,3,:,:].detach().cpu())
        plt.show()
        return res

    def forward(self, state, batch, x_prior=None):
        if x_prior==None:
            return F.mse_loss(state[0], self.forward_ae(state,batch))
        else:
            return F.mse_loss(state[0], x_prior)

