import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from contrib.DMI.SST.VAE import VAE as VAE_attention
from contrib.DMI.SST.solver import GradSolver_wgeo
from kornia.filters import box_blur as smoother

def smooth(inp, mask):
    sum_pool = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2,
                                  divisor_override=1)
    s1 = sum_pool(inp)
    s2 = sum_pool(mask)
    res = torch.where(s2==0.,0.,s1/s2)
    return res

class GradSolver_VAE(nn.Module):

    def __init__(self, obs_cost, prior_cost, gen_mod, grad_mod, n_step, lr_grad=0.2, smoothing=False, latent = True, **kwargs):
        super().__init__()

        # Need 3 models (obs, gen(anom), solver-J) )
        self.obs_cost = obs_cost
        self.prior_cost = prior_cost
        self.gen_mod = gen_mod
        self.grad_mod = grad_mod
        self.latent = latent
        self.n_step = n_step
        self.lr_grad = lr_grad
        self._grad_norm = None
        self.lambda_obs = torch.nn.Parameter(torch.Tensor([1.]))
        self.lambda_reg = torch.nn.Parameter(torch.Tensor([1.]))
        self.smoothing = smoothing

    def init_state(self, batch, x_init=None):
        if x_init is None:
            x_init = batch.input.nan_to_num().detach().requires_grad_(True)
        coords_cov = torch.stack((batch.latv[:,0].nan_to_num(),
                                  batch.lonv[:,0].nan_to_num(),
                                  batch.land_mask[:,0].nan_to_num(),
                                  batch.topo[:,0].nan_to_num(),
                                  batch.fg_std[:,0].nan_to_num()), dim=1).to(x_init.device)
        x_aug = torch.cat((x_init, coords_cov),dim=1)
        #z = self.gen_mod.encoder(x_aug)[0]
        #x_init = self.gen_mod.decoder(z)
        #x_init = torch.where(torch.isnan(batch.input),x_init,batch.input)
        return (x_init, coords_cov)

    def set_prior(self, batch):
        y = batch.input.nan_to_num().detach().requires_grad_(True)
        coords_cov = torch.stack((batch.latv[:,0].nan_to_num(),
                                  batch.lonv[:,0].nan_to_num(),
                                  batch.land_mask[:,0].nan_to_num(),
                                  batch.topo[:,0].nan_to_num(),
                                  batch.fg_std[:,0].nan_to_num()), dim=1).to(y.device)
        x_aug = torch.cat((y, coords_cov),dim=1)
        z = self.gen_mod.encoder(x_aug)[0]
        x_prior = self.gen_mod.decoder(z)
        return x_prior

    def solver_step(self, state, batch, prior, step):
        # ! in this setup:
        # x is the anom x_HR-x_LR

        # full state
        x, coords_cov = state

        # assimilation is done in latent space
        x_aug = torch.cat((x, state[1]),dim=1)

        if self.latent:
            z = self.gen_mod.encoder(x_aug)[0]
            var_cost = self.lambda_reg**2 * torch.norm(z) + self.lambda_obs**2 * self.obs_cost(x, batch)
            grad = torch.autograd.grad(var_cost, z, create_graph=True)[0]
            gmod = self.grad_mod(grad)
            z_update = ( 1 / (step + 1) * gmod + self.lr_grad * (step + 1) / self.n_step * grad)
            x = self.gen_mod.decoder(z - z_update)
            #x = torch.where(batch.input==0,x,batch.input)
        else:
            #var_cost = self.lambda_reg**2 * torch.norm(x) + self.lambda_obs**2 * self.obs_cost(x, batch)
            var_cost = self.lambda_reg**2 * self.prior_cost(x, prior) + self.lambda_obs**2 * self.obs_cost(x, batch)
            grad = torch.autograd.grad(var_cost, x, create_graph=True)[0]
            gmod = self.grad_mod(grad)
            x_update = ( 1 / (step + 1) * gmod + self.lr_grad * (step + 1) / self.n_step * grad)
            x = x - x_update
            #print(self.lambda_reg, self.lambda_obs)

        state = (x, coords_cov)
        
        return state

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            prior = self.set_prior(batch)

            if self.latent:
                self.grad_mod.reset_state(self.gen_mod.encoder(torch.cat(state,dim=1))[0])
            else:
                self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, prior, step=step)
                if not self.training:
                    state = [s.detach().requires_grad_(True) for s in state]

            state = state[0]
            if self.smoothing:
                state = torch.where(batch.land_mask==1,0.,state)
                mask = (1-(batch.land_mask))
                state = smooth(state,mask)

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

class BaseObsCost(nn.Module):
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w=w

    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return self.w * F.mse_loss(state[msk], batch.input.nan_to_num()[msk])

class BasePriorCost(nn.Module):
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w=w

    def forward(self, state, prior):
        return self.w * F.mse_loss(state, prior)

class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, gen_mod, kernel_size=3, downsamp=None, bilin_quad=True, nt=None):
        super().__init__()
        self.nt = nt
        self.gen_mod = gen_mod
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

    def forward(self, state):
        # done in latent space
        z_state = self.gen_mod.project_latent_space(state)
        phi_z_state = self.gen_mod.project_latent_space(self.forward_ae(state))
        return F.mse_loss(z_state, phi_z_state)

