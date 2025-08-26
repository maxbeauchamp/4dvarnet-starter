import pandas as pd
from pathlib import Path
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
from contrib.CROSCIM.data import *
from dataclasses import dataclass
from collections import Counter
from scipy.interpolate import RegularGridInterpolator

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
            norm_tgt_vars=["asip_sic","cimr_SIT"],
            norm_stats_covs=None,
            *args, **kwargs):

         # optim_weight, srnn_weight, rec_weight are now multi-resolution dictionnaries

         super().__init__(*args, **kwargs)

         self.var_groups = VAR_GROUPS
         self.covariates = COVARIATES
         self.tgt_vars = tgt_vars
         self.norm_tgt_vars = norm_tgt_vars

         self.frcst_lead = frcst_lead
         self.domain_limits = domain_limits
         self.multires = multires
         #self.maxlen_daw = self.trainer.datamodule.test_dataloader()[f"patch_x{self.multires[0]}"].dataset.patch_dims["time"]
         self.maxlen_daw = 15
         n = len(self.multires)
         step = max(1, self.maxlen_daw // n)
         self.len_daw = {
                 r: max(1, self.maxlen_daw - i * step)
                 for i, r in enumerate(self.multires)
         }

         self._norm_stats_cov = norm_stats_covs

         self.optim_weight = {}
         for key, weight_array in optim_weight.items():  # key = "patch_x10", etc.
             buffer_name = f"_optim_weight_{key}"
             weight_tensor = torch.from_numpy(weight_array).to("cuda")
             self.register_buffer(buffer_name, weight_tensor, persistent=persist_rw)
             self.optim_weight[key] = getattr(self, buffer_name)

         self.prior_weight = {}
         for key, weight_array in prior_weight.items():  # key = "patch_x10", etc.
             buffer_name = f"_prior_weight_{key}"
             weight_tensor = torch.from_numpy(weight_array).to("cuda")
             self.register_buffer(buffer_name, weight_tensor, persistent=persist_rw)
             self.prior_weight[key] = getattr(self, buffer_name)

         # Dictionnaire d'équivalences : var canonique → liste d'alias
         self.equivalence_map = {
             "sic": ["sic", "SIC", "sea_ice_concentration"],
             "SIT": ["sit", "SIT", "sea_ice_thickness"]
         }

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
               "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100),
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
                    # Masquage temporel (on suppose dim=1 correspond au temps)
                    if self.frcst_lead is not None and self.frcst_lead > 0:
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
        item_dict = batch._asdict()
        item_dict = self.crop_daw(item_dict, res)

        new_item = {}
        for var in item_dict:
            data = item_dict[var]
            if isinstance(data, torch.Tensor) and data.ndim == 4 and data.shape[1] > 1:
                # Masquage temporel (on suppose dim=1 correspond au temps)
                if self.frcst_lead is not None and self.frcst_lead > 0:
                    data[:, -self.frcst_lead:, :, :] = torch.nan
                new_item[var] = data.to(device)
            else:
                new_item[var] = data  # gardé tel quel (land_mask, latv, lonv...)
        # Reconstruction de l'item
        batch = type(batch)(**new_item)
        return batch

    def format_batch_for_solver(self, batch):
        """
        À partir d'un batch de type TrainingItem, retourne un dictionnaire avec uniquement :
          - 'input' : concaténation des VAR_GROUPS et COVARIATES
          - 'tgt'   : concaténation des variables de tgt_vars
        """
        input_tensors = []
        for group, vars_ in self.var_groups.items():
            for var in vars_:
                key = f"{group}_{var}"
                if hasattr(batch, key):
                    input_tensors.append(getattr(batch, key))
    
        for cov in self.covariates:
            if hasattr(batch, cov):
                input_tensors.append(getattr(batch, cov))
    
        tgt_tensors = []
        for var in self.tgt_vars:
            if hasattr(batch, var):
                tgt_tensors.append(getattr(batch, var))
    
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

            # Extrait la variable canonique (ex: "tgt_sic" → "sic")
            canon_var = pred_var.replace("tgt_", "") if pred_var.startswith("tgt_") else pred_var
            aliases = self.equivalence_map.get(canon_var, [canon_var])

            # Compute the anomaly
            for batch_var in batch_dict:
                for alias in aliases:
                    if batch_var.lower().endswith(alias.lower()):
                        batch_dict[batch_var] = batch_dict[batch_var] - coarse_prediction
                        break  # évite de mettre à jour plusieurs fois le même batch_var
    
        return type(batch)(**batch_dict)

    def interpolate_torch_orig(self,coarse_dict,
                          xc_coarse, yc_coarse,
                          xc_target, yc_target,
                          mode='bilinear',dtype=torch.float32):
        """
        Interpolate dict of (B, T, H, W) tensors on batch-varying regular grids using torch.vmap.
        coarse_dict: dict of {var_name: (B, T, Hc, Wc)}
        xc_coarse, yc_coarse: (B, Wc), (B, Hc)
        xc_target, yc_target: (B, Wf), (B, Hf)
        Returns: dict of {var_name: (B, T, Hf, Wf)}
        """
    
        def make_normalized_grid(xc_c, yc_c, xc_t, yc_t):
            # Build normalized grid in [-1, 1] for grid_sample
            x_min, x_max = xc_c.min(), xc_c.max()
            y_min, y_max = yc_c.min(), yc_c.max()
            grid_x, grid_y = torch.meshgrid(xc_t, yc_t, indexing='xy')  # (Wf, Hf)
            grid_x = grid_x.permute(1, 0).float()
            grid_y = grid_y.permute(1, 0).float()
            # Normalisation basée sur les extrémités d'INDEX (pas min/max)
            # Cela marche aussi si x_c/y_c décroissent (dénominateur < 0)
            x0, x1 = xc_c[0], xc_c[-1]
            y0, y1 = yc_c[0], yc_c[-1]
            # Evite division par zéro si grille dégénérée
            eps = torch.finfo(dctype := dtype).eps
            dx = torch.clamp(x1 - x0, min=-1e-12, max=-1e-12) if (x1-x0).abs() < eps else (x1 - x0)
            dy = torch.clamp(y1 - y0, min=-1e-12, max=-1e-12) if (y1-y0).abs() < eps else (y1 - y0)
            norm_x = 2.0 * (grid_x - x0) / dx - 1.0
            norm_y = 2.0 * (grid_y - y0) / dy - 1.0
            grid = torch.stack((norm_x, norm_y), dim=-1)  # (Wf, Hf, 2)
            grid = torch.clamp(grid, -1.0001, 1.0001)
            #grid = grid.permute(1, 0, 2)  # (Hf, Wf, 2)
            return grid  # (Hf, Wf, 2)
    
        def interpolate_one_sample(xb, grid):
            # xb: (T, Hc, Wc)
            # grid: (Hf, Wf, 2)
            xb = xb.unsqueeze(1)  # (T, 1, Hc, Wc)
            grid = grid.unsqueeze(0).repeat(xb.shape[0], 1, 1, 1)  # (T, Hf, Wf, 2)
            out = F.grid_sample(xb, grid.to(device),
                                mode=mode, align_corners=True)  # (T, 1, Hf, Wf)
            return out.squeeze(1)  # (T, Hf, Wf)
    
        result = {}
        for var, tensor in coarse_dict.items():
            if (tensor is not None) and (var not in ["time","yc","xc"]):
                B = tensor.shape[0]
                grids = []
                for b in range(B):
                    grid = make_normalized_grid(
                        xc_coarse[b], yc_coarse[b],
                        xc_target[b], yc_target[b]
                    )  # (Hf, Wf, 2)
                    grids.append(grid)
                grids = torch.stack(grids, dim=0)  # (B, Hf, Wf, 2)
    
                # vmap interpolation over batch
                out = torch.vmap(interpolate_one_sample)(tensor.to(device),
                                                         grids.to(device))  # (B, T, Hf, Wf)
                result[var] = out
    
        return result
    
    def interpolate_torch(self, coarse_dict, 
                                xc_coarse, yc_coarse, 
                                xc_target, yc_target):
        """
        Interpolate dict of (B, T, Hc, Wc) numpy/tensor arrays onto new target grid (Hf, Wf).
        Uses scipy RegularGridInterpolator with explicit loops over batch and time.
        
        coarse_dict: dict of {var_name: (B, T, Hc, Wc)}
        xc_coarse: (B, Wc) 1D array of x-coords for each batch
        yc_coarse: (B, Hc) 1D array of y-coords for each batch
        xc_target: (B, Wf) 1D array of target x-coords for each batch
        yc_target: (B, Hf) 1D array of target y-coords for each batch
        
        Returns: dict of {var_name: (B, T, Hf, Wf)}
        """
        result = {}
        
        for var, tensor in coarse_dict.items():
            if (tensor is None) or (var in ["time","yc","xc"]):
                continue
            
            # Convert to numpy si tensor est torch.Tensor
            if hasattr(tensor, "detach"):
                tensor = tensor.detach().cpu().numpy()
            
            T, Hc, Wc = tensor.shape[1:]
            B = yc_target.shape[0]
            Hf, Wf = yc_target.shape[1], xc_target.shape[1]
            
            out = np.zeros((B, T, Hf, Wf), dtype=np.float32)
            
            for b in range(B):
                # build interpolator for each time step
                x_c = xc_coarse[b].cpu().numpy() if hasattr(xc_coarse[b], "cpu") else xc_coarse[b]
                y_c = yc_coarse[b].cpu().numpy() if hasattr(yc_coarse[b], "cpu") else yc_coarse[b]
                X_t, Y_t = np.meshgrid(xc_target[b].cpu().numpy(), yc_target[b].cpu().numpy(), indexing="xy")
                target_points = np.stack([Y_t.ravel(), X_t.ravel()], axis=-1)  # (Hf*Wf, 2)
                
                for t in range(T):
                    f_interp = RegularGridInterpolator(
                        (y_c, x_c),  # ordre (yc, xc)
                        tensor[b, t], 
                        bounds_error=False, fill_value=np.nan
                    )
                    interp_vals = f_interp(target_points).reshape(Hf, Wf)
                    out[b, t] = interp_vals
            
            result[var] = torch.tensor(out).to(device)
        
        return result

    def split_tensor_to_dict(self, tensor):
        """
        Découpe un tenseur interpolé (B, C, H, W) en dictionnaire {var: (B, T, H, W)}.
        Args:
            tensor: torch.Tensor de shape (B, C=T*V, H, W)
        Returns:
            dict {var_name: tensor de shape (B, T, H, W)}
        """
        B, C, H, W = tensor.shape
        V = len(self.tgt_vars)
        time_steps = C//V
        assert C == time_steps * V, f"Expected C={time_steps}×{V}, but got {C}"
        tensor_reshaped = tensor.view(B, V, time_steps, H, W)  # (B, V, T, H, W)
        tensor_reshaped = tensor_reshaped.permute(0, 2, 1, 3, 4)  # (B, T, V, H, W)
        out_dict = {
            var: tensor_reshaped[:, :, i]  # (B, T, H, W)
            for i, var in enumerate(self.tgt_vars)
        }
        return out_dict

    def training_step(self, batch, batch_idx):
        return self.multistep(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.multistep(batch, "val")[0]

    def forward(self, batch, res=1):
        model = self.solver.solvers[f"solver_x{res}"].to(device)
        return model(batch)

    def on_epoch_start(self):
        epoch = self.current_epoch
        res_idx = min(epoch // (self.trainer.max_epochs // len(self.multires)), len(self.multires) - 1)
        train_res = self.multires[res_idx]

        for res in self.multires:
            model = self.solver.solvers[f"solver_x{res}"].to(device)
            if res == train_res:
                model.train()
                for p in model.parameters():
                    p.requires_grad = True
            else:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
        if self.global_rank == 0:
            print(f"[Epoch {epoch}] Training resolution: {train_res}")

    def multistep(self, batch, phase=""):

        batch = self.modify_multires_batch(batch)
        out = {}

        epoch = self.current_epoch
        n_res = len(self.multires)
        total_epochs = self.trainer.max_epochs
        steps_per_res = max(1,total_epochs // n_res)
        res_index = min(epoch // steps_per_res, n_res - 1)  # limit to the last resolution

        train_res = self.multires[res_index]
        print(f"epoch_{epoch}, training resolution {res_index}")
        total_loss = 0.
        for i, res in enumerate(self.multires):
            batch_res = batch[f"patch_x{res}"]
            if (res==self.multires[0]):
                if res==train_res:
                    loss, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=phase)
                    total_loss += loss
                else:
                    # inference only if not training this resolution
                    with torch.no_grad():
                        _, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=phase)
            else:
                coarser_res = self.multires[i-1]
                # project coarser_res batch on res batch
                xc_target = batch_res.xc
                yc_target = batch_res.yc
                xc_coarse = batch[f"patch_x{coarser_res}"].xc
                yc_coarse = batch[f"patch_x{coarser_res}"].yc
                out[f"patch_x{coarser_res}_on_x{res}"] = self.interpolate_torch(out[f"patch_x{coarser_res}"],
                                                                                xc_coarse, yc_coarse,
                                                                                xc_target, yc_target)
                out[f"patch_x{coarser_res}_on_x{res}"] = self.crop_daw(out[f"patch_x{coarser_res}_on_x{res}"], res)
                # modify batch to work on anomaly compared to coarser resolution
                batch_res = self.update_batch_as_anomaly(batch_res, out[f"patch_x{coarser_res}_on_x{res}"])
                if res==train_res:
                    loss, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=phase)
                    # sum out[f"patch_x{coarser_res}"] and  out[f"patch_x{res}"]
                    total_loss+=loss
                else:
                    # inference only if not training this resolution
                    with torch.no_grad():
                        _, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=phase)
        return loss, out

    def step(self, batch, res, phase=""):

        loss, out = self.base_step(batch, res=res, phase=phase)
        res_key = f"patch_x{res}"
    
        total_grad_loss = 0.0
        total_srnn_loss = 0.0
    
        for var_name in self.tgt_vars:
            if not hasattr(batch, var_name):
                raise ValueError(f"Batch missing variable: {var_name}")
    
            target = getattr(batch, var_name)
            pred = out[var_name]
    
            tgt_sobel = kfilts.sobel(target)
            pred_sobel = kfilts.sobel(pred)
    
            mask = tgt_sobel.isfinite()
    
            grad_loss = self.weighted_mse(
                torch.where(mask, pred_sobel, torch.tensor(float('nan'), device=pred.device)) - tgt_sobel,
                self.optim_weight[res_key]
            )
            total_grad_loss += grad_loss
    
        # Prior / SRNN loss
        if hasattr(self.solver.solvers[f"solver_x{res}"], "prior_cost"):
            sbatch = self.format_batch_for_solver(batch)
            model = self.solver.solvers[f"solver_x{res}"].to(device)
            prior = model.prior_cost.forward_ae(sbatch.input)
            #total_prior_loss = self.weighted_mse(sbatch.tgt-prior,
            #                                    self.prior_weight[res_key])
            total_prior_loss = 0.0
        else:
            total_prior_loss = 0.0

        self.log(f"{phase}_gloss", total_grad_loss, prog_bar=True, on_step=False, on_epoch=True)
    
        training_loss = 50 * loss + 1000 * total_grad_loss + 10 * total_prior_loss
        print(50 * loss, 10000 * total_grad_loss, 10 * total_prior_loss)
    
        return training_loss, out

    def base_step(self, batch, res, phase=""):
        """
        Compute loss over selected target variables in a multi-variate model.
        Args:
            batch: a NamedTuple with target fields matching tgt_vars.
            phase: string for logging ("train", "val", etc.)
        Returns:
           loss: total loss
           out: model output tensor
        """

        sbatch = self.format_batch_for_solver(batch)

        out = self(batch=sbatch, res=res)  # out is a tensor 
        out = self.split_tensor_to_dict(out)
        res_key = f"patch_x{res}"

        total_loss = 0.0
        for i, var_name in enumerate(self.tgt_vars):
            if not hasattr(batch, var_name):
                raise ValueError(f"Batch does not contain variable '{var_name}'")
            target = getattr(batch, var_name)  # (B, T, Y, X)
            pred = out[var_name]  # (B, T, Y, X)
            mask = target.isfinite() 
            loss = self.weighted_mse(torch.where(mask, pred, 
                                                torch.tensor(float('nan'), device=pred.device)) - target,
                                                self.optim_weight[res_key])
            total_loss += loss

        with torch.no_grad():
            self.log(f"{phase}_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return total_loss, out

    def reconstruct(self, dl, items, daw, time, weight=None):
        """
        takes as input a list of tensor of dimensions (V, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims
        items: list of torch tensor corresponding to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting 
        """

        if weight is None:
            weight = np.ones(list(dl.dataset.patch_dims.values()))
        weight = torch.tensor(weight)

        nvars = items[0].shape[0]

        result_tensor = torch.full((nvars, 1, dl.dataset.da_dims['yc'], dl.dataset.da_dims['xc']),
                                   float('nan'))
        count_tensor = torch.zeros((nvars, 1, dl.dataset.da_dims['yc'], dl.dataset.da_dims['xc']))

        coords = dl.dataset.get_coords()[(daw*len(items)):((daw+1)*len(items))]

        for idx, item in enumerate(items):
            c = coords[idx]
            iy = [np.where(dl.dataset.yc == y)[0][0] for y in c.yc.values]
            ix = [np.where(dl.dataset.xc == x)[0][0] for x in c.xc.values]
            result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] = torch.where(torch.isnan(result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]),
                                                                              0.,
                                                                              result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1])
            result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += torch.squeeze(item * weight)
            count_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += weight

        result_tensor /= np.maximum(count_tensor, 1e-6)
        result_da = xr.DataArray(
            result_tensor,
            dims=[f'v{i}' for i in range(nvars - len(coords[0].dims))] + ["time", "yc", "xc"],
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
                                     use_datamodule=False):

        dl = self.trainer.test_dataloaders[self.dataloader_keys[dataloader_idx]]
        
        res = self.multires[dataloader_idx]
        last = self.len_daw[res]
        res_key = f"patch_x{res}"

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
                            self.rec_weight[res_key].cpu().numpy()[[i],:,:]
                    )
            test_data_ldt = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')
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
                          use_datamodule=False):

        res = self.multires[dataloader_idx]

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
            group, var = varname.split("_")
            stats = self.norm_stats[group][var]
            if stats["type"] == "zscore":
                return data * stats["std"] + stats["mean"]
            elif stats["type"] == "minmax":
                return data * (stats["max"] - stats["min"]) + stats["min"]
            else:
                raise ValueError(f"Unknown normalization type for {varname}")

        for idx_daw in torch.unique(daws):
            sel_daw = torch.where(daws==idx_daw)[0]
            test_data_sel = [test_data[i] for i in sel_daw.tolist()]
            test_data_uniq = self.aggregate_batches_one_domain(idx_daw, idx_rec,
                                                               test_data_sel,
                                                               dataloader_idx,
                                                               use_datamodule)
            # prepare unnormalization for metrics and storage
            test_data_unnorm = test_data_uniq.copy(deep=False)
            for i, var in enumerate(self.tgt_vars):
                norm_var = self.norm_tgt_vars[i]
                _, var = norm_var.split("_")
                test_data_unnorm = test_data_unnorm.update({f"pred_{var}" : (("time","yc","xc"),
                                                             unnormalize(norm_var, test_data_uniq[f"pred_{var}"].data))})
                test_data_unnorm = test_data_unnorm.update({f"tgt_{var}" : (("time","yc","xc"),
                                                             unnormalize(norm_var, test_data_uniq[f"tgt_{var}"].data))})
            if metrics:
                metric_data = test_data_unnorm.pipe(self.pre_metric_fn),
                metrics = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                print(metrics.to_frame(name="Metrics").to_markdown())
            # save NetCDFs
            time = [ datetime.datetime.strptime(str(t)[:10], "%Y-%m-%d").strftime("%Y%m%d") for t in test_data_unnorm.time.data ]
            file = f'test_data_{time[0]}_{time[-1]}_patch_x{res}.nc'
            if self.logger and write_netcdf:
                 test_data_unnorm.to_netcdf(Path(self.logger.log_dir) / file)
                 print(Path(self.trainer.log_dir) / file)
                 if metrics:
                     self.logger.log_metrics(metrics.to_dict())
            # stack each daw
            netcdf_final.append(test_data_uniq)

        # merge all time steps in a dictionary
        return { f"daw_{i}": nc for i, nc in enumerate(netcdf_final) }
        #return xr.concat(netcdf_final, dim="daw").assign_coords(daw=torch.unique(daws)).sortby("daw")

    def convert_xr_to_batch(self, coarse, batch, spatial_sel=False):
        """
        Convert an xarray.Dataset (coarse) to a dictionary of PyTorch tensors,
        matching the batch's temporal indices.
        Args:
            coarse (xr.Dataset): xarray with dims (time, yc, xc)
            batch (dict): Dictionary with keys 'time', 'yc', 'xc' etc., 
                          values are tensors of shape (B, T, H, W)
        Returns:
            coarse_dict: dict with same keys as coarse.data_vars, each of shape (B, T, H, W)
        """
        times = batch.time.cpu().numpy().astype('datetime64[s]').astype('datetime64[ns]')
        times = times.astype('datetime64[D]').astype('datetime64[ns]')
        nbatch = len(batch.time)  # batch.time: shape (B, T)
        T, H, W = batch.time.shape[1], batch.yc.shape[1], batch.xc.shape[1]
        coarse_dict = {}
        # Pour chaque variable du Dataset
        for var in self.tgt_vars + ["time", "yc", "xc"]:
            B_array = []
            for i in range(nbatch):
                times_i = np.squeeze(times[i])  # (T,)
                xcs_i = batch.xc[i].cpu().numpy()      # (W,)
                ycs_i = batch.yc[i].cpu().numpy()      # (H,)
                # temporal selection
                matching_key = [
                                key
                                for key, ds in coarse.items()
                                if set(ds.time.values) == set(times_i)
                                ][0]
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
                # Convertir en numpy
                arr = sel_patch[var].values  # (T, H, W)
                if var == "time":
                    arr = arr.astype('datetime64[ns]').astype('int64')
                if var in ["time", "yc", "xc"]:
                    arr = np.expand_dims(arr,axis=0)
                B_array.append(torch.from_numpy(arr).float())
            # Stack B (B, T, H, W)
            coarse_dict[var] = torch.stack(B_array, dim=0)
            # Récupérer tous les champs
            fields = batch._fields
            # Construire un nouveau dict avec les valeurs de coarse_dict
            # ou None par défaut si clé manquante
            complete_dict = {field: coarse_dict.get(field, None) for field in fields}
        return  type(batch)(**complete_dict)

    def on_test_start(self):
        # Stocker les dataloader keys dans l'ordre des indices
        self.dataloader_keys = list(self.trainer.test_dataloaders.keys())
    
        # Calculer le nombre de batchs par dataloader
        self.num_test_batches = {
            i: len(dl)
            for i, dl in enumerate(self.trainer.test_dataloaders.values())
        }

    def is_last_batch(self, batch_idx, dataloader_idx):
        total_batches = self.num_test_batches[dataloader_idx]
        return batch_idx == total_batches - 1
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):

        res = self.multires[dataloader_idx]
        res_key = f"patch_x{res}"
        last = self.len_daw[res]

        print(f"Dataloader_{dataloader_idx}, Batch_{batch_idx}, res_{res}")
        if (dataloader_idx == 0) and (batch_idx == 0) :
            self.test_data = {}
            self.test_times = {}
            self.aggregate_results = {}

        if batch_idx == 0:
            self.test_data[res_key] = []
            self.test_times[res_key] = []
            
        batch = self.modify_batch(batch, res)
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
            coarse = self.convert_xr_to_batch(coarse, batch)
            xc_coarse = torch.squeeze(coarse.xc, dim=1)
            yc_coarse = torch.squeeze(coarse.yc, dim=1)
            itrp_coarse = self.interpolate_torch(coarse._asdict(),
                                                 xc_coarse, yc_coarse,
                                                 xc_target, yc_target)
            #itrp_coarse = self.crop_daw(itrp_coarse,res)
            # modify batch to work on anomaly compared to coarser resolution
            batch = self.update_batch_as_anomaly(batch, itrp_coarse)

        sbatch = self.format_batch_for_solver(batch)
        out = self(batch=sbatch, res=res)
        out = self.split_tensor_to_dict(out)
        # add coarser resolution to output
        if dataloader_idx > 0:
            out = {k: out[k] + itrp_coarse[k] for k in out}
            #out = {k: itrp_coarse[k] for k in out}
        for i, var in enumerate(self.tgt_vars):
            out[var] = torch.where(batch.land_mask==1.,np.nan,out[var])

        # Stockage des sorties et des cibles
        # Unnormalization is done in aggregate
        out_norm, tgt_norm = {}, {}
        for i, var in enumerate(self.tgt_vars):
            pred = out[var] 
            out_norm[var] = pred
            if dataloader_idx == 0:
                tgt_norm[var] = getattr(batch, var)
            else:
                tgt_norm[var] = getattr(batch, var) + itrp_coarse[var]
        
        combined = list(out_norm.values()) + list(tgt_norm.values())
        stacked = torch.stack(combined, dim=1)

        # stacked has shape (B,V,T,H,W) with V the number of variables
        self.test_data[res_key].append(stacked)
        self.test_times[res_key].append(torch.squeeze(batch.time, dim=1))

        # if last batch, agreggate (as an xarray dataset with the estimation for a given resolution)
        if self.is_last_batch(batch_idx, dataloader_idx):
            self.test_data[res_key] = list(itertools.chain(*self.test_data[res_key]))
            self.test_times[res_key] = list(itertools.chain(*self.test_times[res_key]))
            if dataloader_idx == (len(self.multires)-1):
                #idx_rec = np.arange(batch.time.shape[-1]-self.frcst_lead+1,
                #                    batch.time.shape[-1])
                idx_rec = np.arange(batch.time.shape[-1])
                write_netcdf = True
            else:
                idx_rec = np.arange(batch.time.shape[-1])
                write_netcdf = True
            # the idea behind: the aggregation is :
            # on the full window for coarser resolutions
            # only for nowcast/forecast lead times for final resolution
            self.aggregate_results[res_key] = self.aggregate_batches(idx_rec,
                                                                     self.test_data[res_key],
                                                                     self.test_times[res_key],
                                                                     dataloader_idx,
                                                                     metrics=False,
                                                                     write_netcdf=write_netcdf)
            print(self.aggregate_results[res_key])

        batch, out = None, None

    @property
    def test_quantities(self):
        return [prefix + var.split("_", 1)[-1] for prefix in ["pred_", "tgt_"] for var in self.tgt_vars]

    def on_test_epoch_end(self):
        print("hello")

    def on_load_checkpoint(self, checkpoint):
        """
        very useful when shapes of the patches/weights between
        training and inference
        """
        for key in self.state_dict().keys():
            if key.startswith("rec_weight") or key.startswith("optim_weight") or key.startswith("prior_weight"):
                print(key)
                checkpoint["state_dict"][key] = self.state_dict()[key]
