import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from src.utils import get_last_time_wei, get_linear_time_wei
from src.models import Lit4dVarNet

class Lit4dVarNet_ASIP_OSISAF(Lit4dVarNet):

    def __init__(self, path_mask, optim_weight, sr_weight, domain_limits, persist_rw=True, *args, **kwargs):
         super().__init__(*args, **kwargs)

         self.domain_limits = domain_limits
         self.mask_land = np.isfinite(xr.open_dataset(path_mask).sel(**(self.domain_limits or {})).sic[0])

         self.register_buffer('optim_weight', torch.from_numpy(optim_weight), persistent=persist_rw)
         self.register_buffer('sr_weight', torch.from_numpy(sr_weight), persistent=persist_rw)

    def modify_batch(self,batch):
        #batch = batch._replace(input=batch.input.nan_to_num())
        #batch = batch._replace(tgt=batch.tgt.nan_to_num())
        return batch

    def remove_useless_patches(self,batch):
        def nanvar(tensor, dim=None, keepdim=False):
            tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
            output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
            return output
        idx = []
        for i in range(len(batch.tgt)):
            # keep the patch if not full of NaN or not full of ice/water (var=0)
            if ( (batch.tgt[i].isfinite().float().mean() != 0) and (nanvar(batch.coarse[i])!=0) ):
                idx.append(i)
        if len(idx)>0:
            batch = batch._replace(input=batch.input[idx])
            batch = batch._replace(tgt=batch.tgt[idx])
            batch = batch._replace(coarse=batch.coarse[idx])
        else:
            batch = None
        return batch

    def step(self, batch, phase=""):

        batch = self.remove_useless_patches(batch)
        if batch is None:
            return None, None

        if self.training and batch.coarse.isfinite().float().mean() < 0.05:
            return None, None

        #batch = self.modify_batch(batch)
        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(kfilts.sobel(out)-kfilts.sobel(batch.tgt),
                                      self.optim_weight)
        srnn_loss = self.weighted_mse(batch.tgt-self.solver.prior_cost.forward_ae(batch.coarse.nan_to_num()),
                                      self.sr_weight)

        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 10 * srnn_loss
        print(50*loss, 10000 * grad_loss, 10 * srnn_loss)
        return training_loss, out

    def base_step(self, batch, phase=""):

        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.optim_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []

        #batch = self.modify_batch(batch)
        out = self(batch=batch)
        m, s = self.norm_stats
        self.test_data.append(torch.stack(
            [
                #(batch.input*s+m).cpu(),
                #(batch.tgt*s+m).cpu(),
                (out*(s-m)+m).squeeze(dim=-1).detach()#.cpu(),
                #(out*s+m).squeeze(dim=-1).detach()#.cpu(),
            ],
            dim=1,
        ))

        #out = (out*s+m).squeeze(dim=-1).detach()
        #if batch_idx == 0:
        #    self.test_data = torch.stack([out],dim=1)
        #else:
        #    self.test_data = torch.cat((self.test_data,torch.stack([out],dim=1)))

        batch = None
        out = None

    @property
    def test_quantities(self):
        #return ['inp', 'tgt', 'out']
        return ['out']

    def on_test_epoch_end(self):

        #self.test_data = torch.cat(self.test_data).cuda()

        if isinstance(self.trainer.test_dataloaders,list):
            rec_da = self.trainer.test_dataloaders[0].dataset.reconstruct(
                self.test_data, self.rec_weight.cpu().numpy()
            )
        else:
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_data, self.rec_weight.cpu().numpy()
            )

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        # crop (if necessary) 
        self.test_data = self.test_data.sel(**(self.domain_limits or {}))
        self.test_data = self.test_data.update({#'inp':(('time','yc','xc'),self.test_data.inp.data),
                                                #'tgt':(('time','yc','xc'),self.test_data.tgt.data),
                                                'sic':(('time','yc','xc'),self.test_data.out.data)})
        """
        if self.mask_land is not None:
             self.mask_land = self.mask_land.sel(**(self.domain_limits or {}))
             self.test_data.coords['mask'] = (('yc', 'xc'), self.mask_land.values)
             self.test_data = self.test_data.where(self.test_data.mask)
        
        metric_data = self.test_data.pipe(self.pre_metric_fn),
        metrics = pd.Series({
            metric_n: metric_fn(metric_data)
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        """
        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            #self.logger.log_metrics(metrics.to_dict())

