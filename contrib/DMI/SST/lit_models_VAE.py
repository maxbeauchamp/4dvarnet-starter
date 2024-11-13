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
from src.models import Lit4dVarNet

class Lit4dVarNet_VAE(Lit4dVarNet):
    def __init__(self, path_mask, optim_weight, domain_limits, persist_rw=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_limits = domain_limits
        self.mask_land = np.isfinite(xr.open_dataset(path_mask).sel(**(self.domain_limits or {})).analysed_sst_LR[20])
        self.register_buffer('optim_weight', torch.from_numpy(optim_weight), persistent=persist_rw)

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)

    @staticmethod
    def weighted_mse(err, weight):
        err_w = err * weight[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def forward(self, batch):
        return self.solver(batch)
    
    def modify_batch(self,batch):
        batch = batch._replace(input=batch.input.nan_to_num())
        batch = batch._replace(tgt=batch.tgt.nan_to_num())
        return batch

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.25:
            return None, None
 
        #batch = self.modify_batch(batch)

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        # classic 4DVarNet loss (MSE loss for solver)
        training_loss = 50 * loss + 1000 * grad_loss
        #print(50 * loss, 1000 * grad_loss)
        # VAE loss (supervised)
        toto=False
        if toto:
        #if self.opt_fn.train_vae:
            covs = self.solver.init_state(batch)[1]
            z, mean, log_var = self.solver.gen_mod.encoder(torch.cat((batch.tgt,covs),dim=1))
            x_hat = self.solver.gen_mod.decoder(z)
            loss_vae, rec, KL = self.solver.gen_mod.vae_loss(x_hat, batch.tgt, mean, log_var, wKL=0.001)
            # total loss
            #print(training_loss, rec, KL)
            training_loss += loss_vae

        return training_loss, out

    def base_step(self, batch, phase=""):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 50 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []

        batch = self.modify_batch(batch)
        out = self(batch=batch)

        covs = self.solver.init_state(batch)[1]
        z, mean, log_var = self.solver.gen_mod.encoder(torch.cat((batch.tgt,covs),dim=1))
        x_hat = self.solver.gen_mod.decoder(z)

        """
        # plot
        data = xr.Dataset(data_vars=dict(
                               out=(["time", "lat", "lon"], x_hat[0].detach().cpu().data),
                               ),
                          coords=(dict(time=range(7),
                                  lon=range(240),
                                  lat=range(240))))
        data.out.plot(col='time',col_wrap=7)
        plt.show()
        """

        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                batch.input.cpu() * s + m + batch.coarse.cpu(),
                batch.tgt.cpu() * s + m + batch.coarse.cpu(),
                out.squeeze(dim=-1).detach().cpu() * s + m + batch.coarse.cpu(),
                x_hat.cpu() * s + m
            ],
            dim=1,
        )) 

        batch = None
        out = None

    @property
    def test_quantities(self):
        return ['inp', 'tgt', 'out', 'vae']

    def on_test_epoch_end(self):

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

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data) 
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            self.logger.log_metrics(metrics.to_dict())
