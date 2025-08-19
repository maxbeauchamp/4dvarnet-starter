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

def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # set to eval mode
    return model

class Lit4dVarNet_ASIP_OSISAF(Lit4dVarNet):

    def __init__(self,
                 optim_weight,
                 sr_weight,
                 domain_limits,
                 persist_rw=True, 
                 frcst_lead=0,
                 training_mode="join", 
                 srnn_training_mode="from_osisaf",
                 use_cov=True, *args, **kwargs):

         super().__init__(*args, **kwargs)
         # remove the weighting to enable modifications during inference
         # self.save_hyperparameters(ignore=["rec_weight","optim_weight","sr_weight"]) 

         if training_mode=="srnn_only":
             freeze_model(self.solver.grad_mod)
         if training_mode=="solver_only":
             freeze_model(self.solver.prior_cost)

         self.frcst_lead = frcst_lead
         self.use_cov = use_cov
         self.domain_limits = domain_limits
         self.srnn_training_mode = srnn_training_mode
         self.register_buffer('optim_weight', torch.from_numpy(optim_weight), persistent=persist_rw)
         self.register_buffer('sr_weight', torch.from_numpy(sr_weight), persistent=persist_rw)

    def configure_optimizers(self):
        if self.opt_fn is not None:
            return self.opt_fn(self)
        else:
            opt = torch.optim.Adam(
            [
                {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
                {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
                {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
            ], weight_decay=1e-5
            )
            return {
               "optimizer": opt,
               "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100),
            }

    def modify_batch(self,batch):
        batch_ = batch
        new_input = batch_.input
        new_coarse = batch_.coarse
        device = batch_.input.device
        if (self.frcst_lead is not None) and (self.frcst_lead>0):
            new_input[:,(-self.frcst_lead):,:,:] = np.nan
            new_coarse[:,(-self.frcst_lead):,:,:] = np.nan
        batch_ = batch_._replace(input=new_input.to(device))
        batch_ = batch_._replace(coarse=new_coarse.to(device))
        return batch_

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
            batch = batch._replace(land_mask=batch.land_mask[idx])
            if self.use_cov:
                batch = batch._replace(lonv=batch.lonv[idx])
                batch = batch._replace(latv=batch.latv[idx])
                batch = batch._replace(t2m=batch.t2m[idx])
                batch = batch._replace(istl1=batch.istl1[idx])
                batch = batch._replace(sst=batch.sst[idx])
                batch = batch._replace(skt=batch.skt[idx])
        else:
            batch = None
        return batch

    def step(self, batch, phase=""):

        batch = self.modify_batch(batch)

        batch = self.remove_useless_patches(batch)
        if batch is None:
            return None, None

        if self.training and batch.coarse.isfinite().float().mean() < 0.25:
            return None, None

        loss, out = self.base_step(batch, phase)
        #grad_loss = self.weighted_mse(kfilts.sobel(out)-kfilts.sobel(batch.tgt),
        #                              self.optim_weight)
        # new grad_loss
        mask = kfilts.sobel(batch.tgt).isfinite() * (~kfilts.sobel(batch.input).isfinite())
        grad_loss1 = self.weighted_mse(torch.where(mask,kfilts.sobel(out),np.nan) - kfilts.sobel(batch.tgt),
                                       self.optim_weight)
        mask = kfilts.sobel(batch.tgt).isfinite() * kfilts.sobel(batch.input).isfinite()
        grad_loss2 = self.weighted_mse(torch.where(mask,kfilts.sobel(out),np.nan) - kfilts.sobel(batch.tgt),
                                       self.optim_weight)
        grad_loss = grad_loss1 + grad_loss2

        # super-resolution loss
        if not self.use_cov:
            srnn = self.solver.prior_cost.forward_ae(batch.coarse.nan_to_num())
        else:
            if self.srnn_training_mode=="from_osisaf": 
                srnn = self.solver.prior_cost.forward_ae((torch.cat([
                                                                batch.coarse.nan_to_num(),
                                                                #batch.lonv.nan_to_num()[:,[0],:,:],
                                                                #batch.latv.nan_to_num()[:,[0],:,:],
                                                                #batch.land_mask.nan_to_num()[:,[0],:,:],
                                                                batch.t2m.nan_to_num(),
                                                                batch.istl1.nan_to_num(),
                                                                batch.sst.nan_to_num(),
                                                                batch.skt.nan_to_num()],dim=1)))
            else:
                srnn = self.solver.prior_cost.forward_ae((torch.cat([
                                                                F.avg_pool2d(batch.tgt.nan_to_num(),
                                                                             kernel_size=21,
                                                                             stride=1,
                                                                             padding=21//2),
                                                                #batch.lonv.nan_to_num()[:,[0],:,:],
                                                                #batch.latv.nan_to_num()[:,[0],:,:],
                                                                #batch.land_mask.nan_to_num()[:,[0],:,:],
                                                                batch.t2m.nan_to_num(),
                                                                batch.istl1.nan_to_num(),
                                                                batch.sst.nan_to_num(),
                                                                batch.skt.nan_to_num()],dim=1)))
        srnn_loss = self.weighted_mse(batch.tgt-srnn,self.sr_weight)

        # prior regularization loss
        nb, nt, ny, nx = batch.tgt.shape
        # create kernel
        sigma = 5
        x = np.arange(0,15)
        y = np.arange(0,15)
        t = np.arange(0,5)
        tt, yy, xx = np.meshgrid(t,y,x)
        tt = np.transpose(tt,(1,0,2))
        yy = np.transpose(yy,(1,0,2))
        xx = np.transpose(xx,(1,0,2))
        kernel = np.exp(-(np.abs(xx-(len(x)//2))**2 + np.abs(yy-(len(y)//2))**2 + np.abs(tt-(len(t)//2))**2)/(2*sigma**2))
        kt, ky, kx = kernel.shape
        krnl = torch.tensor(kernel).reshape((1,1,kt,ky,kx)).to(out.device)
        mask = torch.squeeze(F.conv3d(torch.unsqueeze(batch.tgt.isfinite().float(),dim=1), krnl.float(), padding="same"),dim=1)      
        mask = torch.where(mask<0.01,0,1).bool()
        prior_loss = self.weighted_mse(torch.where(mask,out,np.nan) - srnn, self.optim_weight)

        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 10 * srnn_loss + 10 * prior_loss
        print(50*loss, 10000 * grad_loss, 10 * srnn_loss, 10 * prior_loss)
        return training_loss, out

    def base_step(self, batch, phase=""):

        out, sr= self(batch=batch)
        #loss = self.weighted_mse(out - batch.tgt, self.optim_weight)
        # new_loss
        mask = batch.tgt.isfinite() * (~batch.input.isfinite())
        loss1 = self.weighted_mse(torch.where(mask,out,np.nan) - batch.tgt, self.optim_weight)
        mask = batch.tgt.isfinite() * batch.input.isfinite()
        loss2 = self.weighted_mse(torch.where(mask,out,np.nan) - batch.tgt, self.optim_weight)
        loss = loss1 + loss2

        with torch.no_grad():
            self.log(f"{phase}_mse", 10 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []

        coarse_orig = batch.coarse.clone().detach()
        batch = self.modify_batch(batch)

        out, sr = self(batch=batch)
        out = torch.where(batch.land_mask==1.,np.nan,out)
        sr = torch.where(batch.land_mask==1.,np.nan,sr)
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                #(batch.input*s+m).cpu(),
                (batch.tgt*(s-m)+m)[:,-(self.frcst_lead+1):,:,:],#.cpu(),
                (coarse_orig*(s-m)+m)[:,-(self.frcst_lead+1):,:,:],#.cpu(),
                (out*(s-m)+m).squeeze(dim=-1).detach()[:,-(self.frcst_lead+1):,:,:],#.cpu(),
                (sr*(s-m)+m).squeeze(dim=-1).detach()[:,-(self.frcst_lead+1):,:,:]#.cpu(),
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
        sr = None

    @property
    def test_quantities(self):
        #return ['out']
        #return ['inp', 'tgt', 'out']
        return ['tgt', 'coarse', 'out', 'sr']

    def on_test_epoch_end(self):

        #self.test_data = torch.cat(self.test_data).cuda()

        for i in np.arange(self.frcst_lead+1):
            print("Reconstructing LEADTIME "+str(i))
            if isinstance(self.trainer.test_dataloaders,list):
                rec_da = self.trainer.test_dataloaders[0].dataset.reconstruct(
                        [ self.test_data[j][:,:,[i],:,:] for j in range(len(self.test_data)) ],
                        -(self.frcst_lead-i+1),
                        self.rec_weight.cpu().numpy()[[-(self.frcst_lead-i+1)],:,:]
                )
            else:
                rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                        [ self.test_data[j][:,:,[i],:,:] for j in range(len(self.test_data)) ],
                        -(self.frcst_lead-i+1),
                        self.rec_weight.cpu().numpy()[[-(self.frcst_lead-i+1)],:,:]
                )

            test_data_ldt = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')

            # crop (if necessary) 
            test_data_ldt = test_data_ldt.sel(**(self.domain_limits or {}))
        
            """
            metric_data = test_data_ldt.pipe(self.pre_metric_fn),
            metrics = pd.Series({
                metric_n: metric_fn(metric_data)
                for metric_n, metric_fn in self.metrics.items()
            })
            print(metrics.to_frame(name="Metrics").to_markdown())
            """
            time = datetime.strptime(str(test_data_ldt.time.data[0])[:10], "%Y-%m-%d").strftime("%Y%m%d")  
            if i==0:
                init_time = time
            file = 'test_data_'+init_time+'_'+time+'.nc'
            if self.logger:
                 test_data_ldt.to_netcdf(Path(self.logger.log_dir) / file)
                 print(Path(self.trainer.log_dir) / file)
                 #self.logger.log_metrics(metrics.to_dict())

    def on_load_checkpoint(self, checkpoint):
        """
        very useful whn shapes of the patches/weights between
        training and inference
        """
        for key in self.state_dict().keys():
            if key.startswith("rec_weight") or key.startswith("optim_weight") or key.startswith("sr_weight"):
                print(key)
                checkpoint["state_dict"][key] = self.state_dict()[key]
