import sys
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
print(os.getcwd())
sys.path.append('../../..')
from contrib.DMI.ASIP_OSISAF.data import *
from contrib.DMI.ASIP_OSISAF.load_data import *
from src.utils import *
from src.models import *

import matplotlib.pyplot as plt
import torch
import itertools
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature

datamodule = BaseDataModule(asip_paths=load_data(),
                            osisaf_paths=load_data(type="osisaf"),
                            covariates_paths=load_data("era5"),
                            covariates=["t2m", "istl1", "siconc", "sst", "skt"],
                            mask_path="/dmidata/users/maxb/4dvarnet-starter/contrib/DMI/ASIP_OSISAF/mask_PanArctic.nc",
                            domain_name="arctic_small",
                            domains={'train': {'time': slice('2021-01-01', '2021-12-31',)},
                                     'val': {'time': [ slice('2021-01-01', '2021-06-30',),
                                                       slice('2021-07-01', '2021-12-31',) ]},
                                     'test': {'time': slice('2022-01-01', '2022-01-20',)}},
                            xrds_kw={'patch_dims': {'time': 15, 'yc': 240, 'xc': 240},
                                     'strides': {'time': 1, 'yc': 20, 'xc': 20},
                                     'strides_test': {'time': 1, 'yc': 200, 'xc': 200},
                                     'domain_limits':  dict(xc=slice(-3849750.,3749750.,),
                                                            yc=slice(3849750.,-3349750.,))
                                     },
                            dl_kw={'batch_size': 2, 'num_workers': 20},
                            res=500,
                            pads=[False,False,True],
                            norm_stats =[0,100],
                            norm_stats_covs = [ {'t2m': 270.08, 'istl1': 267.68, 'siconc': 0, 'sst': 276.97, 'skt': 270.50},
                                               {'t2m': 14.67, 'istl1': 7.80, 'siconc': 1, 'sst': 6.82, 'skt': 15.21}
                                             ])
datamodule.setup()

data_loader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()
test_dataloader = datamodule.test_dataloader()

def save_batch_as_NetCDF(batch, ibatch, keep_obs=False):
    
    data = xr.Dataset(data_vars={
                      'asip':(('sample','time','yc','xc'),batch.tgt.detach().cpu()),
                      'osisaf':(('sample','time','yc','xc'),batch.coarse.detach().cpu()),
                      'lat':(('sample','time','yc','xc'),batch.latv.detach().cpu()),
                      'lon':(('sample','time','yc','xc'),batch.lonv.detach().cpu()),
                      'land_mask':(('sample','time','yc','xc'),batch.land_mask.detach().cpu()),
                      't2m':(('sample','time','yc','xc'),batch.t2m.detach().cpu()),
                      'istl1':(('sample','time','yc','xc'),batch.istl1.detach().cpu()),
                      'sst':(('sample','time','yc','xc'),batch.sst.detach().cpu()),
                      'skt':(('sample','time','yc','xc'),batch.skt.detach().cpu()),                      
                      },
           coords={'sample':np.arange(batch.input.shape[0]),
                   'time':np.arange(15),
                   'yc':np.arange(0, 240, 1),
                   'xc':np.arange(0, 240, 1)})
    if keep_obs:
        data = data.update({"input":(('sample','time','yc','xc'),batch.input.detach().cpu())})
    data.to_netcdf("/dmidata/users/maxb/ASIP_OSISAF_dataset/PREPROC/preproc_asip_"+ibatch+"_.nc")

    
def remove_useless_patches(batch):

    def nanvar(tensor, dim=None, keepdim=False):
        tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
        output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
        return output
    
    idx = []
    for i in range(len(batch.tgt)):
        # keep the patch if not full of NaN or not full of ice/water (var=0)
        if ( (batch.tgt[i].isfinite().float().mean() != 0) and (nanvar(batch.tgt[i])>=0.02) ):
            idx.append(i)
    if len(idx)>0:
        batch = batch._replace(input=batch.input[idx])
        batch = batch._replace(tgt=batch.tgt[idx])
        batch = batch._replace(coarse=batch.coarse[idx])
        batch = batch._replace(latv=batch.latv[idx])
        batch = batch._replace(lonv=batch.lonv[idx])
        batch = batch._replace(land_mask=batch.land_mask[idx])
        batch = batch._replace(t2m=batch.t2m[idx])
        batch = batch._replace(istl1=batch.istl1[idx])
        batch = batch._replace(sst=batch.sst[idx])
        batch = batch._replace(skt=batch.skt[idx])
    else:
        batch = None
    return batch

i = 0
for batch in data_loader:
    batch = remove_useless_patches(batch)
    if batch is None:
        continue
    x = batch.tgt
    if x.isfinite().float().mean() < 0.05:
        continue
    save_batch_as_NetCDF(batch, str(i), keep_obs=True)
    i = i+1

