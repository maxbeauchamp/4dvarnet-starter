import sys
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
print(os.getcwd())
sys.path.append('../..')
from contrib.CROSCIM.data import *
from contrib.CROSCIM.load_data import *
from src.utils import *
from src.models import *

import matplotlib.pyplot as plt
import torch
import itertools
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature

# Instanciation du datamodule avec la nouvelle logique

norm_stats = {'asip': {'sic': {'min': 0.0, 'max': 100.0, 'type': 'minmax'}, 'standard_deviation_sic': {'mean': 1.952899634621331, 'std': 4.922985470259985, 'type': 'zscore'}, 'status_flag': {'min': 0.0, 'max': 1536.0, 'type': 'minmax'}}, 'cimr': {'SIC': {'min': -0.049950417409564574, 'max': 1.0498479215517267, 'type': 'minmax'}, 'SIT': {'mean': 0.09544839558401243, 'std': 0.13231399596810897, 'type': 'zscore'}, 'Tsurf': {'mean': -7.098038910302271, 'std': 9.410905679026854, 'type': 'zscore'}, 'SICnoise': {'mean': 9.402084735405344e-07, 'std': 0.007446822627062462, 'type': 'zscore'}, 'SITnoise': {'mean': -1.244936869388826e-05, 'std': 0.003507472996277258, 'type': 'zscore'}, 'Tsurfnoise': {'mean': -1.6181464698155027e-05, 'std': 0.0901130348999105, 'type': 'zscore'}}, 'cristal': {'HS': {'mean': 0.15701109538657623, 'std': 0.14167074971036836, 'type': 'zscore'}, 'SIT': {'mean': 1.713899766845625, 'std': 1.0358012026065266, 'type': 'zscore'}, 'SSH': {'mean': 0.36872446726198005, 'std': 0.3961191636146796, 'type': 'zscore'}, 'HSnoise': {'mean': 0.237219498422366, 'std': 0.11141783328896704, 'type': 'zscore'}, 'SITnoise': {'mean': 1.7138151410883338, 'std': 1.0360327276131651, 'type': 'zscore'}, 'SSHnoise': {'mean': 0.36873967481665904, 'std': 0.3961927398202259, 'type': 'zscore'}}}

norm_stats_covs = {'msl': {'mean': 101397.28559169156, 'std': 1178.6647666762826, 'type': 'zscore'}, 't2m': {'mean': 271.533815513639, 'std': 14.303287836457294, 'type': 'zscore'}, 'u10': {'mean': 0.5536510314408732, 'std': 4.341491519336185, 'type': 'zscore'}, 'v10': {'mean': 0.031581173569483305, 'std': 4.287231013697467, 'type': 'zscore'}, 'tcc': {'min': 0.0, 'max': 1.0, 'type': 'minmax'}, 'd2m': {'mean': 268.22918333426907, 'std': 13.96055261063253, 'type': 'zscore'}, 'ssrd': {'mean': -701143.5402508647, 'std': 785444.8336077137, 'type': 'zscore'}, 'strd': {'mean': -1943008.2121575351, 'std': 466464.69271380285, 'type': 'zscore'}, 'tp': {'mean': -0.0001834410365694742, 'std': 0.0005019462438424605, 'type': 'zscore'}}

datamodule = BaseDataModule(
    asip_paths=load_data("asip"),
    cimr_paths=load_data("cimr"),
    cristal_paths=load_data("cristal"),
    covariates_paths=load_data("era5"),
    covariates=["msl", "t2m", "u10", "v10", "tcc", "d2m", "ssrd", "strd", "tp"],
    tgt_vars=["asip_sic","cimr_SIT"],
    mask_path="/dmidata/users/maxb/4dvarnet-starter/contrib/CROSCIM/mask_PanArctic.nc",
    domain_name="arctic_small",
    domains={
        'train': {'time': slice('2022-01-01', '2022-12-31')},
        'val': {'time': [slice('2022-01-01', '2022-06-30'), slice('2022-07-01', '2022-12-31')]},
        'test': {'time': slice('2022-02-01', '2022-02-02')}
    },
    xrds_kw={
        'patch_dims': {'time': 15, 'yc': 240, 'xc': 240},
        'strides': {'time': 1, 'yc': 20, 'xc': 20},
        'strides_test': {'time': 1, 'yc': 200, 'xc': 200},
        'domain_limits': dict(xc=slice(-3849750., 3749750.), yc=slice(3849750., -3349750.))
    },
    dl_kw={'batch_size': 1, 'num_workers': 1},
    res=500,
    pads=[False, False, True],
    norm_stats=norm_stats,
    norm_stats_covs=norm_stats_covs
)
datamodule.setup()
data_loader = datamodule.train_dataloader()

def remove_useless_patches(batch, var_tgt='sic'):
    """
    Remove patches with NaNs or constant values from the batch.
    Assumes 'sic' or equivalent is the target.
    """
    def nanvar(tensor):
        tensor_mean = tensor.nanmean(dim=None, keepdim=True)
        return (tensor - tensor_mean).square().nanmean()

    idx = []
    for i in range(getattr(batch, var_tgt).shape[0]):
        x = getattr(batch, var_tgt)[i]
        if (x.isfinite().float().mean() != 0) and (nanvar(x) >= 0.02):
            idx.append(i)

    if len(idx) > 0:
        batch_dict = batch._asdict()
        for key in batch_dict:
            if torch.is_tensor(batch_dict[key]) and batch_dict[key].shape[0] == len(getattr(batch, var_tgt)):
                batch_dict[key] = batch_dict[key][idx]
        return type(batch)(**batch_dict)
    else:
        return None   
     
for i, batch in enumerate(data_loader):
    #batch = remove_useless_patches(batch, var_tgt="asip_sic")
    #if batch is None:
    #    continue
    #if getattr(batch, "asip_sic").isfinite().float().mean() < 0.99:
    #    continue
    datamodule.save_batch_as_NetCDF(batch, str(i), patch_dims=datamodule.xrds_kw['patch_dims'], keep_obs=True)
    i+=1
