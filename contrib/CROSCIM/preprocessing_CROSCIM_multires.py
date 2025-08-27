import sys
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
print(os.getcwd())
sys.path.append('../..')
from contrib.CROSCIM.data_multires import *
from contrib.CROSCIM.load_data import *
from src.utils import *
from src.models import *

import matplotlib.pyplot as plt
import torch
import itertools
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature

import random

# Instanciation du datamodule avec la nouvelle logique

norm_stats = {'asip': {'sic': {'min': 0.0, 'max': 100.0, 'type': 'minmax'}, 'standard_deviation_sic': {'mean': 1.952899634621331, 'std': 4.922985470259985, 'type': 'zscore'}, 'status_flag': {'min': 0.0, 'max': 1536.0, 'type': 'minmax'}}, 'cimr': {'SIC': {'min': -0.049950417409564574, 'max': 1.0498479215517267, 'type': 'minmax'}, 'SIT': {'mean': 0.09544839558401243, 'std': 0.13231399596810897, 'type': 'zscore'}, 'Tsurf': {'mean': -7.098038910302271, 'std': 9.410905679026854, 'type': 'zscore'}, 'SICnoise': {'mean': 9.402084735405344e-07, 'std': 0.007446822627062462, 'type': 'zscore'}, 'SITnoise': {'mean': -1.244936869388826e-05, 'std': 0.003507472996277258, 'type': 'zscore'}, 'Tsurfnoise': {'mean': -1.6181464698155027e-05, 'std': 0.0901130348999105, 'type': 'zscore'}}, 'cristal': {'HS': {'mean': 0.15701109538657623, 'std': 0.14167074971036836, 'type': 'zscore'}, 'SIT': {'mean': 1.713899766845625, 'std': 1.0358012026065266, 'type': 'zscore'}, 'SSH': {'mean': 0.36872446726198005, 'std': 0.3961191636146796, 'type': 'zscore'}, 'HSnoise': {'mean': 0.237219498422366, 'std': 0.11141783328896704, 'type': 'zscore'}, 'SITnoise': {'mean': 1.7138151410883338, 'std': 1.0360327276131651, 'type': 'zscore'}, 'SSHnoise': {'mean': 0.36873967481665904, 'std': 0.3961927398202259, 'type': 'zscore'}}}

norm_stats_covs = {'msl': {'mean': 101397.28559169156, 'std': 1178.6647666762826, 'type': 'zscore'}, 't2m': {'mean': 271.533815513639, 'std': 14.303287836457294, 'type': 'zscore'}, 'u10': {'mean': 0.5536510314408732, 'std': 4.341491519336185, 'type': 'zscore'}, 'v10': {'mean': 0.031581173569483305, 'std': 4.287231013697467, 'type': 'zscore'}, 'tcc': {'min': 0.0, 'max': 1.0, 'type': 'minmax'}, 'd2m': {'mean': 268.22918333426907, 'std': 13.96055261063253, 'type': 'zscore'}, 'ssrd': {'mean': -701143.5402508647, 'std': 785444.8336077137, 'type': 'zscore'}, 'strd': {'mean': -1943008.2121575351, 'std': 466464.69271380285, 'type': 'zscore'}, 'tp': {'mean': -0.0001834410365694742, 'std': 0.0005019462438424605, 'type': 'zscore'}}

datamodule = BaseDataModuleMultiRes(
    asip_paths=load_data("asip"),
    cimr_paths=load_data("cimr"),
    cristal_paths=load_data("cristal"),
    covariates_paths=load_data("era5"),
    covariates=["msl", "t2m", "u10", "v10", "tcc", "d2m", "ssrd", "strd", "tp"],
    tgt_vars=["asip_sic","cimr_SIT"],
    mask_path="/dmidata/users/maxb/4dvarnet-starter/contrib/CROSCIM/mask_PanArctic.nc",
    domain_name="arctic_croscim",
    domains={
        'train': {'time': slice('2022-05-01', '2022-12-31')},
        'val': {'time': [slice('2022-05-01', '2022-06-30'), slice('2022-07-01', '2022-12-31')]},
        'test': {'time': slice('2022-02-01', '2022-02-15')}
    },
    xrds_kw={
        'patch_dims': {'time': 15, 'yc': 256, 'xc': 256},
        'strides': {'time': 1, 'yc': 28, 'xc': 28},
        'strides_test': {'time': 1, 'yc': 200, 'xc': 200},
        'domain_limits': dict(xc=slice(-3849750., 3749750.), yc=slice(2473750.,-4896250.))
    },
    dl_kw={'batch_size': 2, 'num_workers': 20},
    res=500,
    pads=[False, False, True],
    norm_stats=norm_stats,
    norm_stats_covs=norm_stats_covs,
    multires=[50,10,2]
)

datamodule.setup()
data_loader = datamodule.train_dataloader()

def remove_useless_patches_multires(batch, multires, vars_tgt=['tgt_sic', 'cimr_SIT'], 
                                    threshold_num=0.2, threshold_var=0.02):
    """
    Filtre les batchs multirésolution en ne gardant que les patchs valides (NaN-free et suffisamment variables)
    sur la résolution la plus fine. Applique la sélection à toutes les résolutions.
    Args:
        batch: dict, contient des TrainingItems nommés "patch_x{res}"
        multires: list of int, les résolutions, e.g., [50, 10, 1]
        vars_tgt: liste des variables cibles utilisées pour la sélection (e.g. ['sic', 'SIT'])
        threshold_num: seuil minimum de data
        threshold_var: seuil minimum de variance
    Returns:
        dict filtré contenant les mêmes clés que `batch`, ou None si aucun patch n'est utile.
    """
    def nanvar(tensor):
        mean = tensor.nanmean()
        return ((tensor - mean) ** 2).nanmean()

    fine_res_key = f"patch_x{multires[-1]}"
    batch_fine = batch[fine_res_key]
    B = getattr(batch_fine, vars_tgt[0]).shape[0]
    valid_idx = []

    for i in range(B):
        keep = False
        for var in vars_tgt:
            x = getattr(batch_fine, var)[i]
            if x.isfinite().float().mean() > threshold_num and nanvar(x) >= threshold_var:
                keep = True
                break
        if keep:
            valid_idx.append(i)

    if not valid_idx:
        return None

    # Appliquer le filtre à chaque résolution
    batch_filtered = {}
    for res in multires:
        key = f"patch_x{res}"
        item = batch[key]
        item_dict = item._asdict()
        for k, v in item_dict.items():
            if torch.is_tensor(v) and v.shape[0] == B:
                item_dict[k] = v[valid_idx]
        batch_filtered[key] = type(item)(**item_dict)

    return batch_filtered

for i, batch in enumerate(data_loader):
    batch = remove_useless_patches_multires(batch, multires=[50,10,2])
    if batch is None:
        continue
    datamodule.save_batch_as_NetCDF_multires(batch,
                                             ibatch=str(random.randint(1, 100000)),
                                             patch_dims_dict={res: datamodule.xrds_kw['patch_dims'] for res in datamodule.multires}
                                             )
