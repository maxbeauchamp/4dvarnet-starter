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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import importlib 

norm_stats = {'asip': {'sic': {'min': 0.0, 'max': 100.0, 'type': 'minmax'}, 'standard_deviation_sic': {'mean': 1.952899634621331, 'std': 4.922985470259985, 'type': 'zscore'}, 'status_flag': {'min': 0.0, 'max': 1536.0, 'type': 'minmax'}}, 'cimr': {'SIC': {'min': -0.049950417409564574, 'max': 1.0498479215517267, 'type': 'minmax'}, 'SIT': {'mean': 0.09544839558401243, 'std': 0.13231399596810897, 'type': 'zscore'}, 'Tsurf': {'mean': -7.098038910302271, 'std': 9.410905679026854, 'type': 'zscore'}, 'SICnoise': {'mean': 9.402084735405344e-07, 'std': 0.007446822627062462, 'type': 'zscore'}, 'SITnoise': {'mean': -1.244936869388826e-05, 'std': 0.003507472996277258, 'type': 'zscore'}, 'Tsurfnoise': {'mean': -1.6181464698155027e-05, 'std': 0.0901130348999105, 'type': 'zscore'}}, 'cristal': {'HS': {'mean': 0.15701109538657623, 'std': 0.14167074971036836, 'type': 'zscore'}, 'SIT': {'mean': 1.713899766845625, 'std': 1.0358012026065266, 'type': 'zscore'}, 'SSH': {'mean': 0.36872446726198005, 'std': 0.3961191636146796, 'type': 'zscore'}, 'HSnoise': {'mean': 0.237219498422366, 'std': 0.11141783328896704, 'type': 'zscore'}, 'SITnoise': {'mean': 1.7138151410883338, 'std': 1.0360327276131651, 'type': 'zscore'}, 'SSHnoise': {'mean': 0.36873967481665904, 'std': 0.3961927398202259, 'type': 'zscore'}}}

norm_stats_covs = {'msl': {'mean': 101397.28559169156, 'std': 1178.6647666762826, 'type': 'zscore'}, 't2m': {'mean': 271.533815513639, 'std': 14.303287836457294, 'type': 'zscore'}, 'u10': {'mean': 0.5536510314408732, 'std': 4.341491519336185, 'type': 'zscore'}, 'v10': {'mean': 0.031581173569483305, 'std': 4.287231013697467, 'type': 'zscore'}, 'tcc': {'min': 0.0, 'max': 1.0, 'type': 'minmax'}, 'd2m': {'mean': 268.22918333426907, 'std': 13.96055261063253, 'type': 'zscore'}, 'ssrd': {'mean': -701143.5402508647, 'std': 785444.8336077137, 'type': 'zscore'}, 'strd': {'mean': -1943008.2121575351, 'std': 466464.69271380285, 'type': 'zscore'}, 'tp': {'mean': -0.0001834410365694742, 'std': 0.0005019462438424605, 'type': 'zscore'}}

importlib.reload(contrib.CROSCIM.load_data)
from contrib.CROSCIM.load_data import *
importlib.reload(contrib.CROSCIM.data)
from contrib.CROSCIM.data import *
importlib.reload(contrib.CROSCIM.data_multires)
from contrib.CROSCIM.data_multires import *

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
        'patch_dims': {'time': 15, 'yc': 240, 'xc': 240},
        'strides': {'time': 1, 'yc': 20, 'xc': 20},
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

# train dataloader
"""
data_loader = datamodule.train_dataloader()
for i, batch in enumerate(data_loader):
    datamodule.save_batch_as_NetCDF_multires(batch,
                                             ibatch=str(i),
                                             patch_dims_dict={res: datamodule.xrds_kw['patch_dims'] for res in datamodule.multires}
                                             )
sample = data_loader.dataset[57440]
"""

# test dataloader
test_dataloader = datamodule.test_dataloader()

# test lightning module
from contrib.CROSCIM.solver import *
from contrib.CROSCIM.models import Lit4dVarNet_CROSCIM
from src.utils import *

optim_weight = {}
optim_weight["patch_x50"] = get_linear_time_wei(
                                                patch_dims={"time": 15, "yc": 240, "xc": 240},
                                                crop={"time": 0, "yc": 20, "xc": 20},
                                                offset=1,
                                                dim_order=["time","yc","xc"])
optim_weight["patch_x10"] = get_linear_time_wei(
                                                patch_dims={"time": 10, "yc": 240, "xc": 240},
                                                crop={"time": 0, "yc": 20, "xc": 20},
                                                offset=1,
                                                dim_order=["time","yc","xc"])
optim_weight["patch_x2"] = get_linear_time_wei(
                                                patch_dims={"time": 5, "yc": 240, "xc": 240},
                                                crop={"time": 0, "yc": 20, "xc": 20},
                                                offset=1,
                                                dim_order=["time","yc","xc"])

import importlib 
importlib.reload(contrib.CROSCIM.load_data)
from contrib.CROSCIM.load_data import *
importlib.reload(contrib.CROSCIM.data)
from contrib.CROSCIM.data import *
importlib.reload(contrib.CROSCIM.data_multires)
from contrib.CROSCIM.data_multires import *
importlib.reload(src.models)

from src.models import *
importlib.reload(contrib.CROSCIM.solver)
from contrib.CROSCIM.solver import *
importlib.reload(contrib.CROSCIM.models)
from contrib.CROSCIM.models import Lit4dVarNet_CROSCIM

model = Lit4dVarNet_CROSCIM(
    opt_fn = None,
    rec_weight=optim_weight,
    optim_weight=optim_weight,
    prior_weight=optim_weight,                    
    domain_limits={"xc":slice(-3849750.,3749750.), 
                   "yc":slice(2473750.,-4896250.)},
    frcst_lead=0,
    persist_rw=False,                 
    multires=[50,10,2],
    tgt_vars=["tgt_sic", "tgt_SIT"],
    solver=GradSolvers({
           "solver_x2": GradSolver(n_step=10,
                                   lr_grad=1e-3,
                                   prior_cost = BilinAEPriorCost(dim_in=45,dim_hidden=64, dim_out=10,
                                                                  bilin_quad=False,downsamp=2).to("cuda"),
                                   obs_cost = BaseObsCost().to("cuda"),
                                   grad_mod = ConvLstmGradModel(dim_in=45,dim_hidden=96)).to("cuda"),
           "solver_x10": GradSolver(n_step=5,
                                    lr_grad=1e-3,
                                    prior_cost = BilinAEPriorCost(dim_in=90,dim_hidden=64, dim_out=20,
                                                                  bilin_quad=False,downsamp=2).to("cuda"),
                                    obs_cost = BaseObsCost().to("cuda"),
                                    grad_mod = ConvLstmGradModel(dim_in=90,dim_hidden=96)).to("cuda"),
           "solver_x50": GradSolver(n_step=5,
                                   lr_grad=1e-3,
                                   prior_cost = BilinAEPriorCost(dim_in=135,dim_hidden=64, dim_out=30,
                                                                  bilin_quad=False,downsamp=2).to("cuda"),
                                   obs_cost = BaseObsCost().to("cuda"),
                                   grad_mod = ConvLstmGradModel(dim_in=135,dim_hidden=96).to("cuda"))
           }),
    norm_stats=norm_stats,
    norm_stats_covs=norm_stats_covs
)

"""
trainer = Trainer(accelerator="gpu",
                  devices=1,           # single GPU
                  precision=32,        # or 16 for mixed precision
                  #fast_dev_run=True,
                  max_epochs=10
                 )
trainer.fit(model, datamodule=datamodule)
# or Call training_step manually
# _, out = model.training_step(batch, batch_idx=0)
"""
from pytorch_lightning import Trainer
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    precision=32
)

#trainer.test(model, datamodule=datamodule)
trainer.test(model, dataloaders=test_dataloader)
