"""
Preprocessing script for CROSCIM single-resolution data.
Extracts and filters useful patches for training.
"""
import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
print(os.getcwd())
sys.path.append('../../..')

from contrib.CROSCIM.dataloaders.data import *
from contrib.CROSCIM.dataloaders.load_data import *
from src.utils import *
from src.models import *

import matplotlib.pyplot as plt
import torch
import itertools
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature
import random

# ===== VARIABLE CONFIGURATION =====
# 'asip' is now optional: if omitted (or its value list left empty), the
# reference grid falls back to the finest-resolution remaining active source.
satellite_vars = {
    'asip': ['sic'],
    'cimr': ['SIC', 'SIT'],
    'cristal': ['SIT', 'SSH']
}

covariates = ["msl", "t2m", "u10", "v10"]

target_vars = ["tgt_sic", "tgt_SIT"]

var_mapping = {
    'tgt_sic': 'asip_sic',
    'tgt_SIT': 'cimr_SIT'
}

# ===== NORMALIZATION STATS =====
norm_stats = {
    'asip': {
        'sic': {'min': 0.0, 'max': 100.0, 'type': 'minmax'},
        'standard_deviation_sic': {'mean': 1.952899634621331, 'std': 4.922985470259985, 'type': 'zscore'},
        'status_flag': {'min': 0.0, 'max': 1536.0, 'type': 'minmax'}
    },
    'cimr': {
        'SIC': {'min': -0.049950417409564574, 'max': 1.0498479215517267, 'type': 'minmax'},
        'SIT': {'mean': 0.09544839558401243, 'std': 0.13231399596810897, 'type': 'zscore'},
        'Tsurf': {'mean': -7.098038910302271, 'std': 9.410905679026854, 'type': 'zscore'},
        'SICnoise': {'mean': 9.402084735405344e-07, 'std': 0.007446822627062462, 'type': 'zscore'},
        'SITnoise': {'mean': -1.244936869388826e-05, 'std': 0.003507472996277258, 'type': 'zscore'},
        'Tsurfnoise': {'mean': -1.6181464698155027e-05, 'std': 0.0901130348999105, 'type': 'zscore'}
    },
    'cristal': {
        'HS': {'mean': 0.15701109538657623, 'std': 0.14167074971036836, 'type': 'zscore'},
        'SIT': {'mean': 1.713899766845625, 'std': 1.0358012026065266, 'type': 'zscore'},
        'SSH': {'mean': 0.36872446726198005, 'std': 0.3961191636146796, 'type': 'zscore'},
        'HSnoise': {'mean': 0.237219498422366, 'std': 0.11141783328896704, 'type': 'zscore'},
        'SITnoise': {'mean': 1.7138151410883338, 'std': 1.0360327276131651, 'type': 'zscore'},
        'SSHnoise': {'mean': 0.36873967481665904, 'std': 0.3961927398202259, 'type': 'zscore'}
    }
}

norm_stats_covs = {
    'msl': {'mean': 101397.28559169156, 'std': 1178.6647666762826, 'type': 'zscore'},
    't2m': {'mean': 271.533815513639, 'std': 14.303287836457294, 'type': 'zscore'},
    'u10': {'mean': 0.5536510314408732, 'std': 4.341491519336185, 'type': 'zscore'},
    'v10': {'mean': 0.031581173569483305, 'std': 4.287231013697467, 'type': 'zscore'},
    'tcc': {'min': 0.0, 'max': 1.0, 'type': 'minmax'},
    'd2m': {'mean': 268.22918333426907, 'std': 13.96055261063253, 'type': 'zscore'},
    'ssrd': {'mean': -701143.5402508647, 'std': 785444.8336077137, 'type': 'zscore'},
    'strd': {'mean': -1943008.2121575351, 'std': 466464.69271380285, 'type': 'zscore'},
    'tp': {'mean': -0.0001834410365694742, 'std': 0.0005019462438424605, 'type': 'zscore'}
}

# ===== DATAMODULE INSTANTIATION =====
datamodule = BaseDataModule(
    # Data paths
    asip_paths=get_paths_for_source("asip"),
    cimr_paths=get_paths_for_source("cimr"),
    cristal_paths=get_paths_for_source("cristal"),
    covariates_paths=get_paths_for_source("covariates"),
    
    # Variable configuration
    satellite_vars=satellite_vars,
    covariates=covariates,
    target_vars=target_vars,
    var_mapping=var_mapping,
    
    # Mask and domain
    mask_path="/Odyssey/private/m19beauc/4dvarnet-starter/contrib/CROSCIM/mask_PanArctic.nc",
    domain_name="arctic_croscim",
    
    # Time domains
    domains={
        'train': {'time': slice('2022-05-01', '2022-12-31')},
        'val': {'time': [slice('2022-05-01', '2022-06-30'), slice('2022-07-01', '2022-12-31')]},
        'test': {'time': slice('2022-02-01', '2022-02-15')}
    },
    
    # Dataset configuration
    xrds_kw={
        'patch_dims': {'time': 15, 'yc': 256, 'xc': 256},
        'strides': {'time': 1, 'yc': 28, 'xc': 28},
        'strides_test': {'time': 1, 'yc': 200, 'xc': 200},
        'domain_limits': dict(
            xc=slice(-3849750., 3749750.),
            yc=slice(2473750., -4896250.)
        )
    },
    
    # DataLoader configuration
    dl_kw={'batch_size': 2, 'num_workers': 20},  # num_workers=0 for preprocessing
    
    # Resolution and normalization
    res=500,
    pads=[False, False, True],
    resize=2,
    norm_stats=norm_stats,
    norm_stats_covs=norm_stats_covs
)

print("\n" + "="*70)
print("DATAMODULE CONFIGURATION")
print("="*70)
print(f"Satellite vars: {satellite_vars}")
print(f"Covariates: {covariates}")
print(f"Target vars: {target_vars}")
print(f"Var mapping: {var_mapping}")
print(f"Active sources: {datamodule.active_sources}")
print(f"Input vars: {datamodule.input_vars}")
print("="*70 + "\n")

# Setup and create dataloader
datamodule.setup()
data_loader = datamodule.train_dataloader()


def remove_useless_patches(batch, vars_tgt=['tgt_sic', 'tgt_SIT'], 
                           threshold_num=0.2, threshold_var=0.02):
    """
    Filter batch to keep only useful patches (NaN-free and sufficiently variable).
    
    Args:
        batch: TrainingItem with batch dimension
        vars_tgt: list of target variable names used for filtering
        threshold_num: minimum proportion of valid (finite) values
        threshold_var: minimum variance threshold
        
    Returns:
        Filtered TrainingItem or None if no patches are useful.
    """
    def nanvar(tensor):
        """Compute variance ignoring NaN values."""
        mean = tensor.nanmean()
        return ((tensor - mean) ** 2).nanmean()

    # Get batch size from first target variable
    B = getattr(batch, vars_tgt[0]).shape[0]
    valid_idx = []

    # Check each sample in batch
    for i in range(B):
        keep = False
        for var in vars_tgt:
            x = getattr(batch, var)[i]
            # Check if enough valid data and sufficient variance
            if x.isfinite().float().mean() > threshold_num and nanvar(x) >= threshold_var:
                keep = True
                break
        if keep:
            valid_idx.append(i)

    if not valid_idx:
        print(f"  All {B} patches filtered out (no useful data)")
        return None

    print(f"  Kept {len(valid_idx)}/{B} patches")

    # Filter all fields in the TrainingItem
    batch_dict = batch._asdict() if hasattr(batch, '_asdict') else vars(batch)
    filtered_dict = {}
    
    for k, v in batch_dict.items():
        if torch.is_tensor(v) and v.shape[0] == B:
            filtered_dict[k] = v[valid_idx]
        else:
            filtered_dict[k] = v
    
    # Return filtered TrainingItem
    return TrainingItem(**filtered_dict)


# ===== MAIN PREPROCESSING LOOP =====
print("\nStarting preprocessing...")
print(f"Processing {len(data_loader)} batches\n")

num_saved = 0
num_filtered = 0

for i, batch in enumerate(data_loader):
    if i % 10 == 0:
        print(f"Processing batch {i}/{len(data_loader)}...")
    
    # Filter useless patches
    batch_filtered = remove_useless_patches(
        batch, 
        vars_tgt=target_vars,
        threshold_num=0.2,
        threshold_var=0.02
    )
    
    if batch_filtered is None:
        num_filtered += 1
        continue
    
    # Save to NetCDF
    datamodule.save_batch_as_NetCDF(
        batch_filtered,
        ibatch=str(random.randint(1, 100000)),
        patch_dims=datamodule.xrds_kw['patch_dims']
    )
    num_saved += 1

print("\n" + "="*70)
print("PREPROCESSING COMPLETE")
print("="*70)
print(f"Total batches processed: {len(data_loader)}")
print(f"Batches saved: {num_saved}")
print(f"Batches filtered out: {num_filtered}")
print("="*70)
