import sys
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
print(os.getcwd())
sys.path.append('../../..')
from contrib.CROSCIM.dataloaders.data_multires_supervised import *
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
satellite_vars = {
    'asip': ['sic'],
    'cimr': ['SIC', 'SIT'],
    'cristal': ['SIT', 'SSH']
}

# NEW: Model variables
models_vars = ['SIC', 'SIT', 'HS', 'SSH']

# MODIFIED: Reduced covariates (t2m, msl now from models)
covariates = ["t2m", "msl", "u10", "v10"]

target_vars = {
    "patch_x50": ["models_SIT", "models_SIC"],
    "patch_x10": ["models_SIT", "models_SIC"],
    "patch_x2": ["models_SIT", "tgt_SIC"],
}

var_mapping = {
    "patch_x50": {
        "models_SIT": "cristal_SIT",
        "models_SIC": "cimr_SIC",
    },
    "patch_x10": {
        "models_SIT": "cristal_SIT",
        "models_SIC": "cimr_SIC",
    },
    "patch_x2": {
        "models_SIT": "cristal_SIT",
        "tgt_SIC": "asip_sic",
    },
}
# ===== NORMALIZATION STATS =====
norm_stats = {
    'asip': {
        'sic': {'min': 0.0, 'max': 100.0, 'type': 'minmax'},
        'standard_deviation_sic': {'mean': 1.8761531586097209, 'std': 4.922335431544444, 'type': 'zscore'},
        'status_flag': {'min': 0.0, 'max': 1536.0, 'type': 'minmax'}
    },
    'cimr': {
        'SIC': {'min': 0.0, 'max': 1.0, 'type': 'minmax'},
        #'SIT': {'mean': 0.02392856443828687, 'std': 0.08694672390807144, 'type': 'zscore'},
        'SIT': {'mean': 0.5695980677516013, 'std': 0.8216149733058172, 'type': 'zscore'},
        'Tsurf': {'mean': -6.3703807274976665, 'std': 9.394544766238061, 'type': 'zscore'},
    },
    'cristal': {
        'HS': {'mean': 0.15517908473300512, 'std': 0.13962813089282153, 'type': 'zscore'},
        #'SIT': {'mean': 1.3583107984977745, 'std': 0.7628621685375375, 'type': 'zscore'},
        'SIT': {'mean': 0.5695980677516013, 'std': 0.8216149733058172, 'type': 'zscore'},
        'SSH': {'mean': 0.1912636630889516, 'std': 0.41697635927509086, 'type': 'zscore'},
    }
}

norm_stats_covs = {
    'msl': {'mean': 101261.65922657882, 'std': 1112.7187312302365, 'type': 'zscore'},
    't2m': {'mean': -2.5985455071070955, 'std': 13.613745769755358, 'type': 'zscore'},
    'u10': {'mean': 0.6010106010949151, 'std': 4.545559141871505, 'type': 'zscore'},
    'v10': {'mean': -0.2104646166032582, 'std': 4.707062023219409, 'type': 'zscore'},
    'tcc': {'min': 1.0896958883677144e-05, 'max': 0.024753304198384285, 'type': 'minmax'},
    'd2m': {'mean': 2.4095910147380064e-08, 'std': 4.900962447374831e-08, 'type': 'zscore'},
    'ssrd': {'mean': 95.72918023746189, 'std': 93.44316534428312, 'type': 'zscore'},
    'strd': {'mean': 265.25246499467744, 'std': 63.12553499850183, 'type': 'zscore'},
    'tp': {'mean': 2.4095910147380064e-08, 'std': 4.900962447374831e-08, 'type': 'zscore'}
}

norm_stats_models = {
    'SIC': {'min': 0.0, 'max': 1.0, 'type': 'minmax'},
    'SIT': {'mean': 0.5695980677516013, 'std': 0.8216149733058172, 'type': 'zscore'},
    'HS': {'mean': 0.06844576249551623, 'std': 0.1137041260225963, 'type': 'zscore'},
    'SSH': {'mean': 0.21988234269231605, 'std': 0.2757902493660287, 'type': 'zscore'}
}

# ===== DATAMODULE INSTANTIATION =====
datamodule = BaseDataModuleMultiRes(
    # Data paths
    asip_paths=get_paths_for_source("asip"),
    cimr_paths=get_paths_for_source("cimr"),
    cristal_paths=get_paths_for_source("cristal"),
    covariates_paths=get_paths_for_source("covariates"),
    models_paths=get_paths_for_source("models"),
    
    # Variable configuration
    satellite_vars=satellite_vars,
    covariates=covariates,
    models_vars=models_vars, 
    target_vars=target_vars,
    var_mapping=var_mapping,  # NOW RESOLUTION-DEPENDENT
    
    # Mask and domain
    mask_path="/dmidata/users/maxb/4dvarnet-starter/contrib/CROSCIM/mask_PanArctic.nc",
    domain_name="arctic_croscim",
    
    # Time domains
    domains={
        'train': {'time': slice('2021-01-01', '2021-08-28')},
        'val': {'time': slice('2021-01-01', '2021-01-15')},
        'test': {'time': slice('2021-01-01', '2021-01-15')}
        #'train': {'time': slice('2022-05-01', '2022-12-31')},
        #'val': {'time': [slice('2022-05-01', '2022-06-30'), slice('2022-07-01', '2022-12-31')]},
        #'test': {'time': slice('2022-02-01', '2022-02-15')}
    },
    
    # Dataset configuration
    xrds_kw={
        'patch_dims': {'time': 15, 'yc': 256, 'xc': 256},
        'strides': {'time': 1, 'yc': 28, 'xc': 28},
        'strides_test': {'time': 1, 'yc': 200, 'xc': 200},
        'domain_limits': dict(
            xc=slice(-3849750., 3749750.),
            yc=slice(2473750., -4896250.)
        ),
    },
    
    # DataLoader configuration
    dl_kw={'batch_size': 2, 'num_workers': 20},
    
    # Resolution and normalization
    res=500,
    pads=[False, False, True],
    multires=[50, 10, 2],
    norm_stats=norm_stats,
    norm_stats_covs=norm_stats_covs,
    norm_stats_models=norm_stats_models,
    rand_obs=["asip_sic"]#, "cristal_SIT"]
)

print("\n" + "="*70)
print("DATAMODULE CONFIGURATION (SUPERVISED - MULTI-RESOLUTION TARGETS)")
print("="*70)
print(f"Satellite vars: {satellite_vars}")
print(f"Models vars: {models_vars}")
print(f"Covariates: {covariates}")
print(f"Target vars: {target_vars}")
print(f"\nVariable mapping BY RESOLUTION:")
for res_key, mapping in var_mapping.items():
    print(f"  {res_key}:")
    for tgt, src in mapping.items():
        print(f"    {tgt} <- {src}")
print(f"\nMulti-resolution factors: {datamodule.multires}")
print(f"Active sources: {datamodule.active_sources}")
print("="*70 + "\n")

# Setup and create dataloader
datamodule.setup()
data_loader = datamodule.train_dataloader()

def remove_useless_patches_multires(batch, multires, vars_tgt=['asip_sic', 'cristal_SIT'], 
                                    threshold_num=None, threshold_var=None, 
                                    relaxed_threshold_var=None, relaxed_acceptance_rate=0.05):
    """
    Filtre les batchs multirésolution en ne gardant que les patchs valides (NaN-free et suffisamment variables)
    sur la résolution la plus fine. Applique la sélection à toutes les résolutions.
    
    Accepte occasionnellement (10% du temps) des patchs avec variance plus faible pour augmenter la diversité.
    
    Args:
        batch: dict, contient des TriningItems nommés "patch_x{res}"
        multires: list of int, les résolutions, e.g., [50, 10, 2]
        vars_tgt: liste des variables cibles utilisées pour la sélection
        threshold_num: seuil minimum de data (proportion de valeurs finies) - float ou dict {var: threshold}
        threshold_var: seuil minimum de variance (critère strict) - float ou dict {var: threshold}
        relaxed_threshold_var: seuil relaxé de variance - float ou dict {var: threshold}
        relaxed_acceptance_rate: proportion de batchs autorisés avec variance relaxée (0.1 = 10%)
        
    Returns:
        dict filtré contenant les mêmes clés que `batch`, ou None si aucun patch n'est utile.
    """
    def nanvar(tensor):
        """Compute variance ignoring NaN values."""
        mean = tensor.nanmean()
        return ((tensor - mean) ** 2).nanmean()
    
    # Convert scalar thresholds to dicts if needed
    if threshold_num is None:
        threshold_num = {var: 0.2 for var in vars_tgt}
    elif not isinstance(threshold_num, dict):
        threshold_num = {var: threshold_num for var in vars_tgt}
    
    if threshold_var is None:
        threshold_var = {var: 0.02 for var in vars_tgt}
    elif not isinstance(threshold_var, dict):
        threshold_var = {var: threshold_var for var in vars_tgt}
    
    if relaxed_threshold_var is None:
        relaxed_threshold_var = {var: 0.005 for var in vars_tgt}
    elif not isinstance(relaxed_threshold_var, dict):
        relaxed_threshold_var = {var: relaxed_threshold_var for var in vars_tgt}

    # Use finest resolution for filtering
    fine_res_key = f"patch_x{multires[-1]}"
    batch_fine = batch[fine_res_key]
    
    # Get batch size from first target variable
    B = getattr(batch_fine, vars_tgt[0]).shape[0]
    valid_idx_strict = []
    valid_idx_relaxed = []

    # Check each sample in batch
    for i in range(B):
        keep_strict = False
        keep_relaxed = False
        for var in vars_tgt:
            x = getattr(batch_fine, var)[i]
            valid_ratio = x.isfinite().float().mean()
            var_value = nanvar(x)
            
            # Get thresholds for this variable
            thr_num = threshold_num.get(var, 0.2)
            thr_var = threshold_var.get(var, 0.02)
            thr_var_relaxed = relaxed_threshold_var.get(var, 0.005)
            
            # Check strict criteria
            if valid_ratio > thr_num and var_value >= thr_var:
                keep_strict = True
                break
            # Check relaxed criteria
            elif valid_ratio > thr_num and var_value >= thr_var_relaxed:
                keep_relaxed = True
        
        if keep_strict:
            valid_idx_strict.append(i)
        elif keep_relaxed:
            valid_idx_relaxed.append(i)

    # Accept with probability to maintain 90% strict / 10% relaxed ratio
    # If we have strict samples, randomly add some relaxed ones
    valid_idx = valid_idx_strict.copy()
    
    if valid_idx_relaxed:
        # Calculate how many relaxed samples to add to maintain ~10% relaxed
        n_strict = len(valid_idx_strict)
        # Target: relaxed / (strict + relaxed) ≈ 0.1
        # So: relaxed ≈ 0.1 * (strict + relaxed) => relaxed ≈ 0.111 * strict
        n_relaxed_target = max(1, int(n_strict * relaxed_acceptance_rate / (1 - relaxed_acceptance_rate)))
        n_relaxed_to_add = min(len(valid_idx_relaxed), n_relaxed_target)
        
        # Randomly select relaxed samples to add
        if n_relaxed_to_add > 0:
            relaxed_selected = random.sample(valid_idx_relaxed, n_relaxed_to_add)
            valid_idx.extend(relaxed_selected)
            print(f"  Kept {len(valid_idx_strict)}/{B} strict + {n_relaxed_to_add}/{len(valid_idx_relaxed)} relaxed patches")
        else:
            print(f"  Kept {len(valid_idx_strict)}/{B} patches (strict only)")
    else:
        if valid_idx:
            print(f"  Kept {len(valid_idx)}/{B} patches (strict only)")

    if not valid_idx:
        print(f"  All {B} patches filtered out (no useful data)")
        return None

    # Apply filter to all resolutions
    batch_filtered = {}
    for res in multires:
        key = f"patch_x{res}"
        item = batch[key]
        item_dict = item._asdict()
        
        # Filter all tensor fields
        for k, v in item_dict.items():
            if torch.is_tensor(v) and v.shape[0] == B:
                item_dict[k] = v[valid_idx]
        
        batch_filtered[key] = type(item)(**item_dict)

    return batch_filtered

# ===== MAIN PREPROCESSING LOOP =====
print("\nStarting preprocessing (supervised mode with resolution-dependent targets)...")
print(f"Processing {len(data_loader)} batches\n")

num_saved = 0
num_filtered = 0
first_batch_verified = False

for i, batch in enumerate(data_loader):
    if i % 10 == 0:
        print(f"Processing batch {i}/{len(data_loader)}...")
    
    # Filter useless patches
    batch_filtered = remove_useless_patches_multires(
        batch, 
        multires=[50, 10, 2],
        vars_tgt=["asip_sic","cristal_SIT"],  # Use finest resolution targets
        threshold_num={'asip_sic': 0.2, 'cristal_SIT': 0.02},
        threshold_var={'asip_sic': 0.02, 'cristal_SIT': 0.02},
        relaxed_threshold_var={'asip_sic': 0.0, 'cristal_SIT': 0.0},  # Relaxed variance threshold
        relaxed_acceptance_rate=0.05   # Accept 5% with relaxed criteria
    )
    
    if batch_filtered is None:
        num_filtered += 1
        continue
    
    # Save to NetCDF
    datamodule.save_batch_as_NetCDF_multires(
        batch_filtered,
        ibatch=str(random.randint(1, 100000)),
        patch_dims_dict={
            res: datamodule.xrds_kw['patch_dims'] 
            for res in datamodule.multires
        }
    )
    num_saved += 1

print("\n" + "="*70)
print("PREPROCESSING COMPLETE (SUPERVISED - MULTI-RESOLUTION TARGETS)")
print("="*70)
print(f"Total batches processed: {len(data_loader)}")
print(f"Batches saved: {num_saved}")
print(f"Batches filtered out: {num_filtered}")
