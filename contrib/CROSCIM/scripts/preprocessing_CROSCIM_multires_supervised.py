import sys
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
print(os.getcwd())
sys.path.append('../../..')
from contrib.CROSCIM.dataloaders.data_multires_supervised import *
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
        'standard_deviation_sic': {'mean': 1.952899634621331, 'std': 4.922985470259985, 'type': 'zscore'},
        'status_flag': {'min': 0.0, 'max': 1536.0, 'type': 'minmax'}
    },
    'cimr': {
        'SIC': {'min': -0.049950417409564574, 'max': 1.0498479215517267, 'type': 'minmax'},
        'SIT': {'mean': 0.09544839558401243, 'std': 0.13231399596810897, 'type': 'zscore'},
        'Tsurf': {'mean': -7.098038910302271, 'std': 9.410905679026854, 'type': 'zscore'},
    },
    'cristal': {
        'HS': {'mean': 0.15701109538657623, 'std': 0.14167074971036836, 'type': 'zscore'},
        'SIT': {'mean': 1.713899766845625, 'std': 1.0358012026065266, 'type': 'zscore'},
        'SSH': {'mean': 0.36872446726198005, 'std': 0.3961191636146796, 'type': 'zscore'},
    }
}

# NEW: Normalization stats for model variables
norm_stats_models = {
    'SIC': {'min': -0.049950417409564574, 'max': 1.0498479215517267, 'type': 'minmax'},
    'SIT': {'mean': 0.09544839558401243, 'std': 0.13231399596810897, 'type': 'zscore'},
    'HS': {'mean': 0.15701109538657623, 'std': 0.14167074971036836, 'type': 'zscore'},
    'SSH': {'mean': 0.36872446726198005, 'std': 0.3961191636146796, 'type': 'zscore'}
}

norm_stats_covs = {
    't2m': {'mean': 271.533815513639, 'std': 14.303287836457294, 'type': 'zscore'},
    'msl': {'mean': 101397.28559169156, 'std': 1178.6647666762826, 'type': 'zscore'},
    'u10': {'mean': 0.5536510314408732, 'std': 4.341491519336185, 'type': 'zscore'},
    'v10': {'mean': 0.031581173569483305, 'std': 4.287231013697467, 'type': 'zscore'},
    'tcc': {'min': 0.0, 'max': 1.0, 'type': 'minmax'},
    'd2m': {'mean': 268.22918333426907, 'std': 13.96055261063253, 'type': 'zscore'},
    'ssrd': {'mean': -701143.5402508647, 'std': 785444.8336077137, 'type': 'zscore'},
    'strd': {'mean': -1943008.2121575351, 'std': 466464.69271380285, 'type': 'zscore'},
    'tp': {'mean': -0.0001834410365694742, 'std': 0.0005019462438424605, 'type': 'zscore'}
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
        'train': {'time': slice('2022-01-01', '2022-03-28')},
        'val': {'time': slice('2022-01-01', '2022-02-28')},
        'test': {'time': slice('2022-02-01', '2022-02-15')}
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
        )
    },
    
    # DataLoader configuration
    dl_kw={'batch_size': 2, 'num_workers': 20},
    
    # Resolution and normalization
    res=500,
    pads=[False, False, True],
    multires=[50, 10, 2],
    norm_stats=norm_stats,
    norm_stats_covs=norm_stats_covs,
    norm_stats_models=norm_stats_models  # NEW
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

def remove_useless_patches_multires(batch, multires, vars_tgt=['tgt_sic', 'tgt_SIT'], 
                                    threshold_num=0.2, threshold_var=0.02):
    """
    Filtre les batchs multirésolution en ne gardant que les patchs valides (NaN-free et suffisamment variables)
    sur la résolution la plus fine. Applique la sélection à toutes les résolutions.
    
    Args:
        batch: dict, contient des TrainingItems nommés "patch_x{res}"
        multires: list of int, les résolutions, e.g., [50, 10, 2]
        vars_tgt: liste des variables cibles utilisées pour la sélection
        threshold_num: seuil minimum de data (proportion de valeurs finies)
        threshold_var: seuil minimum de variance
        
    Returns:
        dict filtré contenant les mêmes clés que `batch`, ou None si aucun patch n'est utile.
    """
    def nanvar(tensor):
        """Compute variance ignoring NaN values."""
        mean = tensor.nanmean()
        return ((tensor - mean) ** 2).nanmean()

    # Use finest resolution for filtering
    fine_res_key = f"patch_x{multires[-1]}"
    batch_fine = batch[fine_res_key]
    
    # Get batch size from first target variable
    B = getattr(batch_fine, vars_tgt[0]).shape[0]
    valid_idx = []

    # Check each sample in batch
    for i in range(B):
        keep = False
        for var in vars_tgt:
            x = getattr(batch_fine, var)[i]
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

def verify_target_initialization(batch, var_mapping, multires):
    """
    NEW: Verify that targets were correctly initialized from their sources
    according to resolution-dependent var_mapping.
    
    Args:
        batch: dict with keys 'patch_x2', 'patch_x10', 'patch_x50'
        var_mapping: dict mapping resolution keys to {target: source} dicts
        multires: list of resolution factors
    """
    print("\n" + "="*70)
    print("VERIFYING TARGET INITIALIZATION BY RESOLUTION")
    print("="*70)
    
    for res_factor in multires:
        res_key = f"patch_x{res_factor}"
        
        if res_key not in batch:
            print(f"  {res_key}: NOT FOUND in batch")
            continue
        
        batch_res = batch[res_key]
        mapping = var_mapping.get(res_key, {})
        
        print(f"\n{res_key}:")
        print(f"  Mapping: {mapping}")
        
        for tgt_var, src_var in mapping.items():
            if hasattr(batch_res, tgt_var) and hasattr(batch_res, src_var):
                tgt_data = getattr(batch_res, tgt_var)
                src_data = getattr(batch_res, src_var)
                
                # Check if they match (allowing for numerical precision)
                match = torch.allclose(tgt_data, src_data, rtol=1e-5, atol=1e-5, equal_nan=True)
                
                print(f"    {tgt_var} <- {src_var}: ", end="")
                if match:
                    print(f"✓ MATCH (shape: {tgt_data.shape})")
                else:
                    print(f"✗ MISMATCH!")
                    print(f"      Target range: [{tgt_data.nanmin():.4f}, {tgt_data.nanmax():.4f}]")
                    print(f"      Source range: [{src_data.nanmin():.4f}, {src_data.nanmax():.4f}]")
            else:
                missing = []
                if not hasattr(batch_res, tgt_var):
                    missing.append(tgt_var)
                if not hasattr(batch_res, src_var):
                    missing.append(src_var)
                print(f"    {tgt_var} <- {src_var}: ✗ MISSING ({', '.join(missing)})")
    
    print("="*70 + "\n")

# ===== MAIN PREPROCESSING LOOP =====
print("\nStarting preprocessing (supervised mode with resolution-dependent targets)...")
print(f"Processing {len(data_loader)} batches\n")

num_saved = 0
num_filtered = 0
first_batch_verified = False

for i, batch in enumerate(data_loader):
    if i % 10 == 0:
        print(f"Processing batch {i}/{len(data_loader)}...")
    
    # Verify target initialization on first batch
    if not first_batch_verified:
        verify_target_initialization(batch, var_mapping, datamodule.multires)
        first_batch_verified = True
    
    # Filter useless patches
    batch_filtered = remove_useless_patches_multires(
        batch, 
        multires=[50, 10, 2],
        vars_tgt=target_vars,
        threshold_num=0.2,
        threshold_var=0.02
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
print(f"\nTarget initialization strategy:")
for res_key, mapping in var_mapping.items():
    print(f"  {res_key}:")
    for tgt, src in mapping.items():
        src_type = "satellite" if "asip" in src or "cimr" in src or "cristal" in src else "model"
        print(f"    {tgt} <- {src} ({src_type})")
print("="*70)
