from random import sample
import contrib
from contrib.CROSCIM.dataloaders.load_data import *
from contrib.CROSCIM.dataloaders.data_simple import *
import datetime
import pyresample
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature 
import shapely.geometry as sgeom
import os
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F

class XrDatasetMultiRes_simplify(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    
    def __init__(self, paths, split, multires, build_batch, input_vars=None):
        'Initialization'
        self.multires = multires
        self.input_vars = input_vars  # Store which variables are available
        self.db = {}

        for res in self.multires:
            self.db[f"patch_x{res}"] = xr.open_dataset(paths[f"patch_x{res}"]).isel(sample=split)
        
        self.build_batch = build_batch

    def __len__(self):
        'Denotes the total number of samples'
        res_min = self.multires[-1]
        # Use first available variable instead of hardcoded 'asip_sic'
        first_var = list(self.db[f"patch_x{res_min}"].data_vars)[0]
        return len(self.db[f"patch_x{res_min}"][first_var])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        out = {}
        for res in self.multires:
            item = self.db[f"patch_x{res}"].isel(sample=idx)
            
            # Only select variables that exist in TrainingItem
            available_fields = [f for f in TrainingItem._fields if f in item.data_vars or f in item.coords]
            item = item[available_fields]
            
            var_dict = {var: item[var].values for var in item.data_vars}
            var_dict["time"] = item.time.data
            var_dict["xc"] = item.xc.data
            var_dict["yc"] = item.yc.data
            
            item = self.build_batch(var_dict)
            out[f"patch_x{res}"] = item
        
        return out

class BaseDataModuleMultiRes_simplify(pl.LightningDataModule):
    def __init__(self, 
                 croscim_preproc_paths,
                 multires,
                 split_train,
                 split_val,
                 split_test,
                 norm_stats,
                 norm_stats_covs,
                 satellite_vars=None,  
                 covariates=None,      
                 target_vars=None,    
                 **kwargs):

        super().__init__()
        self.croscim_preproc_paths = croscim_preproc_paths
        self.multires = multires
        self.split_train = split_train
        self.split_val = split_val
        self.split_test = split_test
        self._norm_stats = norm_stats
        self._norm_stats_covs = norm_stats_covs
        
        # Store variable configuration
        self.satellite_vars = satellite_vars or DEFAULT_VAR_GROUPS
        self.covariates = covariates or DEFAULT_COVARIATES
        self.target_vars = target_vars or ["tgt_sic", "tgt_SIT"]
        
        # Construct input_vars automatically
        self.input_vars = self._construct_input_vars()
        
        # Determine which sources are actually needed
        self.active_sources = [src for src, vars in self.satellite_vars.items() if vars]
        
        print(f"\n{'='*60}")
        print(f"DataModule configuration:")
        print(f"{'='*60}")
        print(f"  Satellite vars: {self.satellite_vars}")
        print(f"  Active sources: {self.active_sources}")
        print(f"  Covariates: {self.covariates}")
        print(f"  Target vars: {self.target_vars}")
        print(f"  Input vars: {self.input_vars}")
        print(f"  Multires: {self.multires}")
        print(f"{'='*60}\n")

    def _construct_input_vars(self):
        """Construct input variable names from satellite_vars + covariates"""
        input_vars = []
        
        # Add satellite variables with source prefix
        for source, vars in self.satellite_vars.items():
            for var in vars:
                input_vars.append(f"{source}_{var}")
        
        # Add covariates
        if self.covariates:
            input_vars.extend(self.covariates)
        
        return input_vars

    @property
    def norm_stats(self):
        return self._norm_stats

    @property
    def norm_stats_covs(self):
        return self._norm_stats_covs

    def build_batch(self, item_dict):
        """
        Build batch from item_dict, handling only available fields.
        """
        # Extract only the fields defined in TrainingItem that are present in item_dict
        fields = {k: v for k, v in item_dict.items() if k in TrainingItem._fields}
        return TrainingItem(**fields)

    def setup(self, stage='test'):
        """
        Setup datasets for train/val/test.
        Validates that preprocessed data contains the required variables.
        """
        build_batch = self.build_batch
        
        # Validate that preprocessed files contain required variables
        self._validate_preprocessed_data()
        
        self.train_ds = XrDatasetMultiRes_simplify(
            self.croscim_preproc_paths, 
            self.split_train, 
            self.multires, 
            build_batch,
            input_vars=self.input_vars
        )
        
        self.val_ds = XrDatasetMultiRes_simplify(
            self.croscim_preproc_paths,
            self.split_val, 
            self.multires, 
            build_batch,
            input_vars=self.input_vars
        )
        
        self.test_ds = XrDatasetMultiRes_simplify(
            self.croscim_preproc_paths,
            self.split_test, 
            self.multires, 
            build_batch,
            input_vars=self.input_vars
        )
        
        print(f"Datasets ready:")
        print(f"  Train: {len(self.train_ds)} samples")
        print(f"  Val: {len(self.val_ds)} samples")
        print(f"  Test: {len(self.test_ds)} samples")

    def _validate_preprocessed_data(self):
        """
        Validate that preprocessed data files contain the required variables.
        """
        for res in self.multires:
            path_key = f"patch_x{res}"
            if path_key not in self.croscim_preproc_paths:
                raise ValueError(f"Missing path for {path_key} in croscim_preproc_paths")
            
            # Open one file to check variables
            try:
                ds = xr.open_dataset(self.croscim_preproc_paths[path_key])
                available_vars = list(ds.data_vars)
                
                # Check if all required input vars are present
                missing_vars = []
                for var in self.input_vars:
                    if var not in available_vars:
                        missing_vars.append(var)

                # Get resolution-specific target_vars
                if isinstance(self.target_vars, dict) and path_key in self.target_vars:
                    target_vars = self.target_vars[path_key]
                else:
                    # Fallback to base target_vars (for backward compatibility)
                    target_vars = self.target_vars

                # Check if all target vars are present
                for var in target_vars:
                    if var not in available_vars:
                        missing_vars.append(var)
                
                if missing_vars:
                    print(f"\n Warning for {path_key}:")
                    print(f"  Missing variables: {missing_vars}")
                    print(f"  Available variables: {available_vars}")
                    print(f"  This may cause errors during training!")
                else:
                    print(f"{path_key}: All required variables present")
                
                ds.close()
                
            except Exception as e:
                print(f" Error validating {path_key}: {e}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, 
            shuffle=True, 
            batch_size=2,
            num_workers=5, 
            persistent_workers=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, 
            shuffle=False, 
            batch_size=2,
            num_workers=5, 
            persistent_workers=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, 
            shuffle=False, 
            batch_size=2,
            num_workers=5, 
            persistent_workers=True
        )