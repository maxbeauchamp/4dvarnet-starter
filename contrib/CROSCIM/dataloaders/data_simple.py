import pytorch_lightning as pl
import numpy as np
import torch.utils.data
import torch
import xarray as xr
import itertools
import functools as ft
import tqdm
from collections import namedtuple
from torch.utils.data import ConcatDataset
import multiprocessing
import gc
from random import sample
import contrib
from contrib.CROSCIM.dataloaders.load_data import *
from contrib.CROSCIM.dataloaders.data import *
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

class XrDataset_simplify(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    
    def __init__(self, path, split, build_batch, training_item_class=None):
        'Initialization'
        self.db = xr.open_dataset(path).isel(record=split)
        self.build_batch = build_batch
        self.training_item_class = training_item_class or TrainingItem

    def __len__(self):
        'Denotes the total number of samples'
        # Use first available variable instead of hardcoded 'asip_sic'
        first_var = list(self.db.data_vars)[0]
        return len(self.db[first_var])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        item = self.db.isel(record=idx, sample=0)
        
        # Filter to only include fields that exist in both dataset and TrainingItem
        available_fields = [f for f in self.training_item_class._fields 
                          if f in item.data_vars or f in item.coords]
        
        item = item[available_fields]
        
        # Build dictionary with data variables
        var_dict = {var: item[var].values for var in item.data_vars}
        
        # Add coordinates
        var_dict["time"] = item.time.data
        var_dict["xc"] = item.xc.data
        var_dict["yc"] = item.yc.data
        
        # Build and return batch
        item = self.build_batch(var_dict)
        
        return item


class BaseDataModule_simplify(pl.LightningDataModule):
    def __init__(self,
                 croscim_preproc_path,
                 split_train,
                 split_val,
                 split_test,
                 norm_stats,
                 norm_stats_covs,
                 satellite_vars=None,  # NEW: From config
                 covariates=None,      # NEW: From config
                 target_vars=None,     # NEW: From config
                 **kwargs):

        super().__init__()
        self.croscim_preproc_path = croscim_preproc_path
        self.split_train = split_train
        self.split_val = split_val
        self.split_test = split_test
        self._norm_stats = norm_stats
        self._norm_stats_covs = norm_stats_covs
        
        # Store variable configuration
        self.satellite_vars = satellite_vars or DEFAULT_VAR_GROUPS
        self.covariates = covariates or DEFAULT_COVARIATES
        self.target_vars = target_vars or ["tgt_sic", "tgt_SIT"]
        
        # Create custom TrainingItem class based on config
        self.training_item_class = create_training_item(
            satellite_vars=self.satellite_vars,
            covariates=self.covariates,
            target_vars=self.target_vars
        )
        
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
        # Extract only the fields defined in the custom TrainingItem
        fields = {k: v for k, v in item_dict.items() if k in self.training_item_class._fields}
        return self.training_item_class(**fields)

    def setup(self, stage='test'):
        """
        Setup datasets for train/val/test.
        Validates that preprocessed data contains the required variables.
        """
        build_batch = self.build_batch
        
        # Validate preprocessed data
        self._validate_preprocessed_data()
        
        self.train_ds = XrDataset_simplify(
            self.croscim_preproc_path,
            self.split_train,
            build_batch,
            training_item_class=self.training_item_class
        )
        
        self.val_ds = XrDataset_simplify(
            self.croscim_preproc_path,
            self.split_val,
            build_batch,
            training_item_class=self.training_item_class
        )
        
        self.test_ds = XrDataset_simplify(
            self.croscim_preproc_path,
            self.split_test,
            build_batch,
            training_item_class=self.training_item_class
        )
        
        print(f"Datasets ready:")
        print(f"  Train: {len(self.train_ds)} samples")
        print(f"  Val: {len(self.val_ds)} samples")
        print(f"  Test: {len(self.test_ds)} samples")

    def _validate_preprocessed_data(self):
        """
        Validate that preprocessed data file contains the required variables.
        """
        try:
            ds = xr.open_dataset(self.croscim_preproc_path)
            available_vars = list(ds.data_vars)
            
            # Check if all required input vars are present
            missing_vars = []
            for var in self.input_vars:
                if var not in available_vars:
                    missing_vars.append(var)
            
            # Check if all target vars are present
            for var in self.target_vars:
                if var not in available_vars:
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"\n⚠️  Warning:")
                print(f"  Missing variables: {missing_vars}")
                print(f"  Available variables: {available_vars}")
                print(f"  This may cause errors during training!")
            else:
                print(f"✅ All required variables present in {self.croscim_preproc_path}")
            
            ds.close()
            
        except Exception as e:
            print(f"❌ Error validating preprocessed data: {e}")

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