"""
Simple multi-resolution dataset with models_vars support.
Minimal adaptation of data_simple_multires.py to include models data as input variables.
"""

from random import sample
import contrib
from contrib.CROSCIM.load_data import *
from contrib.CROSCIM.dataloaders.data_simple import *
from contrib.CROSCIM.dataloaders.data_simple_multires import *
import datetime
import pyresample
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature 
import shapely.geometry as sgeom
import os
import numpy as np
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import torch

# Create TrainingItem at module level with default config
def create_training_item(satellite_vars, covariates, target_vars, models_vars=None):
    """
    Dynamically create a TrainingItem namedtuple with fields for:
    - All satellite variables (e.g., 'asip_sic', 'cimr_SIC', etc.)
    - All model variables (e.g., 'models_SIC', 'models_SIT')  #  NEW
    - All target variables (e.g., 'tgt_sic', 'tgt_SIT')
    - All covariates
    - Coordinates and mask
    """
    fields = []
    
    # Satellite variables (e.g., 'asip_sic', 'cimr_SIC', 'cristal_SIT')
    # Exclude 'models' from satellite_vars to avoid duplicates
    for source, vars_list in satellite_vars.items():
        if source == 'models':  #  Skip models here
            continue
        for var in vars_list:
            fields.append(f"{source}_{var}")
    
    # Model variables (e.g., 'models_SIC', 'models_SIT')
    # These are added separately to avoid duplicates
    if models_vars:
        for var in models_vars:
            fields.append(f"models_{var}")
    
    # Target variables (e.g., 'tgt_sic', 'tgt_SIT', or 'models_SIC', 'models_SIT')
    # Target vars might include model vars, so we need to deduplicate
    for var in target_vars:
        if var not in fields:
            fields.append(var)
    
    # Covariates (e.g., 'msl', 't2m', 'u10', 'v10')
    fields.extend(covariates)
    
    # Coordinates and mask
    fields.extend(['lat', 'lon', 'land_mask', 'time', 'yc', 'xc'])
    
    # Remove any remaining duplicates while preserving order
    seen = set()
    fields = [x for x in fields if not (x in seen or seen.add(x))]
    
    # Debug: print fields to verify no duplicates
    if len(fields) != len(set(fields)):
        duplicates = [f for f in fields if fields.count(f) > 1]
        raise ValueError(f"Duplicate fields detected: {set(duplicates)}")
    
    # Create namedtuple with custom __reduce__ for pickling
    TrainingItemBase = namedtuple("TrainingItem", fields)
    
    class PicklableTrainingItem(TrainingItemBase):
        """TrainingItem with pickling support."""
        def __reduce__(self):
            # Store parameters to recreate the class
            return (
                _rebuild_training_item,
                (satellite_vars, covariates, target_vars, models_vars, tuple(self))
            )
    
    return PicklableTrainingItem

def _rebuild_training_item(satellite_vars, covariates, target_vars, models_vars, values):
    """Helper function to rebuild TrainingItem during unpickling."""
    TrainingItemClass = create_training_item(satellite_vars, covariates, target_vars, models_vars)
    return TrainingItemClass(*values)


class XrDatasetMultiResSupervised_simplify(XrDatasetMultiRes_simplify):
    """
    Custom dataset that passes resolution key to build_batch for var_mapping.
    """
    
    def __getitem__(self, idx):
        out = {}
        for res in self.multires:
            res_key = f"patch_x{res}"
            item = self.db[res_key].isel(record=idx, sample=0)
            
            # Only select variables that exist in TrainingItem
            available_fields = [f for f in TrainingItem._fields if f in item.data_vars or f in item.coords]
            item = item[available_fields]
            
            var_dict = {var: item[var].values for var in item.data_vars}
            var_dict["time"] = item.time.data
            var_dict["xc"] = item.xc.data
            var_dict["yc"] = item.yc.data
            
            # Pass res_key to build_batch for var_mapping lookup
            item = self.build_batch(var_dict, res_key=res_key, add_noise=self.add_noise)
            out[res_key] = item
        
        return out


class BaseDataModuleMultiResSupervised_simplify(pl.LightningDataModule):
    """
    Simple multi-resolution datamodule with models support.
    Just extends BaseDataModuleMultiRes_simplify to handle models_vars.
    """
    
    def __init__(self, 
                 croscim_preproc_paths,
                 multires,
                 split_train,
                 split_val,
                 split_test,
                 norm_stats,
                 norm_stats_covs,
                 noise_level,
                 satellite_vars=None,
                 models_vars=None,      # NEW: models variables
                 covariates=None,
                 target_vars=None,
                 var_mapping=None,      # NEW: var_mapping for target initialization
                 add_noise=False,  
                 **kwargs):

        super().__init__()
        self.croscim_preproc_paths = croscim_preproc_paths
        self.multires = multires
        self.split_train = split_train
        self.split_val = split_val
        self.split_test = split_test
        self._norm_stats = norm_stats
        self._norm_stats_covs = norm_stats_covs
        self.add_noise = add_noise
        self.noise_level = noise_level
        
        # Store variable configuration
        self.satellite_vars = satellite_vars or DEFAULT_VAR_GROUPS
        self.models_vars = models_vars or []  # NEW: models variables list
        self.covariates = covariates or DEFAULT_COVARIATES
        
        # Convert target_vars to dict (handle OmegaConf)
        from omegaconf import OmegaConf
        if target_vars is not None:
            if hasattr(target_vars, '_metadata'):
                self.target_vars = OmegaConf.to_container(target_vars, resolve=True)
            else:
                self.target_vars = target_vars
        else:
            self.target_vars = []

        # Convert var_mapping to dict (handle OmegaConf)
        if var_mapping is not None:
            if hasattr(var_mapping, '_metadata'):
                self.var_mapping = OmegaConf.to_container(var_mapping, resolve=True)
            else:
                self.var_mapping = dict(var_mapping)
        else:
            self.var_mapping = {}

        # Get all unique target vars (flatten if resolution-specific)
        all_target_vars = self._get_all_target_vars()

        global TrainingItem
        TrainingItem = create_training_item(
            satellite_vars=self.satellite_vars,
            covariates=self.covariates,
            target_vars=all_target_vars,  # Use flattened list
            models_vars=self.models_vars
        )
        
        print(TrainingItem._fields)  # Debug: print TrainingItem fields
        # Construct input_vars automatically (now includes models)
        self.input_vars = self._construct_input_vars()
        
        # Determine which sources are actually needed
        self.active_sources = [src for src, vars in self.satellite_vars.items() if vars]
        if self.models_vars:  # NEW: add models to active sources if present
            self.active_sources.append('models')
        
        print(f"\n{'='*60}")
        print(f"DataModule configuration:")
        print(f"{'='*60}")
        print(f"  Satellite vars: {self.satellite_vars}")
        print(f"  Models vars: {self.models_vars}")  # NEW
        print(f"  Active sources: {self.active_sources}")
        print(f"  Covariates: {self.covariates}")
        print(f"  Target vars: {self.target_vars}")
        print(f"  Input vars: {self.input_vars}")
        print(f"  Multires: {self.multires}")
        print(f"{'='*60}\n")

    def _get_all_target_vars(self):
        """
        Get all unique target variables across all resolutions.
        If target_vars is a dict, concatenate all values and deduplicate.
        If target_vars is a list, return as-is.
        """
        if isinstance(self.target_vars, dict):
            # Flatten all resolution-specific target vars
            all_vars = []
            for res_vars in self.target_vars.values():
                all_vars.extend(res_vars)
            # Remove duplicates while preserving order
            return list(dict.fromkeys(all_vars))
        else:
            # Already a simple list
            return self.target_vars

    def _get_target_vars_for_resolution(self, res):
        """
        Get target variables for a specific resolution.
        Args:
            res: resolution number (e.g., 50 for patch_x50)
        Returns:
            List of target variable names for this resolution
        """
        res_key = f"patch_x{res}"
        if isinstance(self.target_vars, dict):
            return self.target_vars.get(res_key, [])
        else:
            return self.target_vars

    def _construct_input_vars(self):
        """
        Construct input variable names from satellite_vars + models_vars + covariates.
        NEW: includes models_vars with 'models_' prefix
        """
        input_vars = []
        
        # Add satellite variables with source prefix
        for source, vars in self.satellite_vars.items():
            for var in vars:
                input_vars.append(f"{source}_{var}")
        
        # NEW: Add models variables with 'models_' prefix
        for var in self.models_vars:
            input_vars.append(f"models_{var}")
        
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

    def build_batch(self, item_dict, res_key=None, add_noise=False):
        """
        Build batch from item_dict, handling only available fields.
        Also handles target variable mapping based on resolution.
        
        Args:
            item_dict: Dictionary with variable data
            res_key: Resolution key (e.g., 'patch_x50') for var_mapping lookup
        """
        # Get resolution-specific mapping if available
        if res_key and isinstance(self.var_mapping, dict) and res_key in self.var_mapping:
            mapping = self.var_mapping[res_key]
        else:
            # Fallback to base var_mapping (for backward compatibility)
            mapping = self.var_mapping if isinstance(self.var_mapping, dict) else {}
        
        # Get resolution-specific target_vars
        if isinstance(self.target_vars, dict) and res_key in self.target_vars:
            target_vars = self.target_vars[res_key]
        else:
            # Fallback to base target_vars (for backward compatibility)
            target_vars = self.target_vars

        # Initialize targets from their sources based on mapping
        for target_var in target_vars:
            if target_var not in item_dict:  # Only map if target doesn't exist
                source_var = mapping.get(target_var)
                if source_var and source_var in item_dict:
                    item_dict[target_var] = item_dict[source_var].copy()
        
        # Add noise to satellite variables if requested
        if add_noise:
            print('ADD NOISE TO SATELLITE VARIABLES')
            #cimr SIC
            if 'cimr_SIC' in item_dict:
                stats = self.norm_stats['cimr']['SIC']
                item_norm = item_dict['cimr_SIC']
                item_dict['cimr_SIC'] = np.clip(
                    item_norm * (1 + self.noise_level['cimr_SIC'] * np.random.randn(*item_norm.shape)),
                    a_min=stats['min'], a_max=stats['max']
                )
            #cimr SIT
            if 'cimr_SIT' in item_dict:
                stats = self.norm_stats['cimr']['SIT']
                item_norm = item_dict['cimr_SIT']
                item_phys = item_norm * stats['std'] + stats['mean']
                noise_phys = self.noise_level['cimr_SIT'] * item_phys * np.random.randn(*item_phys.shape)
                item_dict['cimr_SIT'] = (item_phys + noise_phys - stats['mean']) / stats['std']

            #cristal SIT
            if 'cristal_SIT' in item_dict:
                stats = self.norm_stats['cristal']['SIT']
                noise_norm = self.noise_level['cristal_SIT']/stats['std']
                item_norm = item_dict['cristal_SIT']
                item_dict['cristal_SIT'] = item_norm + noise_norm * np.random.randn(*item_norm.shape)

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
        
        self.train_ds = XrDatasetMultiResSupervised_simplify(
            self.croscim_preproc_paths, 
            self.split_train, 
            self.multires, 
            build_batch,
            input_vars=self.input_vars,
            add_noise=self.add_noise
        )
        
        self.val_ds = XrDatasetMultiResSupervised_simplify(
            self.croscim_preproc_paths,
            self.split_val, 
            self.multires, 
            build_batch,
            input_vars=self.input_vars,
            add_noise=self.add_noise
        )
        
        self.test_ds = XrDatasetMultiResSupervised_simplify(
            self.croscim_preproc_paths,
            self.split_test, 
            self.multires, 
            build_batch,
            input_vars=self.input_vars,
            add_noise=self.add_noise    
        )
        
        print(f"Datasets ready:")
        print(f"  Train: {len(self.train_ds)} samples")
        print(f"  Val: {len(self.val_ds)} samples")
        print(f"  Test: {len(self.test_ds)} samples")

    def _validate_preprocessed_data(self):
        """
        Validate that preprocessed data files contain the required variables.
        NEW: Also checks for models_vars
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
                    print(f"\n  Warning for {path_key}:")
                    print(f"  Missing variables: {missing_vars}")
                    print(f"  Available variables: {available_vars}")
                    print(f"  This may cause errors during training!")
                else:
                    print(f"✓ {path_key}: All required variables present")
                
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