import pytorch_lightning as pl
import numpy as np
import torch.utils.data
import torch
import xarray as xr
import itertools
import functools as ft
import tqdm
from collections import namedtuple
from torch.utils.data import  ConcatDataset
import multiprocessing
import gc
from random import sample 
import contrib
from contrib.CROSCIM.dataloaders.load_data import *
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
import cartopy
from cartopy.io.shapereader import Reader
cartopy.config['pre_existing_data_dir'] = os.path.abspath('contrib/CROSCIM')

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
    
    # Ensure each models_XXX has a corresponding tgt_XXX field
    if models_vars:
        for var in models_vars:
            tgt_var = f"tgt_{var}"
            if tgt_var not in fields:
                fields.append(tgt_var)
    
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


# Create TrainingItem at MODULE LEVEL with default configuration
# This makes it picklable for multiprocessing
TrainingItem = create_training_item(
    satellite_vars=DEFAULT_VAR_GROUPS, 
    covariates=DEFAULT_COVARIATES,
    target_vars=["tgt_sic", "tgt_SIT"],
    models_vars=["SIC", "SIT", "HS", "SSH"]  # Models handled separately - includes all possible model vars
)

class IncompleteScanConfiguration(Exception):
    pass

class DangerousDimOrdering(Exception):
    pass


def find_idx(coords,c):
    return np.where(coords==c)[0][0]

def pad_batch_with_coords(ds, sl, global_xc, global_yc, global_lon, global_lat):
    """
    Pads an xarray Dataset `ds` so that its yc/xc match a window from the global coords.
    Missing values are NaN-filled, and lon/lat are taken from the global reference.

    Parameters
    ----------
    ds : xr.Dataset
        Input patch (must have coords 'xc' and 'yc').
    sl: slices for coordinates
    global_xc, global_yc : 1D array-like
        Full-resolution reference coordinates for xc and yc.
    global_lon, global_lat : 2D array-like
        Reference longitude and latitude on (yc, xc) grid.

    Returns
    -------
    ds_padded : xr.Dataset
        Dataset aligned on the padded coords with NaN padding.
    """

    ix = [find_idx(global_xc, x) for x in global_xc[sl["xc"].start:sl["xc"].stop]]
    iy = [find_idx(global_yc, y) for y in global_yc[sl["yc"].start:sl["yc"].stop]]
    
    # Create padded coordinate window from global arrays
    padded_coords = {
        "time": ds.time,
        "xc": global_xc[sl["xc"].start:sl["xc"].stop],
        "yc": global_yc[sl["yc"].start:sl["yc"].stop],
        "lon": (["yc", "xc"], global_lon[iy[0]: iy[-1] + 1, ix[0]: ix[-1] + 1]),
        "lat": (["yc", "xc"], global_lat[iy[0]: iy[-1] + 1, ix[0]: ix[-1] + 1]),
    }

    # Create a template Dataset with the padded coords
    padded_template = xr.Dataset(coords=padded_coords)

    # Align → ensures missing coords in ds become NaNs
    _, ds_padded = xr.align(padded_template, ds, join="left")

    return ds_padded

class XrDataset(torch.utils.data.Dataset):

    def __init__(self, asip_paths, cimr_paths, cristal_paths,
                 covariates_paths, covariates, 
                 target_vars,
                 satellite_vars=None,  # NEW: satellite variable config
                 var_mapping=None,     # NEW: mapping config
                 mask=None, times=None,
                 patch_dims=None, domain_limits=None, strides=None,
                 strides_test=None, postpro_fn=None,
                 resize=1, res=500, pad=False, stride_test=False,
                 subsel_patch=False, subsel_patch_path=None,
                 itrp_from_regular=True,
                 load_data=False, domain=None):

        super().__init__()
        
        # Store variable configuration
        self.satellite_vars = satellite_vars or DEFAULT_VAR_GROUPS
        self.covariates = covariates or DEFAULT_COVARIATES
        self.target_vars = target_vars
        self.var_mapping = var_mapping or {}

        # Determine active sources
        self.active_sources = [src for src, vars in self.satellite_vars.items() if vars]
        
        # Store paths only for active sources
        if 'asip' in self.active_sources:
            self.asip_paths = asip_paths
        if 'cimr' in self.active_sources:
            self.cimr_paths = cimr_paths
        if 'cristal' in self.active_sources:
            self.cristal_paths = cristal_paths
        
        self.covariates_paths = covariates_paths
        self.postpro_fn = postpro_fn
        self.mask = mask.sel(**(domain_limits or {}))
        self.times = times
        self.patch_dims = patch_dims
        self.strides = strides or {}
        if stride_test:
            self.strides = strides_test or {}
        self.domain_limits = domain_limits
        self.res = res * resize
        self.pad = pad
        self.subsel_patch = subsel_patch
        self.subsel_patch_path = subsel_patch_path
        self.itrp_from_regular = itrp_from_regular
        self.load_data = load_data
        self.domain = domain
        self.resize = resize
        
        # Use first available ASIP file as reference (ASIP should always be present)
        if 'asip' not in self.active_sources:
            raise ValueError("ASIP is required as reference grid")
            
        asip_base = xr.open_dataset(self.asip_paths[0]).sel(**(domain_limits or {}))
        
        if self.resize != 1:
            print(f"Coarsening target data by factor {resize}")
            asip_base = fast_coarsen_xr(asip_base, factor_x=resize, factor_y=resize)
            self.mask = fast_coarsen_xr_array(self.mask, factor_x=resize, factor_y=resize, 
                                              mode="binary")
        
        self.xc = asip_base.xc.data
        self.yc = asip_base.yc.data
        self.lon = asip_base.lon.data
        self.lat = asip_base.lat.data

        # Load data in memory (for inference) - only active sources
        if self.load_data:
            time_slice = slice(
                datetime.datetime.strftime(self.times[0], "%Y-%m-%d"),
                datetime.datetime.strftime(self.times[-1] + datetime.timedelta(days=1), "%Y-%m-%d")
            )
            
            # Use load_mfdata which returns a dict
            # Build paths_loaders with only active sources
            paths_loaders = {}
            if 'asip' in self.active_sources:
                paths_loaders['asip'] = self.asip_paths
            if 'cimr' in self.active_sources:
                paths_loaders['cimr'] = self.cimr_paths
            if 'cristal' in self.active_sources:
                paths_loaders['cristal'] = self.cristal_paths
            if self.covariates:
                paths_loaders['covariates'] = self.covariates_paths
            datasets = load_mfdata(
                times=time_slice,
                satellite_vars=self.satellite_vars,  # Dict of {source: [vars]}
                covariates=self.covariates,  # List of covariate names
                slices=self.domain_limits,
                path_loaders=paths_loaders,
                type_coords="coords",
                resize=self.resize,
                domain_limits=self.domain_limits
            )
            
            # Extract datasets from the returned dict
            self.full_asip = datasets.get('asip', None)
            self.full_cimr = datasets.get('cimr', None)
            self.full_cristal = datasets.get('cristal', None)
            self.full_covs = datasets.get('covariates', None)
            
            # Validate that ASIP is loaded (required as reference)
            if self.full_asip is None:
                raise ValueError("ASIP dataset is required as reference grid but was not loaded")

        # padding
        if self.pad:
            pad_x = self._find_pad(self.patch_dims['xc'], self.strides['xc'], len(self.xc))
            pad_y = self._find_pad(self.patch_dims['yc'], self.strides['yc'], len(self.yc))
            self.lon = np.pad(self.lon, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])), mode="edge")
            self.lat = np.pad(self.lat, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])), mode="edge")
            self.xc = np.linspace(self.xc[0] - pad_x[0]*self.res, self.xc[-1] + pad_x[1]*self.res, len(self.xc)+sum(pad_x))
            self.yc = np.linspace(self.yc[0] + pad_y[0]*self.res, self.yc[-1] - pad_y[1]*self.res, len(self.yc)+sum(pad_y))

        nt, ny, nx = (len(self.times), len(self.yc), len(self.xc))
        self.da_dims = dict(time=nt, yc=ny, xc=nx)
        self.ds_size = {
            dim: max((self.da_dims[dim] - self.patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in self.patch_dims
        }

        # get patches in ocean
        if self.subsel_patch:
            if not os.path.isfile(self.subsel_patch_path):
                idx0 = self.find_patches_in_ocean()
                print("Saving ocean patches in "+subsel_patch_path)
                np.savetxt(self.subsel_patch_path, idx0, fmt='%i')
            else:
                idx0 = np.loadtxt(self.subsel_patch_path).astype(int)
            nitem_bytime = np.prod([self.ds_size[dim] for dim in self.ds_size if dim != 'time'])
            self.idx_patches_in_ocean = np.concatenate([idx0 + (nitem_bytime * t) for t in range(self.ds_size['time'])])

    def _find_pad(self, sl, st, N):
        k = np.floor(N/st)
        if N>((k*st)+(sl-st)):
            pad = (k+1)*st + (sl-st) - N
        elif N<((k*st)+(sl-st)):
            pad = (k*st) + (sl-st) - N
        else:
            pad = 0
        return int(pad/2), int(pad-int(pad/2))
    
    def __len__(self):
        size = 1
        if self.subsel_patch:
            size = len(self.idx_patches_in_ocean)
        else:
            for v in self.ds_size.values():
                size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self, limit=None):
        coords_list = []
        if limit is None:
            indices = range(len(self))
        else:
            indices = np.random.choice(len(self), size=limit, replace=False)

        for idx in indices:
            if self.subsel_patch:
                idx0 = self.idx_patches_in_ocean[idx]
            else:
                idx0 = idx
            sl = {
                dim: slice(self.strides.get(dim, 1) * idx_dim,
                           self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
                for dim, idx_dim in zip(self.ds_size.keys(), np.unravel_index(idx0, tuple(self.ds_size.values())))
            }
            coords = xr.Dataset(coords=dict(
                xc=self.xc[sl["xc"].start:sl["xc"].stop],
                yc=self.yc[sl["yc"].start:sl["yc"].stop],
                time=self.times[sl["time"].start:sl["time"].stop],
                lon=(["yc", "xc"], self.lon[sl["yc"], sl["xc"]]),
                lat=(["yc", "xc"], self.lat[sl["yc"], sl["xc"]]),
            )).transpose("time", "yc", "xc")
            coords_list.append(coords)
        return coords_list

    def find_patches_in_ocean(self):
        nitem_bytime = np.prod([self.ds_size[dim] for dim in self.ds_size if dim != 'time'])
        idx_ocean = []
        for i in range(nitem_bytime):
            if np.mod(i,1000)==0:
                print(i)
            sl = {
                dim: slice(self.strides.get(dim, 1) * idx_dim,
                           self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
                for dim, idx_dim in zip([d for d in self.ds_size if d != 'time'],
                                        np.unravel_index(i, tuple(self.ds_size[dim] for dim in self.ds_size if dim != 'time')))
            }
            mask_patch = self.mask.isel(xc=sl['xc'], yc=sl['yc']).values
            if np.any(mask_patch == 0):
                idx_ocean.append(i)
        return np.array(idx_ocean)

    def interpolate_dataset(self, target_grid, ds, var_list, prefix=None):
        """
        Interpolates variables from ds onto the target grid.
    
        Args:
            target_grid: either a tuple (xc, yc) for regular grid
                         or a pyresample SwathDefinition for irregular grid
            ds: xarray.Dataset with variables to interpolate
            var_list: list of variable names to interpolate
            prefix: optional prefix for output keys
    
        Returns:
            dict of interpolated numpy arrays (shape: [time, yc, xc])
        """
        data_out = {}
    
        use_regular_grid = isinstance(target_grid, tuple) and len(target_grid) == 2
        isel_time = ds.sizes["time"]
    
        for var in var_list:
            if var not in ds:
                continue
    
            key = f"{prefix}_{var}" if prefix is not None else var
    
            if use_regular_grid:
                # Regular grid interpolation
                xc_target, yc_target = target_grid
                interpolated = ds[var].interp(xc=("xc", xc_target), yc=("yc", yc_target))
                data_out[key] = interpolated.values
            else:
                # Irregular grid using pyresample
                swath_def_target = target_grid
                src_def = pyresample.geometry.SwathDefinition(lons=ds.lon.values, lats=ds.lat.values)
                interpolated = np.stack([
                    pyresample.kd_tree.resample_nearest(
                        src_def,
                        ds[var].isel(time=i).values,
                        swath_def_target,
                        radius_of_influence=30000,
                        fill_value=np.nan
                    ) for i in range(isel_time)
                ])
                data_out[key] = interpolated

        return data_out

    def __getitem__(self, idx):
            
        if self.subsel_patch:
            idx = self.idx_patches_in_ocean[idx]

        sl = {
            dim: slice(self.strides.get(dim, 1) * idx_dim,
                    self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
            for dim, idx_dim in zip(self.ds_size.keys(), np.unravel_index(idx, tuple(self.ds_size.values())))
        }

        t_idx = sl["time"].start
        xc_slice = sl["xc"]
        yc_slice = sl["yc"]

        item_mask = self.mask.sel(xc=slice(self.xc[sl["xc"].start], self.xc[sl["xc"].stop-1]),
                                yc=slice(self.yc[sl["yc"].start], self.yc[sl["yc"].stop-1])).values

        # Loading datasets - only active sources
        if self.load_data:
            datasets = {}
            if 'asip' in self.active_sources:
                datasets['asip'] = self.full_asip.isel(time=sl["time"]).sel(
                    xc=slice(self.xc[sl["xc"].start], self.xc[sl["xc"].stop-1]),
                    yc=slice(self.yc[sl["yc"].start], self.yc[sl["yc"].stop-1])
                )
            if 'cimr' in self.active_sources:
                datasets['cimr'] = self.full_cimr.isel(time=sl["time"])
            if 'cristal' in self.active_sources:
                datasets['cristal'] = self.full_cristal.isel(time=sl["time"])
            if self.covariates:
                datasets['covariates'] = self.full_covs.isel(time=sl["time"])
        else:
            time_indices = np.arange(sl["time"].start, sl["time"].stop)
            if self.resize == 1:
                slices = {"xc": sl["xc"], "yc": sl["yc"]}
                type_coords = "index"
            else:
                slices = {
                    "xc": slice(self.xc[sl["xc"].start], self.xc[sl["xc"].stop]), 
                    "yc": slice(self.yc[sl["yc"].start], self.yc[sl["yc"].stop])
                }
                type_coords = "coords"
            
            datasets = {}
            if 'asip' in self.active_sources:
                datasets['asip'] = concatenate(
                    self.asip_paths[time_indices], 
                    var_list=self.satellite_vars['asip'],
                    slices=slices, 
                    type_coords=type_coords, 
                    resize=self.resize,
                    domain_limits=self.domain_limits
                )
            if 'cimr' in self.active_sources:
                datasets['cimr'] = concatenate(
                    self.cimr_paths[time_indices], 
                    var_list=self.satellite_vars['cimr'], 
                    slices=None
                )
            if 'cristal' in self.active_sources:
                datasets['cristal'] = concatenate(
                    self.cristal_paths[time_indices], 
                    var_list=self.satellite_vars['cristal'], 
                    slices=None
                )
            if self.covariates:
                datasets['covariates'] = concatenate(
                    self.covariates_paths[time_indices], 
                    var_list=self.covariates, 
                    slices=None
                )

        # Get ASIP dataset (must exist)
        asip_ds = datasets.get('asip')
        if asip_ds is None:
            raise ValueError("ASIP dataset is required but not loaded")

        # Padding if necessary
        expected_shape = (self.patch_dims['time'], self.patch_dims['yc'], self.patch_dims['xc'])
        first_asip_var = self.satellite_vars['asip'][0]
        actual_shape = asip_ds[first_asip_var].shape

        asip_ds = asip_ds.update({"mask": (("yc", "xc"), item_mask)})
        if actual_shape != expected_shape:
            ix = [find_idx(self.xc, x) for x in self.xc[sl["xc"].start:sl["xc"].stop]]
            iy = [find_idx(self.yc, y) for y in self.yc[sl["yc"].start:sl["yc"].stop]]
            padded_patch = xr.Dataset(
                coords={
                    "time": asip_ds.time,
                    "xc": self.xc[sl["xc"].start:sl["xc"].stop],
                    "yc": self.yc[sl["yc"].start:sl["yc"].stop],
                    "lon": (["yc", "xc"], self.lon[iy[0]:(iy[-1]+1), ix[0]:(ix[-1]+1)]),
                    "lat": (["yc", "xc"], self.lat[iy[0]:(iy[-1]+1), ix[0]:(ix[-1]+1)])
                }
            )
            asip_ds = xr.align(padded_patch, asip_ds, join="left")[1]
            asip_ds['mask'] = asip_ds['mask'].fillna(1)
            item_mask = asip_ds.mask.data

        # Collect ASIP variables
        sample = {}
        for var in self.satellite_vars['asip']:
            if var in asip_ds:
                sample[f"asip_{var}"] = asip_ds[var].values

        lon_patch = asip_ds.lon.values
        lat_patch = asip_ds.lat.values

        # Interpolate other sources (only if active)
        if self.itrp_from_regular:
            target_grid = (asip_ds.xc.values, asip_ds.yc.values)
        else:
            target_grid = pyresample.geometry.SwathDefinition(lons=lon_patch, lats=lat_patch)
        
        if 'cimr' in datasets:
            cimr_vars = self.interpolate_dataset(target_grid, datasets['cimr'], 
                                                self.satellite_vars['cimr'], prefix="cimr")
            sample.update(cimr_vars)
        
        if 'cristal' in datasets:
            cristal_vars = self.interpolate_dataset(target_grid, datasets['cristal'], 
                                                    self.satellite_vars['cristal'], prefix="cristal")
            sample.update(cristal_vars)
        
        if 'covariates' in datasets:
            covariate_vars = self.interpolate_dataset(target_grid, datasets['covariates'], 
                                                    self.covariates)
            sample.update(covariate_vars)

        # Add metadata
        sample["land_mask"] = np.expand_dims(item_mask, axis=0)
        sample["lat"] = np.expand_dims(lat_patch, axis=0)
        sample["lon"] = np.expand_dims(lon_patch, axis=0)
        
        # Add target variables using var_mapping
        # If target_vars and var_mapping are resolution-specific dicts, use the minimum resolution
        if isinstance(self.target_vars, dict):
            # Get the minimum resolution key (e.g., patch_x2 from [patch_x50, patch_x10, patch_x2])
            res_keys = [int(k.split('_x')[-1]) for k in self.target_vars.keys() if '_x' in k]
            min_res = min(res_keys) if res_keys else list(self.target_vars.keys())[0]
            min_res_key = f"patch_x{min_res}" if isinstance(min_res, int) else min_res
            var_mapping_to_use = self.var_mapping.get(min_res_key, {}) if isinstance(self.var_mapping, dict) else {}
        else:
            # Simple list case
            var_mapping_to_use = self.var_mapping
        
        # Initialize targets from their sources (same logic as data_multires_supervised.py)
        for target_var, source_var in var_mapping_to_use.items():
            # If target starts with models_XXX, create tgt_XXX from models_XXX
            if target_var.startswith('models_') and '_' in target_var:
                suffix = target_var.split('_', 1)[1]  # Extract XXX from models_XXX
                tgt_var = f"tgt_{suffix}"
                if target_var in sample:
                    sample[tgt_var] = sample[target_var].copy()
            # If target starts with tgt_XXX, use the source_var directly
            elif target_var.startswith('tgt_') and source_var in sample:
                sample[target_var] = sample[source_var].copy()

        # Keep track of coordinates
        sample["time"] = np.expand_dims(
            np.array([np.datetime64(t, "s").astype('float64') for t in asip_ds.time.values]),
            axis=0
        )
        sample["xc"] = np.expand_dims(asip_ds.xc.values, axis=0)
        sample["yc"] = np.expand_dims(asip_ds.yc.values, axis=0)

        if self.postpro_fn is not None:
            sample = self.postpro_fn(sample)

        return sample

    def reconstruct(self, batches, index_time, weight=None):
        """
        takes as input a list of np.ndarray of dimensions (b, *, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims

        batches: list of torch tensor correspondin to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting 
        """

        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, index_time, weight)

    def reconstruct_from_items(self, items, index_time, weight=None):
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
            weight = np.expand_dims(weight, 0)

        nvars = items[0].shape[0]
        result_tensor = np.zeros((nvars, 1, self.da_dims['yc'], self.da_dims['xc']))
        count_tensor = np.zeros((nvars, 1, self.da_dims['yc'], self.da_dims['xc']))

        coords = self.get_coords()

        for idx, item in enumerate(items):
            c = coords[idx]
            iy = [np.where(self.yc == y)[0][0] for y in c.yc.values]
            ix = [np.where(self.xc == x)[0][0] for x in c.xc.values]
            result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += item * weight
            count_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += weight

        result_tensor /= np.maximum(count_tensor, 1e-6)
        result_da = xr.DataArray(
            result_tensor,
            dims=[f'v{i}' for i in range(nvars)] + ["time", "yc", "xc"],
            coords={
                "time": [self.times[index_time]],
                "xc": self.xc,
                "yc": self.yc,
                "lon": ("yc", "xc", self.lon),
                "lat": ("yc", "xc", self.lat)
            }
        )
        return result_da

class XrDatasetSupervised(XrDataset):
    """
    Extension of XrDataset with support for numerical model inputs.
    """
    
    def __init__(self, models_paths=None, models_vars=None, *args, **kwargs):
        """
        Args:
            models_paths: Paths to numerical model files
            models_vars: List of model variable names (e.g., ["SIC", "SIT"])
            *args, **kwargs: Passed to parent XrDataset
        """
        # Store models configuration BEFORE calling parent
        self.models_paths = models_paths if models_paths is not None else np.array([])
        self.models_vars = models_vars if models_vars is not None else []
        
        # Call parent __init__
        super().__init__(*args, **kwargs)
        
        # Add 'models' to active sources if configured
        if self.models_vars is not None and len(self.models_vars) > 0 and 'models' not in self.active_sources:
            self.active_sources.append('models')

        # Load data in memory (for inference) - only active sources
        if self.load_data:
            time_slice = slice(
                datetime.datetime.strftime(self.times[0], "%Y-%m-%d"),
                datetime.datetime.strftime(self.times[-1] + datetime.timedelta(days=1), "%Y-%m-%d")
            )
            
            # Use load_mfdata which returns a dict
            # Build paths_loaders with only active sources
            paths_loaders = {}
            if 'asip' in self.active_sources:
                paths_loaders['asip'] = self.asip_paths
            if 'cimr' in self.active_sources:
                paths_loaders['cimr'] = self.cimr_paths
            if 'cristal' in self.active_sources:
                paths_loaders['cristal'] = self.cristal_paths
            if self.covariates:
                paths_loaders['covariates'] = self.covariates_paths
            if 'models' in self.active_sources:
                paths_loaders['models'] = self.models_paths
            datasets = load_mfdata(
                times=time_slice,
                satellite_vars=self.satellite_vars,  # Dict of {source: [vars]}
                covariates=self.covariates,  # List of covariate names
                models_vars=self.models_vars,  # List of model variable names
                slices=self.domain_limits,
                path_loaders=paths_loaders,
                type_coords="coords",
                resize=self.resize,
                domain_limits=self.domain_limits
            )
            
            # Extract datasets from the returned dict
            self.full_asip = datasets.get('asip', None)
            self.full_cimr = datasets.get('cimr', None)
            self.full_cristal = datasets.get('cristal', None)
            self.full_covs = datasets.get('covariates', None)
            self.full_models = datasets.get('models', None)
            
            # Validate that ASIP is loaded (required as reference)
            if self.full_asip is None:
                raise ValueError("ASIP dataset is required as reference grid but was not loaded")

        
        print(f"XrDatasetSupervised initialized:")
        print(f"  Models vars: {self.models_vars}")
        print(f"  Models paths: {len(self.models_paths)} files")
        print(f"  Active sources: {self.active_sources}")
    
    def __getitem__(self, idx):
        """Override to add model data loading and interpolation."""
        
        #  Call parent's __getitem__ to get the sample (returns a dict, not TrainingItem yet)
        # The parent returns a dict before applying postpro_fn
        # We need to intercept BEFORE postpro_fn converts it to TrainingItem
        
        # Save the original postpro_fn
        original_postpro_fn = self.postpro_fn
        
        # Temporarily disable postpro_fn to get raw dict
        self.postpro_fn = None
        
        # Get raw sample dict from parent
        sample = super().__getitem__(idx)
        
        # Restore postpro_fn
        self.postpro_fn = original_postpro_fn
        
        #  If no model data configured, apply postpro and return
        if self.models_vars is None:
            if self.postpro_fn is not None:
                sample = self.postpro_fn(sample)
            return sample
        
        #  Calculate the actual idx used by parent (after subsel_patch)
        actual_idx = self.idx_patches_in_ocean[idx] if self.subsel_patch else idx
        
        # Get the slice for this patch
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx_dim,
                      self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
            for dim, idx_dim in zip(self.ds_size.keys(), np.unravel_index(actual_idx, tuple(self.ds_size.values())))
        }
        
        time_indices = np.arange(sl["time"].start, sl["time"].stop)
        
        #  Load model data
        if self.load_data:
            # Use pre-loaded full dataset
            if hasattr(self, 'full_models'):
                models_ds = self.full_models.isel(time=sl["time"])
            else:
                # No pre-loaded model data, apply postpro and return
                if self.postpro_fn is not None:
                    sample = self.postpro_fn(sample)
                return sample
        else:
            # Load on-the-fly
            if len(self.models_paths) > 0 and len(time_indices) > 0:
                # Select model files for this time range
                selected_model_paths = self.models_paths[time_indices]
                
                models_ds = concatenate_parallel(
                    selected_model_paths,
                    var_list=self.models_vars,
                    slices=None,
                    type_coords="coords",
                    resize=1,
                    domain_limits=self.domain_limits,
                    n_jobs=5  # Fewer jobs for smaller batches
                )
            else:
                # No model data available, apply postpro and return
                if self.postpro_fn is not None:
                    sample = self.postpro_fn(sample)
                return sample
        
        #  Interpolate model data onto ASIP grid
        # sample is a dict with keys like 'xc', 'yc', 'lon', 'lat'
        if self.itrp_from_regular:
            # Get ASIP coords from sample dict - remove batch dimension
            asip_xc = np.squeeze(sample['xc'])  # Shape: (xc,)
            asip_yc = np.squeeze(sample['yc'])  # Shape: (yc,)
            target_grid = (asip_xc, asip_yc)
        else:
            lon = np.squeeze(sample['lon'])  # Shape: (yc, xc)
            lat = np.squeeze(sample['lat'])  # Shape: (yc, xc)
            target_grid = pyresample.geometry.SwathDefinition(
                lons=lon,
                lats=lat
            )
        
        # Interpolate each model variable
        model_vars = self.interpolate_dataset(
            target_grid, 
            models_ds, 
            self.models_vars, 
            prefix="models"
        ) 
        
        #  Add model variables to sample dict
        sample.update(model_vars)
        
        print(f"  Added {len(model_vars)} model variables: {list(model_vars.keys())}")
        
        #  Initialize target variables from their sources (like in data_multires_supervised.py)
        if self.var_mapping:
            # Handle resolution-specific var_mapping (dict) vs simple mapping
            if isinstance(self.var_mapping, dict) and any(k.startswith('patch_x') for k in self.var_mapping.keys()):
                # Multi-resolution case: use the finest resolution (self.resize)
                res_key = f"patch_x{self.resize}"
                mapping = self.var_mapping.get(res_key, {})
                print(f"  Using var_mapping for {res_key}")
            else:
                # Simple case: var_mapping is a flat dict
                mapping = self.var_mapping
            
            for target_var, source_var in mapping.items():
                # If target starts with models_XXX, create tgt_XXX from models_XXX
                if target_var.startswith('models_') and '_' in target_var:
                    suffix = target_var.split('_', 1)[1]  # Extract XXX from models_XXX
                    tgt_var = f"tgt_{suffix}"
                    if target_var in sample:
                        sample[tgt_var] = sample[target_var].copy()
                        print(f"  Created {tgt_var} <- {target_var}")
                    else:
                        print(f"  WARNING: Could not find {target_var} to create {tgt_var}")
                # If target starts with tgt_XXX, use the source_var directly
                elif target_var.startswith('tgt_') and source_var in sample:
                    sample[target_var] = sample[source_var].copy()
                    print(f"  Mapped {target_var} <- {source_var}")
                elif target_var.startswith('tgt_'):
                    print(f"  WARNING: Could not map {target_var} from {source_var}. Available keys: {list(sample.keys())}")
        
        #  NOW apply postpro_fn to convert dict to TrainingItem
        if self.postpro_fn is not None:
            # Check if postpro_fn accepts resolution_key parameter
            import inspect
            sig = inspect.signature(self.postpro_fn)
            if 'resolution_key' in sig.parameters:
                # Pass the resolution key (use self.resize for the current resolution)
                res_key = f"patch_x{self.resize}"
                sample = self.postpro_fn(sample, resolution_key=res_key)
            else:
                # Old-style postpro_fn without resolution_key
                sample = self.postpro_fn(sample)
        
        return sample
    
class XrConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concatenation of XrDatasets
    """
    def reconstruct(self, batches, weight=None):
        """
        Returns list of xarray object, reconstructed from batches
        """
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            rec_das.append(ds.reconstruct_from_items(ds_items, weight))
    
        return xr.concat(rec_das,dim="time")

class CustomBatchSampler(Sampler):
    r"""Yield a mini-batch of indices. 

    Args:
        data: Dataset for building sampling logic.
        batch_size: Size of mini-batch.
    """

    def __init__(self, data, batch_size):
        # build data for sampling here
        self.batch_size = batch_size
        self.data = data
        self.list_samples = np.random.randint(low=0, 
                                              high=len(data), 
                                              size=5000)

        
    def __iter__(self):
        # implement logic of sampling here
        batch = []
        #for i, item in enumerate(self.data):
        #    if int(np.mod(i,self.step))==0.:
        #        batch.append(i)
        #for i in np.arange(len(self.data),step=self.step):
        for i in self.list_samples:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.data)
    
class BaseDataModule(pl.LightningDataModule):
    def __init__(self, asip_paths, cimr_paths, cristal_paths,
                 covariates_paths, covariates,
                 target_vars,
                 satellite_vars=None,  # NEW
                 var_mapping=None,     # NEW
                 models_vars=None,  #  Accept models_vars
                 mask_path=None,
                 domain_name=None, domains=None,
                 xrds_kw=None, dl_kw=None, 
                 norm_stats=None, norm_stats_covs=None,
                 aug_kw=None, res=500, pads=[False,False,False], 
                 resize=1,
                 subsel_path="/dmidata/users/maxb/4dvarnet-starter/contrib/CROSCIM/patch_in_ocean",
                 rand_obs=False,
                 **kwargs):
        
        super().__init__()
        
        # Store variable configuration
        self.satellite_vars = satellite_vars or DEFAULT_VAR_GROUPS
        self.covariates = covariates or DEFAULT_COVARIATES
        self.models_vars = models_vars if models_vars is not None else []
        self.target_vars = target_vars
        self.var_mapping = dict(var_mapping) or {}

        # Recreate TrainingItem with current configuration
        # Now pickable thanks to custom __reduce__ method
        global TrainingItem
        TrainingItem = create_training_item(
            satellite_vars=self.satellite_vars,
            covariates=self.covariates,
            target_vars=self.target_vars,
            models_vars=self.models_vars
        )
        print(f"TrainingItem fields: {TrainingItem._fields}")

        # Determine active sources
        self.active_sources = [src for src, vars in self.satellite_vars.items() if vars]
        
        # Store paths only for active sources
        self.asip_paths = asip_paths if 'asip' in self.active_sources else []
        self.cimr_paths = cimr_paths if 'cimr' in self.active_sources else []
        self.cristal_paths = cristal_paths if 'cristal' in self.active_sources else []
        self.covariates_paths = covariates_paths
        
        self.mask_path = mask_path
        self.domain_name = domain_name
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.aug_kw = aug_kw if aug_kw is not None else {}
        self.res = res
        self.pads = pads
        self.resize = resize
        self._norm_stats = norm_stats
        self._norm_stats_covs = norm_stats_covs
        self.subsel_path = subsel_path
        self.rand_obs = rand_obs

        print(f"\n{'='*60}")
        print(f"BaseDataModule configuration:")
        print(f"{'='*60}")
        print(f"  Satellite vars: {self.satellite_vars}")
        print(f"  Active sources: {self.active_sources}")
        print(f"  Covariates: {self.covariates}")
        print(f"  Target vars: {self.target_vars}")
        print(f"  Var mapping: {self.var_mapping}")
        print(f"{'='*60}\n")
       
        self.resize = resize
        # Load base grid from ASIP to build lat/lon/xc/yc
        asip_base = xr.open_dataset(self.asip_paths[0])
        self.xc = asip_base.xc.data
        self.yc = asip_base.yc.data
        self.lon = asip_base.lon.data
        self.lat = asip_base.lat.data

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._post_fn = None

        if not os.path.isfile(self.mask_path):
            print("Building land mask...")
            self.mask = self.build_land_mask()
            self.mask.to_netcdf(self.mask_path)
            print("Done...")
        else:
            self.mask = xr.open_dataset(self.mask_path).mask

    def build_land_mask(self):
        mask = xr.Dataset(
                        coords={
                            "xc": self.xc,
                            "yc": self.yc,
                            "lon": (["yc","xc"], self.lon),
                            "lat": (["yc","xc"], self.lat)
                            })
        land_mask = np.zeros((len(self.yc),len(self.xc)))
        #land_10m = cfeature.NaturalEarthFeature('physical','land','10m')
        #land_polygons_cartopy = list(land_10m.geometries())
        zip_path = os.path.join(
                    cartopy.config['pre_existing_data_dir'],
                    'natural_earth',
                    'physical',
                    'ne_10m_land.zip'
                )
        reader = Reader(zip_path)
        land_polygons_cartopy = list(reader.geometries())
        land_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=land_polygons_cartopy)
        step_yc = np.concatenate((np.arange(len(self.yc),step=1000),np.array([len(self.yc)])))
        step_xc = np.concatenate((np.arange(len(self.xc),step=1000),np.array([len(self.xc)])))
        for i in range(len(step_yc)-1):
            for j in range(len(step_xc)-1):
                lon = self.lon[step_yc[i]:step_yc[i+1],step_xc[j]:step_xc[j+1]]
                lat = self.lat[step_yc[i]:step_yc[i+1],step_xc[j]:step_xc[j+1]]
                nlat, nlon = lon.shape
                points = GeoSeries(gpd.points_from_xy(lon.flatten(), lat.flatten()))
                points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
                joined = gpd.sjoin(points_gdf, land_gdf, how='left', predicate='within')
                part_land_mask = np.reshape(np.array(joined['index_right'].notnull().to_list()),(nlat,nlon))
                land_mask[step_yc[i]:step_yc[i+1],step_xc[j]:step_xc[j+1]] = part_land_mask
        mask = mask.update({"mask":(("yc","xc"),land_mask)})
        encoding = {
                   var: {"zlib": True, "complevel": 6}  # 9 = compression maximale
                   for var in mask.data_vars
                   }
        mask.to_netcdf(
                      self.mask_path,
                      format="NETCDF4",
                      engine="netcdf4",
                      encoding=encoding
        )
        return mask.mask 
    
    def norm_stats(self):
        return self._norm_stats

    def norm_stats_covs(self):
        return self._norm_stats_covs

    def post_fn(self, rand_obs=False):
        norm_sats = self._norm_stats
        norm_covs = self._norm_stats_covs

        # Determine which variables should have random masking
        if isinstance(self.rand_obs, bool):
            apply_rand_obs_to_all = self.rand_obs
            rand_obs_vars = []
        else:
            # rand_obs is a list of variable names
            apply_rand_obs_to_all = False
            rand_obs_vars = self.rand_obs if len(self.rand_obs) > 0 else []

        def normalize_var(x, stats):
            if stats['type'] == 'zscore':
                return (x - stats['mean']) / stats['std']
            elif stats['type'] == 'minmax':
                return (x - stats['min']) / (stats['max'] - stats['min'])
            elif stats['type'] is None:
                return x
            else:
                raise ValueError(f"Unknown normalization type {stats['type']}")

        def generate_random_obs_mask(gt_item):
            obs_mask_item = ~np.isnan(gt_item)
            _obs_item = gt_item.copy()
            dtime, dyc, dxc = gt_item.shape
            for t in range(dtime):
                if np.sum(obs_mask_item[t]) > .02 * dyc * dxc:
                    obs_obj = .5 * np.sum(obs_mask_item[t])
                    while np.sum(obs_mask_item[t]) >= obs_obj:
                        half_h = np.random.randint(2,10)
                        half_w = np.random.randint(2,10)
                        yc = np.random.randint(0, dyc)
                        xc = np.random.randint(0, dxc)
                        obs_mask_item[t, max(0,yc-half_h):min(dyc,yc+half_h+1),
                                         max(0,xc-half_w):min(dxc,xc+half_w+1)] = 0
            return np.where(obs_mask_item, _obs_item, np.nan)
            
        def apply_norm(item):
            """Normalize a batch item according to configuration."""
            # Recreate TrainingItem with current configuration to ensure correct fields
            TrainingItemClass = create_training_item(
                satellite_vars=self.satellite_vars,
                covariates=self.covariates,
                target_vars=self.target_vars if not isinstance(self.target_vars, dict) else list(set(
                    var for vars_list in self.target_vars.values() for var in vars_list
                )),
                models_vars=self.models_vars
            )
            # Handle missing fields: use None as default for fields not present in item
            item_with_defaults = {field: item.get(field, None) for field in TrainingItemClass._fields}
            data = TrainingItemClass(**item_with_defaults)
            
            # ==================================================================
            # NORMALIZATION LOGIC
            # ==================================================================
            
            # Build a mapping to determine which stats to use for each variable
            # This avoids double normalization
            stats_to_use = {}
            
            # Check var_mapping to determine if satellite variables should use model stats
            for target_var, source_var in self.var_mapping.items():
                if target_var.startswith('models_') and '_' in target_var:
                    # models_XXX -> satellite source should use model stats
                    var_suffix = target_var.split('_', 1)[1]
                    stats_to_use[source_var] = norm_models[var_suffix]
            
            # 1. Normalize satellite variables
            for source in self.active_sources:
                if source == 'models':
                    continue
                for var in self.satellite_vars[source]:
                    var_key = f"{source}_{var}"
                    if hasattr(data, var_key):
                        var_data = getattr(data, var_key)
                        if var_data is None:
                            continue
                        # Apply random obs mask if requested
                        should_apply_mask = apply_rand_obs_to_all or (var_key in rand_obs_vars)
                        if should_apply_mask:
                            var_data = generate_random_obs_mask(var_data)
                        # Use mapped stats if available, else default satellite stats
                        norm_stats = stats_to_use.get(var_key, norm_sats[source][var])
                        var_data = normalize_var(var_data, norm_stats)
                        data = data._replace(**{var_key: var_data})
            
            # 2. Normalize models variables
            for var in self.models_vars:
                var_key = f"models_{var}"
                if hasattr(data, var_key):
                    var_data = getattr(data, var_key)
                    if var_data is None:
                        continue
                    # Normalize with models stats
                    var_data = normalize_var(var_data, norm_models[var])
                    data = data._replace(**{var_key: var_data})
            
            # 3. Normalize target variables based on their source
            for target_var, source_var in self.var_mapping.items():
                if target_var.startswith('tgt_') and hasattr(data, target_var):
                    # tgt_XXX -> normalize with source stats
                    var_data = getattr(data, target_var)
                    if var_data is None:
                        continue
                    if '_' in source_var:
                        group, var = source_var.split('_', 1)
                        if group in norm_sats:
                            norm_stats = norm_sats[group][var]
                        elif group == 'models':
                            norm_stats = norm_models[var]
                        else:
                            continue
                        var_data = normalize_var(var_data, norm_stats)
                        data = data._replace(**{target_var: var_data})
            
            # 4. Normalize tgt_XXX variables created from models_XXX (not in mapping)
            # These are typically created when models_XXX is in target_vars
            target_vars_list = self.target_vars if not isinstance(self.target_vars, dict) else list(set(
                var for vars_list in self.target_vars.values() for var in vars_list
            ))
            for target_var in target_vars_list:
                if target_var.startswith('models_') and '_' in target_var:
                    # models_XXX -> also normalize tgt_XXX if it exists
                    var_suffix = target_var.split('_', 1)[1]
                    tgt_var = f"tgt_{var_suffix}"
                    if hasattr(data, tgt_var):
                        var_data = getattr(data, tgt_var)
                        if var_data is not None:
                            # Use same normalization as models_XXX
                            var_data = normalize_var(var_data, norm_models[var_suffix])
                            data = data._replace(**{tgt_var: var_data})
            
            # Normalize covariates
            for cov in self.covariates:
                if hasattr(data, cov):
                    cov_data = getattr(data, cov)
                    # Skip if None (field was missing)
                    if cov_data is None:
                        continue
                    norm_params = norm_covs[cov]
                    cov_data = normalize_var(cov_data, norm_params)
                    data = data._replace(**{cov: cov_data})
            
            # Normalize coordinates
            data = data._replace(land_mask=data.land_mask)
            data = data._replace(lat=normalize_var(data.lat, {"type": "minmax", "min": 50, "max": 90}))
            data = data._replace(lon=normalize_var(data.lon, {"type": "minmax", "min": -180, "max": 180}))
            
            # Replace None fields with empty tensors to avoid collate issues
            # This keeps the same TrainingItem class (picklable) but allows collate to work
            data_dict = data._asdict()
            for field in data._fields:
                if data_dict[field] is None:
                    # Create empty tensor with shape (0,) - will be filtered out during model processing
                    data_dict[field] = np.array([], dtype=np.float32)
            
            return type(data)(**data_dict)

        return ft.partial(ft.reduce, lambda i, f: f(i), [apply_norm])

    def save_batch_as_NetCDF(self, batch, ibatch, patch_dims, save_dir="/dmidata/users/maxb/PREPROC/"):
        """Save a batch in NetCDF format using dynamic variable configuration."""
        
        data_vars = {}

        # Helper function to check if a field is non-empty
        def is_valid_field(tensor):
            """Check if tensor is valid (not empty array from None replacement)."""
            if torch.is_tensor(tensor):
                return tensor.numel() > 0 and tensor.ndim == 4
            elif isinstance(tensor, np.ndarray):
                return tensor.size > 0
            return False

        # Satellite variables (dynamically from config)
        for source in self.active_sources:
            for var in self.satellite_vars[source]:
                var_key = f"{source}_{var}"
                if hasattr(batch, var_key):
                    tensor = getattr(batch, var_key)
                    if is_valid_field(tensor):
                        data_vars[var_key] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())

        # Covariates
        for cov in self.covariates:
            if hasattr(batch, cov):
                tensor = getattr(batch, cov)
                if is_valid_field(tensor):
                    data_vars[cov] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())

        # Target variables
        for target_var in self.target_vars:
            if hasattr(batch, target_var):
                tensor = getattr(batch, target_var)
                if is_valid_field(tensor):
                    data_vars[target_var] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())

        # Coordinates and mask (always present, not filtered)
        data_vars.update({
            'times': (('sample', 'time'), torch.squeeze(batch.time, dim=1).detach().cpu().numpy().astype("datetime64[s]")),
            'ycs': (('sample', 'yc'), torch.squeeze(batch.yc, dim=1).detach().cpu()),
            'xcs': (('sample', 'xc'), torch.squeeze(batch.xc, dim=1).detach().cpu()),
            'lat': (('sample', 'yc', 'xc'), torch.squeeze(batch.lat, dim=1).detach().cpu()),
            'lon': (('sample', 'yc', 'xc'), torch.squeeze(batch.lon, dim=1).detach().cpu()),
            'land_mask': (('sample', 'yc', 'xc'), torch.squeeze(batch.land_mask, dim=1).detach().cpu()),
        })

        coords = {
            'sample': np.arange(list(data_vars.values())[0][1].shape[0]),
            'time': np.arange(patch_dims['time']),
            'yc': np.arange(patch_dims['yc']),
            'xc': np.arange(patch_dims['xc'])
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"preproc_batch_{ibatch}.nc")
        ds.to_netcdf(save_path)
        print(f"Saved: {save_path}")

    def setup(self, stage='test'):

        def select_paths(files, times, fmt="%Y%m%d"):
            """
            Select files based on time ranges.
            
            Args:
                files: List of file paths
                times: Either a single slice or a list of slices
                fmt: Date format string
            """
            from omegaconf import ListConfig
            
            dates, time_vals = [], []
            
            # Convert OmegaConf ListConfig to list if needed
            if isinstance(times, ListConfig):
                times = list(times)
            
            # Check if times is a list of slices or a single slice
            if isinstance(times, list):
                # Multiple time ranges (e.g., for validation)
                for time_slice in times:
                    start = time_slice.start
                    stop = time_slice.stop
                    dts = pd.date_range(start, stop)
                    dates.extend(dts.strftime(fmt).tolist())
                    time_vals.extend(dts.tolist())
            else:
                # Single time range (e.g., for train/test)
                start = times.start
                stop = times.stop
                dts = pd.date_range(start, stop)
                dates = dts.strftime(fmt).tolist()
                time_vals = dts.tolist()
            
            # Select files matching the dates
            selected_files = np.sort([f for f in files if any(date in f for date in dates)])
            return selected_files, np.array(time_vals)

        def create_dataset(split):
            # Get paths only for active sources
            paths_dict = {}
            times = None
            
            if 'asip' in self.active_sources:
                asip_paths, times = select_paths(self.asip_paths, self.domains[split]['time'])
                paths_dict['asip_paths'] = asip_paths
            
            if 'cimr' in self.active_sources:
                cimr_paths, _ = select_paths(self.cimr_paths, self.domains[split]['time'], fmt="%Y-%m-%d")
                paths_dict['cimr_paths'] = cimr_paths
            
            if 'cristal' in self.active_sources:
                cristal_paths, _ = select_paths(self.cristal_paths, self.domains[split]['time'], fmt="%Y-%m-%d")
                paths_dict['cristal_paths'] = cristal_paths
            
            if self.covariates:
                cov_paths, _ = select_paths(self.covariates_paths, self.domains[split]['time'], fmt="%Y-%m-%d")
                paths_dict['covariates_paths'] = cov_paths
            
            return XrDataset(
                **paths_dict,
                covariates=self.covariates,
                target_vars=self.target_vars,
                satellite_vars=self.satellite_vars,  # NEW
                var_mapping=self.var_mapping,        # NEW
                mask=self.mask,
                times=times,
                **self.xrds_kw,
                postpro_fn=self.post_fn(rand_obs=(split=='train')),
                res=self.res,
                pad=self.pads[0 if split == 'train' else 1 if split == 'val' else 2],
                resize=self.resize,
                stride_test=(split != 'train'),
                load_data=(split == 'test'),
                subsel_patch=True,
                subsel_patch_path=f"{self.subsel_path}/patch_in_ocean_{split}_{self.domain_name}_patch_{self.xrds_kw['patch_dims']['yc']}_{self.xrds_kw['strides']['yc']}_resize_x{self.resize}.txt"
            )

        # self.train_ds = create_dataset('train')
        # self.val_ds = create_dataset('val')
        self.test_ds = create_dataset('test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        sampler = CustomBatchSampler(self.val_ds, batch_size=self.dl_kw["batch_size"])
        return torch.utils.data.DataLoader(self.val_ds, batch_sampler=sampler, num_workers=self.dl_kw["num_workers"])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

class ConcatDataModule(BaseDataModule):

    def setup(self, stage='test'):
        # Postprocessing functions
        post_fn_train = self.post_fn(rand_obs=self.rand_obs)
        post_fn_eval = self.post_fn(rand_obs=False)

        # Training set
        self.train_ds = XrConcatDataset([
            XrDataset(
                asip_paths=self.asip_paths, 
                cimr_paths=self.cimr_paths, 
                cristal_paths=self.cristal_paths, 
                covariates_paths=self.covariates_paths, 
                covariates=COVARIATES,
                mask=self.mask,
                times=None,  # Optional, depending on XrDataset signature
                **self.xrds_kw,
                postpro_fn=post_fn_train,
                domain=domain,
                res=self.res,
                pad=self.pads[0],
                resize = self.resize,
                stride_test=False,
                load_data=False
            )
            for domain in self.domains['train']
        ])

        if self.aug_factor >= 1:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        # Validation set
        self.val_ds = XrConcatDataset([
            XrDataset(
                asip_paths=self.asip_paths, 
                cimr_paths=self.cimr_paths, 
                cristal_paths=self.cristal_paths, 
                covariates_paths=self.covariates_paths, 
                covariates=COVARIATES,
                mask=self.mask,
                times=None,
                **self.xrds_kw,
                postpro_fn=post_fn_eval,
                domain=domain,
                res=self.res,
                resize = self.resize,
                pad=self.pads[1],
                stride_test=True,
                load_data=False
            )
            for domain in self.domains['val']
        ])

        # Test set
        self.test_ds = XrConcatDataset([
            XrDataset(
                asip_paths=self.asip_paths, 
                cimr_paths=self.cimr_paths, 
                cristal_paths=self.cristal_paths, 
                covariates_paths=self.covariates_paths, 
                covariates=COVARIATES,
                mask=self.mask,
                times=None,
                **self.xrds_kw,
                postpro_fn=post_fn_eval,
                domain=domain,
                res=self.res,
                resize = self.resize,
                pad=self.pads[2],
                stride_test=True,
                load_data=True
            )
            for domain in self.domains['test']
        ])
