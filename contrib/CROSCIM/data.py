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
from contrib.CROSCIM.load_data import *
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

def create_training_item(var_groups, covariates, tgt_vars):
    """
    Dynamically create a TrainingItem with variables from var_groups and covariates
    """
    fields = []
    # Variables satellites
    for group in var_groups:
        for var in var_groups[group]:
            fields.append(f"{group}_{var}")
    for group, variables in VAR_GROUPS.items():
        for var in variables:
            var_key = f"{group}_{var}"
            new_key = f"tgt_{var}"
            if (var_key in tgt_vars):
                fields.append(new_key)
    # Covariates
    fields.extend(covariates)
    # Coordonnées et masque
    fields.extend(['lat', 'lon', 'land_mask', "time", "yc", "xc"])
    # Final namedtuple
    return namedtuple("TrainingItem", fields)

TrainingItem = create_training_item(VAR_GROUPS, COVARIATES,
                                    tgt_vars=["asip_sic","cimr_SIT"])


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
                 tgt_vars,
                 mask, times,
                 patch_dims, domain_limits=None, strides=None,
                 strides_test=None, postpro_fn=None,
                 resize=1, res=500, pad=False, stride_test=False,
                 subsel_patch=False, subsel_patch_path=None,
                 itrp_from_regular=True,
                 load_data=False, domain=None):

        super().__init__()
        self.postpro_fn = postpro_fn
        self.asip_paths = asip_paths
        self.cimr_paths = cimr_paths
        self.cristal_paths = cristal_paths
        self.covariates_paths = covariates_paths
        self.covariates = covariates
        self.tgt_vars = tgt_vars
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
        asip_base = xr.open_dataset(self.asip_paths[0]).sel(**(domain_limits or {}))
        if self.resize!=1:
            print(f"coarsening target data by factor {resize}")
            asip_base = fast_coarsen_xr(asip_base, factor_x=resize, factor_y=resize)
            self.mask = fast_coarsen_xr_array(self.mask, factor_x=resize, factor_y=resize, 
                                        mode="binary")
        self.xc = asip_base.xc.data
        self.yc = asip_base.yc.data
        self.lon = asip_base.lon.data
        self.lat = asip_base.lat.data

        # load data in memory (for inference)
        if self.load_data:
            self.full_asip, self.full_cimr, self.full_cristal, self.full_covs = load_mfdata(
                self.asip_paths,
                self.cimr_paths,
                self.cristal_paths,
                self.covariates_paths,
                self.covariates,
                slice(
                    datetime.datetime.strftime(self.times[0], "%Y-%m-%d"),
                    datetime.datetime.strftime(self.times[-1] + datetime.timedelta(days=1), "%Y-%m-%d")
                ),
                slices=self.domain_limits,
                type_coords="coords",
                resize=self.resize
            )

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

        #item_mask = self.mask.isel(xc=xc_slice, yc=yc_slice).values     
        item_mask = self.mask.sel(xc=slice(self.xc[sl["xc"].start],self.xc[sl["xc"].stop-1]),
                                  yc=slice(self.yc[sl["yc"].start],self.yc[sl["yc"].stop-1])).values

        # loading of the datasets
        if self.load_data:
            asip_ds = self.full_asip.isel(time=sl["time"]).sel(xc=slice(self.xc[sl["xc"].start],
                                                                        self.xc[sl["xc"].stop-1]),
                                                               yc=slice(self.yc[sl["yc"].start],
                                                                        self.yc[sl["yc"].stop-1]))
            cimr_ds = self.full_cimr.isel(time=sl["time"])
            cristal_ds = self.full_cristal.isel(time=sl["time"])
            covariate_ds = self.full_covs.isel(time=sl["time"])
        else:
            time_indices = np.arange(sl["time"].start, sl["time"].stop)
            if self.resize==1:
                slices = {"xc": sl["xc"], "yc": sl["yc"]}
                type_coords = "index"
            else:
                slices = {"xc": slice(self.xc[sl["xc"].start],self.xc[sl["xc"].stop]), 
                          "yc": slice(self.yc[sl["yc"].start],self.yc[sl["yc"].stop])
                         }
                type_coords = "coords"
            asip_ds = concatenate(self.asip_paths[time_indices], var_list=VAR_GROUPS["asip"], 
                                  slices=slices, type_coords=type_coords, resize=self.resize,
                                  domain_limits=self.domain_limits)
            cimr_ds = concatenate(self.cimr_paths[time_indices], var_list=VAR_GROUPS["cimr"], slices=None)
            cristal_ds = concatenate(self.cristal_paths[time_indices], var_list=VAR_GROUPS["cristal"], slices=None)
            covariate_ds = concatenate(self.covariates_paths[time_indices], var_list=self.covariates, slices=None)

        # Pad if necessary
        expected_shape = (self.patch_dims['time'], self.patch_dims['yc'], self.patch_dims['xc'])
        actual_shape = asip_ds["sic"].shape

        # padding if necessary
        asip_ds = asip_ds.update({"mask":(("yc","xc"),item_mask)})
        if actual_shape != expected_shape:
            ix = [find_idx(self.xc,x) for x in self.xc[sl["xc"].start:sl["xc"].stop]]
            iy = [find_idx(self.yc,y) for y in self.yc[sl["yc"].start:sl["yc"].stop]]
            padded_patch = xr.Dataset(
                        coords={
                        "time": asip_ds.time,
                        "xc": self.xc[sl["xc"].start:sl["xc"].stop],
                        "yc": self.yc[sl["yc"].start:sl["yc"].stop],
                        "lon": (["yc","xc"], self.lon[iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)]),
                        "lat": (["yc","xc"], self.lat[iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)])
                        })
            asip_ds = xr.align(padded_patch,asip_ds, join="left")[1]
            """
            asip_ds = pad_batch_with_coords(asip_ds,
                                              sl,
                                              self.xc,
                                              self.yc,
                                              self.lon,
                                              self.lat)
            """
            asip_ds['mask'] = asip_ds['mask'].fillna(1)
            item_mask = asip_ds.mask.data
            
        asip_vars = {f"asip_{var}": asip_ds[var].values for var in VAR_GROUPS["asip"] if var in asip_ds}   

        lon_patch = asip_ds.lon.values
        lat_patch = asip_ds.lat.values

        # Prepare swath for interpolation
        if self.itrp_from_regular:
            target_grid=(asip_ds.xc.values, asip_ds.yc.values)
            cimr_vars = self.interpolate_dataset(target_grid, cimr_ds, VAR_GROUPS["cimr"],prefix="cimr")
            cristal_vars = self.interpolate_dataset(target_grid, cristal_ds, VAR_GROUPS["cristal"],prefix="cristal")
            covariate_vars = self.interpolate_dataset(target_grid, covariate_ds, self.covariates)
        else:
            swath_def_target = pyresample.geometry.SwathDefinition(lons=lon_patch, lats=lat_patch)
            cimr_vars = self.interpolate_dataset(swath_def_target, cimr_ds, VAR_GROUPS["cimr"],prefix="cimr")
            cristal_vars = self.interpolate_dataset(swath_def_target, cristal_ds, VAR_GROUPS["cristal"],prefix="cristal")
            covariate_vars = self.interpolate_dataset(swath_def_target, covariate_ds, self.covariates)

        # Assemble sample
        sample = {**asip_vars, **cimr_vars, **cristal_vars, **covariate_vars}
        sample["land_mask"] = np.expand_dims(item_mask, axis=0)
        sample["lat"] = np.expand_dims(lat_patch, axis=0)
        sample["lon"] = np.expand_dims(lon_patch, axis=0)
         
        # add target variables
        for group, variables in VAR_GROUPS.items():
            for var in variables:
                var_key = f"{group}_{var}"
                new_key = f"tgt_{var}"
                if (var_key in sample) and (var_key in self.tgt_vars):
                    sample[new_key] = sample[var_key]

        # keep track of the coordinates
        sample["time"] = np.expand_dims(np.array([ np.datetime64(t,"s").astype('float64') for t in asip_ds.time.values]),
                                        axis=0)
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

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, asip_paths, cimr_paths, cristal_paths,
                 covariates_paths, covariates,
                 tgt_vars,
                 mask_path,
                 domain_name, domains,
                 xrds_kw, dl_kw, 
                 norm_stats, norm_stats_covs,
                 aug_kw=None, res=500, pads=[False,False,False], 
                 resize=1,
                 subsel_path="/dmidata/users/maxb/4dvarnet-starter/contrib/CROSCIM",
                 **kwargs):
        
        super().__init__()
        self.asip_paths = asip_paths
        self.cimr_paths = cimr_paths
        self.cristal_paths = cristal_paths
        self.covariates_paths = covariates_paths
        self.covariates = covariates
        self.tgt_vars = tgt_vars
        self.mask_path = mask_path
        self.domain_name = domain_name
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.aug_kw = aug_kw if aug_kw is not None else {}
        self.res = res
        self.pads = pads
        self.resize = resize
        self._norm_stats = norm_stats  # Satellite variables normalization (VAR_GROUPS)
        self._norm_stats_covs = norm_stats_covs  # Covariate normalization (COVARIATES)
        self.subsel_path = subsel_path
       
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
        land_50m = cfeature.NaturalEarthFeature('physical','land','10m')
        land_polygons_cartopy = list(land_50m.geometries())
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

        def normalize_var(x, stats):
            if stats['type'] == 'zscore':
                return (x - stats['mean']) / stats['std']
            elif stats['type'] == 'minmax':
                return (x - stats['min']) / (stats['max'] - stats['min'])
            elif stats['type'] is None:
                return x
            else:
                raise ValueError(f"Unknown normalization type {stats['type']} for variable")

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
            """
            Normalize a batch item according to norm_stats and norm_stats_covs.
            """
            data = TrainingItem(**item)
            obs_mask_item = None

            # Normalize target variables
            for group, variables in VAR_GROUPS.items():
                for var in variables:
                    var_key = f"{group}_{var}"
                    new_key = f"tgt_{var}"
                    if hasattr(data, var_key) and (var_key in self.tgt_vars):
                        var_data = getattr(data, new_key)
                        norm_params = norm_sats[group][var]
                        var_data = normalize_var(var_data, norm_params)
                        data = data._replace(**{new_key: var_data})

            # Normalisation des satellites
            for group, variables in VAR_GROUPS.items():
                for var in variables:
                    var_key = f"{group}_{var}"
                    if hasattr(data, var_key):
                            var_data = getattr(data, var_key)
                            # Mask obs aléatoires
                            if rand_obs:
                                var_data = generate_random_obs_mask(var_data)
                            norm_params = norm_sats[group][var]
                            var_data = normalize_var(var_data, norm_params)
                            data = data._replace(**{var_key: var_data})

            # Normalisation des covariates
            for cov in COVARIATES:
                if hasattr(data, cov):
                        norm_params = norm_covs[cov]
                        cov_data = normalize_var(getattr(data, cov), norm_params)
                        data = data._replace(**{cov: cov_data})

            # land_mask inchangé
            data = data._replace(land_mask=data.land_mask)
            # Normalisation latitude / longitude
            data = data._replace(lat=normalize_var(data.lat, {"type": "minmax", "min": 50, "max": 90}))
            data = data._replace(lon=normalize_var(data.lon, {"type": "minmax", "min": -180, "max": 180}))

            return data

        return ft.partial(ft.reduce, lambda i, f: f(i), [apply_norm])

    def save_batch_as_NetCDF(self, batch, ibatch, patch_dims, save_dir="/dmidata/users/maxb/PREPROC/"):
        """
        Save a batch in NetCDF format, adapted to VAR_GROUPS logic
        """
    
        # Variables à sauvegarder
        data_vars = {}
    
        # Variables satellites (asip, cimr, cristal)
        for group in VAR_GROUPS:
            for var in VAR_GROUPS[group]:
                if hasattr(batch, "{group}_{var}"):
                    data_vars[var] = (('sample', 'time', 'yc', 'xc'), getattr(batch, var).detach().cpu())
    
        # Covariates
        for cov in COVARIATES:
            if hasattr(batch, cov):
                tensor = getattr(batch, cov)
                if torch.is_tensor(tensor) and tensor.ndim == 4:
                    data_vars[cov] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())
 
        # target variables
        for target in self.tgt_vars:
            if hasattr(batch, target):
                data_vars[target] = (('sample', 'time', 'yc', 'xc'), getattr(batch, target).detach().cpu())
        
        # Coordonnées et masque
        data_vars.update({
            'times': (('sample', 'time'), torch.squeeze(batch.time).detach().cpu().numpy().astype("datetime64[s]")),
            'ycs': (('sample', 'yc'), torch.squeeze(batch.yc.detach().cpu())),
            'xcs': (('sample', 'xc'), torch.squeeze(batch.xc.detach().cpu())),
            'lat': (('sample', 'yc', 'xc'), torch.squeeze(batch.lat).detach().cpu()),
            'lon': (('sample', 'yc', 'xc'), torch.squeeze(batch.lon).detach().cpu()),
            'land_mask': (('sample', 'yc', 'xc'), torch.squeeze(batch.land_mask).detach().cpu()),
        })
    
        coords = {
            'sample': np.arange(list(data_vars.values())[0][1].shape[0]),
            'time': np.arange(patch_dims['time']),
            'yc': np.arange(patch_dims['yc']),
            'xc': np.arange(patch_dims['xc'])
        }
    
        # Sauvegarde
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
    
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"preproc_batch_{ibatch}.nc")
        ds.to_netcdf(save_path)

    def setup(self, stage='test'):

        def select_paths(files, times, fmt="%Y%m%d"):
            if isinstance(times, list):
                dates, time_vals = [], []
                for sl in times:
                    start, end = sl.start, sl.stop
                    dts = pd.date_range(start, end)
                    dates.extend(dts.strftime(fmt).tolist())
                    time_vals.extend(dts.tolist())
            else:
                start, end = times.start, times.stop
                dts = pd.date_range(start, end)
                dates = dts.strftime(fmt).tolist()
                time_vals = dts.tolist()
            files = np.sort([f for f in files if any(date in f for date in dates)])
            return files, np.array(time_vals)

        def create_dataset(split):
            asip_paths, times = select_paths(self.asip_paths, self.domains[split]['time'])
            cimr_paths, _ = select_paths(self.cimr_paths, self.domains[split]['time'], fmt="%Y-%m-%d")
            cristal_paths, _ = select_paths(self.cristal_paths, self.domains[split]['time'], fmt="%Y-%m-%d")
            cov_paths, _ = select_paths(self.covariates_paths, self.domains[split]['time'], fmt="%Y-%m-%d")
            return XrDataset(
                asip_paths=asip_paths,
                cimr_paths=cimr_paths,
                cristal_paths=cristal_paths,
                covariates_paths=cov_paths,
                covariates=COVARIATES,
                tgt_vars=self.tgt_vars,
                mask=self.mask,
                times=times,
                **self.xrds_kw,
                postpro_fn=self.post_fn(rand_obs=(split=='train')),
                res=self.res,
                pad=self.pads[0 if split == 'train' else 1 if split == 'val' else 2],
                resize = self.resize,
                stride_test=(split != 'train'),
                load_data=(split == 'test'),
                subsel_patch=True,
                subsel_patch_path=f"{self.subsel_path}/patch_in_ocean_{split}_{self.domain_name}_patch_{self.xrds_kw['patch_dims']['yc']}_{self.xrds_kw['strides']['yc']}_resize_x{self.resize}.txt"
            )

        #self.train_ds = create_dataset('train')
        #self.val_ds = create_dataset('val')
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
        post_fn_train = self.post_fn(rand_obs=True)
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
