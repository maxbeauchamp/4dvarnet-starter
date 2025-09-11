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
from contrib.ASIP_OSISAF.load_data import *
import datetime
import pyresample
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import os
from torch.utils.data.sampler import Sampler

TrainingItem_wogeo = namedtuple(
    'TrainingItem_wogeo', ['input', 'tgt', 'coarse', 'land_mask']
)


TrainingItem = namedtuple(
    'TrainingItem', ['input', 'tgt', 'coarse', 
                     'latv', 'lonv', 'land_mask',
                     't2m','istl1','siconc','sst','skt']
)

class IncompleteScanConfiguration(Exception):
    pass

class DangerousDimOrdering(Exception):
    pass

def find_pad(sl, st, N):
    k = np.floor(N/st)
    if N>((k*st) + (sl-st)):
        pad = (k+1)*st + (sl-st) - N
    elif N<((k*st) + (sl-st)):
        pad = (k*st) + (sl-st) - N
    else:
        pad = 0
    return int(pad/2), int(pad-int(pad/2))

def find_idx(coords,c):
    return np.where(coords==c)[0][0]

class XrDataset(torch.utils.data.Dataset):
    """
    torch Dataset based on an xarray.DataArray with on the fly slicing.

    ### Usage: #### 
    If you want to be able to reconstruct the input

    the input xr.DataArray should:
        - have coordinates
        - have the last dims correspond to the patch dims in same order
        - have for each dim of patch_dim (size(dim) - patch_dim(dim)) divisible by stride(dim)

    the batches passed to self.reconstruct should:
        - have the last dims correspond to the patch dims in same order
    """
    def __init__(
            self, 
            asip_paths,
            osisaf_paths,
            covariates_paths,
            covariates,
            mask,
            times,
            patch_dims, domain_limits=None, strides=None,
            strides_test=None,
            check_full_scan=False, check_dim_order=False,
            postpro_fn=None,
            res=500,
            pad = False,
            stride_test=False,
            use_mf=False,
            subsel_patch=False,
            subsel_patch_path=None,
            load_data=False
            ):
        """
        da: xarray.DataArray with patch dims at the end in the dim orders
        patch_dims: dict of da dimension to size of a patch 
        domain_limits: dict of da dimension to slices of domain to select for patch extractions
        strides: dict of dims to stride size (default to one)
        check_full_scan: Boolean: if True raise an error if the whole domain is not scanned by the patch size stride combination
        """

        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.asip_paths = asip_paths
        self.osisaf_paths = osisaf_paths
        self.covariates_paths = covariates_paths
        self.covariates = covariates
        self.times = times
        self.patch_dims = patch_dims
        self.strides = strides or {}
        if stride_test:
            self.strides = strides_test or {}
        self.res = res
        self.pad = pad
        self.use_mf = False
        self.subsel_patch = subsel_patch
        self.load_data = load_data

        # store coords 
        times = np.arange(len(self.asip_paths))
        asip_base = xr.open_dataset(self.asip_paths[0]).sel(**(domain_limits or {}))
        xc_orig = asip_base.xc.data
        yc_orig = asip_base.yc.data
        self.lon = asip_base.lon.data
        self.lat = asip_base.lat.data
        self.xc = xc_orig
        self.yc = yc_orig

        self.mask = mask.sel(**(domain_limits or {}))

        if self.load_data:
            self.asip, self.osisaf, self.covs = load_mfdata(self.asip_paths,
                                                            self.osisaf_paths,
                                                            self.covariates_paths,
                                                            self.covariates,
                                                            slice(datetime.datetime.strftime(self.times[0], "%Y-%m-%d"),
                                                            datetime.datetime.strftime(self.times[-1]+datetime.timedelta(days=1), "%Y-%m-%d")),
                                                            domain_limits, type_coords="coords")

        # pad
        nt, ny, nx = (len(times),len(xc_orig),len(yc_orig))
        if self.pad:
            pad_x = find_pad(self.patch_dims['xc'], self.strides['xc'], nx)
            pad_y = find_pad(self.patch_dims['yc'], self.strides['yc'], ny)
            self.lon = np.pad(self.lon,((pad_y[0],pad_y[1]),
                                        (pad_x[0],pad_x[1])),
                                        mode="constant",
                                        constant_values=((self.lon[0,0],self.lon[-1,0]),
                                                    (self.lon[0,0],self.lon[0,-1])
                                                   )
                                        )
            self.lat = np.pad(self.lat,((pad_y[0],pad_y[1]),
                                        (pad_x[0],pad_x[1])),
                                        mode="constant",
                                        constant_values=((self.lat[0,0],self.lat[-1,0]),
                                                    (self.lat[0,0],self.lat[0,-1])
                                                   )

                                        )
            
            pad_ = {'xc':(pad_x[0],pad_x[1]),
                    'yc':(pad_y[0],pad_y[1])}

            self.mask = self.mask.pad(pad_, constant_values=1)

            dx = [pad_ *self.res for pad_ in pad_x]
            dy = [pad_ *self.res for pad_ in pad_y]
            new_xc = np.concatenate((np.linspace(xc_orig[0]-dx[0],
                                                 xc_orig[0],
                                                 pad_x[0],endpoint=False),
                                     xc_orig,
                                     np.linspace(xc_orig[-1]+self.res,
                                                 xc_orig[-1]+dx[1]+ self.res,
                                                 pad_x[1],endpoint=False))) 
            new_yc = np.concatenate((np.linspace(yc_orig[0]+dy[0],
                                                 yc_orig[0],
                                                 pad_y[0],endpoint=False),
                                     yc_orig,
                                     np.linspace(yc_orig[-1]-self.res,
                                                 yc_orig[-1]-dy[1]-self.res,
                                                 pad_y[1],endpoint=False))) 
            yc_padded = np.round(new_yc,2)
            xc_padded = np.round(new_xc,2)
            da_dims = dict(time=nt, xc=len(xc_padded), yc=len(yc_padded))
            self.xc = xc_padded
            self.yc = yc_padded
        else:
            da_dims = dict(time=nt, xc=len(xc_orig), yc=len(yc_orig))
        self.da_dims =da_dims

        self.ds_size = {
            dim: max((da_dims[dim] - patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in patch_dims
        }

        # get patches in ocean
        if self.subsel_patch:
            if not os.path.isfile(subsel_patch_path):
                idx0 = self.find_patches_in_ocean()
                print("Saving ocean patches in "+subsel_patch_path)
                np.savetxt(subsel_patch_path, idx0, fmt='%i')
            else:
                idx0 = np.loadtxt(subsel_patch_path)
            # repeat for the number of times
            nitem_bytime = np.prod(list(self.ds_size.values())[1:])
            self.idx_patches_in_ocean = np.concatenate([ idx0+(nitem_bytime*i) for i in range(list(self.ds_size.values())[0]) ])

        if check_full_scan:
            for dim in patch_dims:
                if (da_dims[dim] - self.patch_dims[dim]) % self.strides.get(dim, 1) != 0:
                    raise IncompleteScanConfiguration(
                        f"""
                        Incomplete scan in dimension dim {dim}:
                        dataarray shape on this dim {da_dims[dim]}
                        patch_size along this dim {self.patch_dims[dim]}
                        stride along this dim {self.strides.get(dim, 1)}
                        [shape - patch_size] should be divisible by stride
                        """
                    )

        if check_dim_order:
            for dim in patch_dims:
                if not '#'.join(da.dims).endswith('#'.join(list(patch_dims))): 
                    raise DangerousDimOrdering(
                        f"""
                        input dataarray's dims should end with patch_dims 
                        dataarray's dim {da.dims}:
                        patch_dims {list(patch_dims)}
                        """
                )

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

    def get_coords_old(self, limit=None):
        self.return_coords = True
        coords = []
        try:
            if limit is None:
                for i in range(len(self)):
                    coords.append(self[i])
            else:
                for i in sample(range(0,len(self)),limit):
                    coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def get_coords(self, limit=None):
        self.return_coords = True
        coords = []
        try:
            if limit is None:
                limit = len(self)
                seq = np.arange(0,limit)
            else:
                seq = sample(range(0,len(self)),limit)
            for i in seq:
                if self.subsel_patch:
                    sl = {
                        dim: slice(self.strides.get(dim, 1) * idx,
                                   self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                        for dim, idx in zip(self.ds_size.keys(),
                                            np.unravel_index(int(self.idx_patches_in_ocean[i]), tuple(self.ds_size.values())))
                    }
                else:
                    sl = {
                        dim: slice(self.strides.get(dim, 1) * idx,
                                   self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                        for dim, idx in zip(self.ds_size.keys(),
                                            np.unravel_index(i, tuple(self.ds_size.values())))
                     }
                coords.append(xr.Dataset(coords=dict(xc=self.xc[sl["xc"].start:sl["xc"].stop],
                                                         yc=self.yc[sl["yc"].start:sl["yc"].stop],
                                                         time=self.times[sl["time"].start:sl["time"].stop],
                                                        )
                                             ).transpose('time', 'yc', 'xc')
                                  )
        finally:
            self.return_coords = False
            return coords

    def item_in_ocean(self, lon, lat):
        points = GeoSeries(gpd.points_from_xy(lon, lat))
        points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
        land_50m = cfeature.NaturalEarthFeature('physical','land','50m')
        land_polygons_cartopy = list(land_50m.geometries())
        land_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=land_polygons_cartopy)
        # Spatially join the points with the land polygons
        joined = gpd.sjoin(points_gdf, land_gdf, how='left', predicate='within')
        # Check if each point is within a land polygon
        is_within_ocean = 1-np.prod((joined['index_right'].notnull()).to_list())
        return is_within_ocean

    def find_patches_in_ocean(self):
        nitem = np.prod(self.ds_size.values())
        nitem_bytime = np.prod(list(self.ds_size.values())[1:])
        coords = []
        for i in range(nitem_bytime):
            if np.mod(i,1000)==0:
                print(i)
            sl = {
                dim: slice(self.strides.get(dim, 1) * idx,
                                   self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                    for dim, idx in zip(self.ds_size.keys(),
                                            np.unravel_index(i, tuple(self.ds_size.values())))
                }
            coords.append(xr.Dataset(coords=dict(xc=self.xc[sl["xc"].start:sl["xc"].stop],
                                                 yc=self.yc[sl["yc"].start:sl["yc"].stop],
                                                 time=self.times[sl["time"].start:sl["time"].stop],
                                                )
                                    ).transpose('time', 'yc', 'xc')
                        )
        idx0 = []
        for k in range(len(coords)):
            if np.mod(k,1000)==0:
                print(k)
            ix = [find_idx(self.xc,x) for x in coords[k].xc.values]
            iy = [find_idx(self.yc,y) for y in coords[k].yc.values]
            item_lon = self.lon[iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)].flatten()
            item_lat = self.lat[iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)].flatten()
            if self.item_in_ocean(item_lon, item_lat):
                idx0.append(k)
        # repeat for the number of times
        # idx = np.concatenate([ idx0+(nitem_bytime*i) for i in range(list(self.ds_size.values())[0]) ])
        return idx0

    def __getitem__(self, item):
        
        gc.collect()
        if self.subsel_patch:
            sl = {
                dim: slice(self.strides.get(dim, 1) * idx,
                           self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                for dim, idx in zip(self.ds_size.keys(),
                    np.unravel_index(int(self.idx_patches_in_ocean[item]), tuple(self.ds_size.values())))
                 }
        else:
            sl = {
                dim: slice(self.strides.get(dim, 1) * idx,
                           self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                for dim, idx in zip(self.ds_size.keys(),
                                    np.unravel_index(item, tuple(self.ds_size.values())))
                 }

        # before reading, check if batch is in land or ocean 
        # if land, return array filled with nan values
        coords = xr.Dataset(coords=dict(xc=self.xc[sl["xc"].start:sl["xc"].stop],
                               yc=self.yc[sl["yc"].start:sl["yc"].stop],
                               time=self.times[sl["time"].start:sl["time"].stop],
                            )
                        ).transpose('time', 'yc', 'xc')
        ix = [find_idx(self.xc,x) for x in coords.xc.values]
        iy = [find_idx(self.yc,y) for y in coords.yc.values]
        item_lon = self.lon[iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)].flatten()
        item_lat = self.lat[iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)].flatten()
        item_mask = self.mask[iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)]
        #item_mask = self.mask.isel(xc=sl["xc"],yc=sl["yc"])

        if not self.item_in_ocean(item_lon, item_lat):
            item = coords
            size = [coords.sizes[d] for d in ["time","yc","xc"]]
            item = item.update({"input":(("time","yc","xc"),np.full(size,np.nan))})
            item['tgt'] = item.input
            item['coarse'] = item.input
            item['land_mask'] = item.input
            item = item[[*contrib.DMI.ASIP_OSISAF.data.TrainingItem._fields]].to_array()
            if self.return_coords:
                return item.coords.to_dataset()[list(self.patch_dims)]
            item = item.data.astype(np.float32)
            if self.postpro_fn is not None:
                return self.postpro_fn(item)
            return item

        if self.load_data:
            asip = self.asip.isel(time=sl["time"])
            ix = [ x in asip.xc for x in self.xc[sl["xc"].start:sl["xc"].stop] ]
            iy = [ y in asip.yc for y in self.yc[sl["yc"].start:sl["yc"].stop] ]
            asip = self.asip.sel(xc=(self.xc[sl["xc"].start:sl["xc"].stop])[ix],
                                 yc=(self.yc[sl["yc"].start:sl["yc"].stop])[iy])
        else:
            # read asip
            start = sl["time"].start
            end = sl["time"].stop
            if self.use_mf:
                asip = xr.open_mfdataset(self.asip_paths[start:end]).isel(xc=sl["xc"],
                                                                          yc=sl["yc"])
            else:
                #asip = xr.concat([xr.open_dataset(self.asip_paths[t],lock=False,cache=False).isel(xc=sl["xc"],
                #                                                           yc=sl["yc"])
                #              for t in range(start,end)],dim="time")
                asip = concatenate(self.asip_paths[np.arange(start,end)],var="sic",
                                   slices={"xc": sl["xc"],"yc":sl["yc"]})
                asip.close()

        # pad
        nt, ny, nx = tuple(self.patch_dims[d] for d in ['time', 'yc', 'xc'])
        if ( (asip.sizes['yc'] !=ny) or (asip.sizes['xc']!=nx) ):
            ix = [find_idx(self.xc,x) for x in self.xc[sl["xc"].start:sl["xc"].stop]]
            iy = [find_idx(self.yc,y) for y in self.yc[sl["yc"].start:sl["yc"].stop]]
            padded_patch = xr.Dataset(
                        coords={
                            "time": asip.time,
                            "xc": self.xc[sl["xc"].start:sl["xc"].stop],
                            "yc": self.yc[sl["yc"].start:sl["yc"].stop],
                            "lon": (["yc","xc"], self.lon[iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)]),
                            "lat": (["yc","xc"], self.lat[iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)])
                            })
            asip = xr.align(padded_patch,asip,join="left")[1]
            #item_mask = xr.align(padded_patch,item_mask,join="left")[1] 
            asip.close()

        lat = asip.lat.values
        lon = asip.lon.values
        asip_swath_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)

        if self.load_data:
            osisaf = self.osisaf.isel(time=sl["time"])
        else:
            # read osisaf
            if self.use_mf:
                osisaf = xr.open_mfdataset(self.osisaf_paths[start:end])
            else:
                #osisaf = xr.concat([xr.open_dataset(self.osisaf_paths[t],lock=False,cache=False) for t in range(start,end)],dim="time")
                osisaf = concatenate(self.osisaf_paths[np.arange(start,end)],var="ice_conc")
            osisaf.close()

        osisaf_sic = osisaf.ice_conc.values
        osisaf_lon = osisaf.lon.values
        osisaf_lat = osisaf.lat.values
        osisaf_swath_def = pyresample.geometry.SwathDefinition(lons=osisaf_lon, lats=osisaf_lat)

        # interpolate osisaf on asip
        pool = torch.nn.AvgPool2d(10, stride=10)
        up = torch.nn.Upsample(scale_factor=10, mode='bilinear', align_corners=True)
        asip_coarse = np.zeros(asip.sic.shape)
        for i in range(len(asip.time)):
            asip_coarse[i] = pyresample.kd_tree.resample_nearest(osisaf_swath_def,
                                                                 osisaf_sic[i],
                                                                 asip_swath_def,
                                                                 radius_of_influence=30000,
                                                                 fill_value=np.nan)
            # pool and bilinear upsample OSISAF
            asip_coarse[i] = up(pool(torch.tensor(asip_coarse[i]).unsqueeze(0)).unsqueeze(0))[0,0,:,:].numpy()
        asip = asip.update({"sic_coarse":(("time","yc","xc"),asip_coarse)})

        # covariates
        try:
            if self.load_data:
                covs = [ self.covs[i].isel(time=sl["time"]) for i in range(len(self.covs)) ]
            else:
                covs = []
                for i in range(len(self.covariates)):
                    covs.append(concatenate(self.covariates_paths[np.arange(start,end)],
                                            var=self.covariates[i])
                                )
            covs_lon = [ ( np.mod((lon + 180),360) - 180) for lon in covs[0].longitude.values ]
            covs_lat = covs[0].latitude.values
            covs_lon, covs_lat = np.meshgrid(covs_lon, covs_lat)
            covs_lon = covs_lon.astype(covs_lat.dtype)
            covs_swath_def = pyresample.geometry.SwathDefinition(lons=covs_lon, lats=covs_lat)
            # interpolate covariates on asip
            pool = torch.nn.AvgPool2d(40, stride=40)
            up = torch.nn.Upsample(scale_factor=40, mode='bilinear', align_corners=True)
            for k in range(len(self.covariates)):
                covs_coarse = np.zeros(asip.sic.shape)
                for i in range(len(asip.time)):
                    covs_coarse[i] = pyresample.kd_tree.resample_nearest(covs_swath_def,
                                                                     covs[k][self.covariates[k]].values[i],
                                                                     asip_swath_def,
                                                                     radius_of_influence=30000,
                                                                     fill_value=np.nan)
                    # pool and bilinear upsample ERA5
                    covs_coarse[i] = up(pool(torch.tensor(covs_coarse[i]).unsqueeze(0)).unsqueeze(0))[0,0,:,:].numpy()
                asip = asip.update({self.covariates[k]:(("time","yc","xc"), covs_coarse)})
            # create final item
            asip = asip.transpose('time', 'yc', 'xc')
            inp = asip.rename_vars({"sic":"input"})
            tgt = asip.rename_vars({"sic":"tgt"})
            coarse = asip.rename_vars({"sic_coarse":"coarse"})
        except:
            # fill tgt with nan to discard the batch when evaluating
            asip = asip.update({'tgt':(("time","yc","xc"), np.full(asip.sic.data.shape,np.nan))})
            covs_coarse = np.zeros(asip.sic.shape)
            # create final item
            asip = asip.transpose('time', 'yc', 'xc')
            inp = asip.rename_vars({"sic":"input"})
            tgt = asip 
            coarse = asip.rename_vars({"sic_coarse":"coarse"})

        item = inp
        item['tgt'] = tgt.tgt
        item['coarse'] = coarse.coarse
        item['lonv'] = inp.lon
        item['latv'] = inp.lat
        for k in range(len(self.covariates)):
            item[self.covariates[k]] = asip[self.covariates[k]]
        item = item.update({'land_mask': (("time","yc","xc"),np.transpose(np.dstack([item_mask]*len(item['tgt'])),(2,0,1)))})
        item = item[[*contrib.DMI.ASIP_OSISAF.data.TrainingItem._fields]].to_array()

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]
        
        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            item = self.postpro_fn(item)
        return item

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

    def reconstruct_from_items_cpu(self, items, weight=None):
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
        w = xr.DataArray(weight, dims=list(self.patch_dims.keys()))
        
        coords = self.get_coords()
        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
        dims = new_dims + list(coords[0].dims)
        das = [xr.DataArray(it.numpy(), dims=dims, coords=co.coords)
               for  it, co in zip(items, coords)]

        da_shape = dict(zip(coords[0].dims, list(self.da.dims.values())[-len(coords[0].dims):]))
        new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

        rec_da = xr.DataArray(
                np.zeros([*new_shape.values(), *da_shape.values()]),
                dims=dims,
                coords={d: self.da[d] for d in self.patch_dims} 
        )
        count_da = xr.zeros_like(rec_da)

        for da in das:
            rec_da.loc[da.coords] = rec_da.sel(da.coords) + da * w
            count_da.loc[da.coords] = count_da.sel(da.coords) + w
        
        return rec_da / count_da

    def reconstruct_from_items(self, items, index_time, weight=None):
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
            weight = np.ones(list(1,self.patch_dims.values()[1],self.patch_dims.values()[2]))
        w = torch.tensor(weight)#.cuda()

        nvars = items[0].shape[0]
        result_tensor = torch.zeros(size=(nvars,
                                       1,#self.da_dims["time"],
                                       self.da_dims["yc"],
                                       self.da_dims["xc"]))#.cuda()
        count_tensor = torch.zeros(size=(nvars,
                                       1,#self.da_dims["time"],
                                       self.da_dims["yc"],
                                       self.da_dims["xc"]))#.cuda()

        coords = self.get_coords()
        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
        dims = new_dims + ["time", "yc", "xc"] #+list(coords[0].dims)

        for idx in range(len(items)):
            it = [find_idx(self.times, pd.to_datetime(np.datetime64(t)).to_pydatetime()) \
                    for t in coords[idx].time.values]
            it = it[-1]
            ix = [find_idx(self.xc,x) for x in coords[idx].xc.values]
            iy = [find_idx(self.yc,y) for y in coords[idx].yc.values]
            #result_tensor[:,it[0]:(it[-1]+1),iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)] += items[idx].cpu() * w
            #count_tensor[:,it[0]:(it[-1]+1),iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)] += w
            result_tensor[:,[0],iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)] += items[idx].cpu() * w
            count_tensor[:,[0],iy[0]:(iy[-1]+1),ix[0]:(ix[-1]+1)] += w

        result_tensor /= count_tensor#.cpu()
        result_tensor = result_tensor.numpy()
        #result_tensor = torch.where(result_tensor==0.,np.nan,result_tensor).numpy()
        result_da = xr.DataArray(
            result_tensor,
            dims=dims,
            coords={#"time": self.times,
                    "time": np.atleast_1d(self.times[index_time]),
                    "xc": self.xc,
                    "yc": self.yc,
                    "lon": (["yc","xc"], self.lon),
                    "lat": (["yc","xc"], self.lat)
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

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, inp_ds, aug_factor, aug_only=False, noise_sigma=None):
        self.aug_factor = aug_factor
        self.aug_only = aug_only
        self.inp_ds = inp_ds
        self.perm = np.random.permutation(len(self.inp_ds))
        self.noise_sigma = noise_sigma

    def __len__(self):
        return len(self.inp_ds) * (1 + self.aug_factor - int(self.aug_only))

    def __getitem__(self, idx):
        if self.aug_only:
            idx = idx + len(self.inp_ds)

        if idx < len(self.inp_ds):
            return self.inp_ds[idx]

        tgt_idx = idx % len(self.inp_ds)
        perm_idx = tgt_idx
        for _ in range(idx // len(self.inp_ds)):
            perm_idx = self.perm[perm_idx]
        
        item = self.inp_ds[tgt_idx]
        perm_item = self.inp_ds[perm_idx]

        noise = np.zeros_like(item.input, dtype=np.float32)
        if self.noise_sigma is not None:
            noise = np.random.randn(*item.input.shape).astype(np.float32) * self.noise_sigma

        return item._replace(input=noise + np.where(np.isfinite(perm_item.input),
                             item.tgt, np.full_like(item.tgt,np.nan)))

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
    def __init__(self, asip_paths, osisaf_paths,
                 covariates_paths, covariates,
                 mask_path,
                 domain_name, domains,
                 xrds_kw, dl_kw, 
                 norm_stats, norm_stats_covs,
                 aug_kw=None, res=0.05, pads=[False,False,False], 
                 subsel_path="/dmidata/users/maxb/4dvarnet-starter/contrib/DMI/ASIP_OSISAF",
                 **kwargs):
        
        super().__init__()
        self.asip_paths = asip_paths
        self.osisaf_paths = osisaf_paths
        self.covariates_paths = covariates_paths
        self.covariates = covariates
        self.mask_path = mask_path
        self.domain_name = domain_name
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.aug_kw = aug_kw if aug_kw is not None else {}
        self.res = res
        self.pads = pads

        asip_base = xr.open_dataset(self.asip_paths[0])
        xc_orig = asip_base.xc.data
        yc_orig = asip_base.yc.data
        self.lon = asip_base.lon.data
        self.lat = asip_base.lat.data
        self.xc = xc_orig
        self.yc = yc_orig

        self._norm_stats = norm_stats
        self._norm_stats_covs = norm_stats_covs

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._post_fn = None

        self.subsel_path = subsel_path

        if not os.path.isfile(self.mask_path):
            mask = xr.Dataset(
                        coords={
                            "xc": self.xc,
                            "yc": self.yc,
                            "lon": (["yc","xc"], self.lon),
                            "lat": (["yc","xc"], self.lat)
                            })
            land_mask = np.zeros((len(self.yc),len(self.xc)))
            land_50m = cfeature.NaturalEarthFeature('physical','land','50m')
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
            self.mask = mask.update({"mask":(("yc","xc"),land_mask)})
            self.mask.to_netcdf(self.mask_path)
            self.mask = mask.mask
        else:
            mask = xr.open_dataset(self.mask_path)
            self.mask = mask.mask
            mask.close()

    def norm_stats(self):
        return self._norm_stats

    def norm_stats_covs(self):
        return self._norm_stats_covs

    def post_fn(self):
        m, s = self.norm_stats()
        m_covs, s_covs = self.norm_stats_covs()
        normalize_zscore = lambda item, mean, std: (item - mean) / std
        minmax_scale = lambda item, min, max: (item-min)/(max-min)
        normalize = lambda item: (item - m) / (s-m)
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(input=normalize(item.input)),
            lambda item: item._replace(coarse=normalize(item.coarse)),
            lambda item: item._replace(tgt=normalize(item.tgt)),
            lambda item: item._replace(land_mask=item.land_mask),
            lambda item: item._replace(latv=minmax_scale(item.latv,50,90)),
            lambda item: item._replace(lonv=minmax_scale(item.lonv,-180,180)),
            lambda item: item._replace(t2m=normalize_zscore(item.t2m,m_covs["t2m"],s_covs["t2m"])),
            lambda item: item._replace(istl1=normalize_zscore(item.istl1,m_covs["istl1"],s_covs["istl1"])),
            lambda item: item._replace(siconc=minmax_scale(item.siconc,m_covs["siconc"],s_covs["siconc"])),
            lambda item: item._replace(sst=normalize_zscore(item.sst,m_covs["sst"],s_covs["sst"])),
            lambda item: item._replace(skt=normalize_zscore(item.skt,m_covs["skt"],s_covs["skt"]))
        ])

    def post_fn_rand(self):
        m, s = self.norm_stats()
        m_covs, s_covs = self.norm_stats_covs()
        normalize_zscore = lambda item, mean, std: (item - mean) / std
        minmax_scale = lambda item, min, max: (item-min)/(max-min)
        normalize = lambda item: (item - m) / (s-m)
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(input=normalize(self.rand_obs(item.input))),
            lambda item: item._replace(coarse=normalize(item.coarse)),
            lambda item: item._replace(tgt=normalize(item.tgt)),
            lambda item: item._replace(land_mask=item.land_mask),
            lambda item: item._replace(latv=minmax_scale(item.latv,50,90)),
            lambda item: item._replace(lonv=minmax_scale(item.lonv,-180,180)),
            lambda item: item._replace(t2m=normalize_zscore(item.t2m,m_covs["t2m"],s_covs["t2m"])),
            lambda item: item._replace(istl1=normalize_zscore(item.istl1,m_covs["istl1"],s_covs["istl1"])),
            lambda item: item._replace(siconc=minmax_scale(item.siconc,m_covs["siconc"],s_covs["siconc"])),
            lambda item: item._replace(sst=normalize_zscore(item.sst,m_covs["sst"],s_covs["sst"])),
            lambda item: item._replace(skt=normalize_zscore(item.skt,m_covs["skt"],s_covs["skt"]))
         ])

    def rand_obs(self, gt_item, obs=True):
        obs_mask_item = ~np.isnan(gt_item)
        _obs_item = gt_item
        dtime = self.xrds_kw['patch_dims']['time']
        dxc = self.xrds_kw['patch_dims']['xc']
        dyc = self.xrds_kw['patch_dims']['yc']
        for t_ in range(dtime):
            obs_mask_item_t_ = obs_mask_item[t_]
            if np.sum(obs_mask_item_t_)>.25*dyc*dxc:
                obs_obj = .5*np.sum(obs_mask_item_t_)
                while  np.sum(obs_mask_item_t_)>= obs_obj:
                    half_patch_height = np.random.randint(2,10)
                    half_patch_width = np.random.randint(2,10)
                    idx_yc = np.random.randint(0,dyc)
                    idx_xc = np.random.randint(0,dxc)
                    obs_mask_item_t_[np.max([0,idx_yc-half_patch_height]):np.min([dyc,idx_yc+half_patch_height+1]),
                                     np.max([0,idx_xc-half_patch_width]):np.min([dxc,idx_xc+half_patch_width+1])] = 0
                obs_mask_item[t_] = obs_mask_item_t_
        obs_mask_item = (obs_mask_item == 1)
        if obs==True:
            obs_item = np.where(obs_mask_item, _obs_item, np.nan)
            return obs_item
        else:
            tgt_item = np.where(obs_mask_item, np.nan, _obs_item)
            return tgt_item

    def rand_obs2(self, gt_item, obs=True):

        npatch = 500
        obs_mask_item = ~np.isnan(gt_item)
        _obs_item = gt_item
        dtime = self.xrds_kw['patch_dims']['time']
        dxc = self.xrds_kw['patch_dims']['xc']
        dyc = self.xrds_kw['patch_dims']['yc']

        # define random size of additional wholes
        half_patch_height = np.random.randint(2,10,(dtime,npatch))
        half_patch_width = np.random.randint(2,10,(dtime,npatch))
        idx_yc = np.random.randint(0,dyc,(dtime,npatch))
        idx_xc = np.random.randint(0,dxc,(dtime,npatch))

        # define objective of missing data (50% of the initial obs)
        obs_obj = .5 * np.sum(obs_mask_item,axis=(1,2))
        
        # define 3d-numpy array index of new mask 
        posy_start = (idx_yc-half_patch_height).clip(min=0)
        posy_stop = (idx_yc+half_patch_height).clip(max=dyc)
        posx_start = (idx_xc-half_patch_width).clip(min=0)
        posx_stop = (idx_xc+half_patch_width).clip(max=dxc)
        id_yc = [ [ np.arange(k,l) for k,l in zip(posy_start[t],posy_stop[t])] for t in range(dtime)]
        id_xc = [ [ np.arange(k,l) for k,l in zip(posx_start[t],posx_stop[t])] for t in range(dtime)]
        idx = np.concatenate([ np.array(np.meshgrid([t],id_yc[t][p],id_xc[t][p])).T.reshape(-1,3) \
                               for t, p in np.array(np.meshgrid(np.arange(dtime),np.arange(npatch))).T.reshape(-1,2) ])
        # clip the number of new missing data according to the objectives
        idx_t = [ np.where(idx[:,0]==t)[0][0] for t in range(dtime) ]
        idx_t.append(len(idx)-1)
        stop_idx = [ idx_t[t] + np.argwhere( np.cumsum(obs_mask_item[idx[idx_t[t]:idx_t[t+1],0],
                                                                     idx[idx_t[t]:idx_t[t+1],1],
                                                                     idx[idx_t[t]:idx_t[t+1],2]]==1) >= obs_obj[t])[0] \
                     if 2*obs_obj[t]/(dyc*dxc)>.4 else idx_t[t] for t in range(dtime) ]
        idx_final = np.concatenate([np.arange(idx_t[t],stop_idx[t]) for t in range(dtime)])
        # fill the new mask with 0
        obs_mask_item[idx[idx_final,0],idx[idx_final,1],idx[idx_final,2]] = 0
        # return the new item
        if obs==True:
            obs_item = np.where(obs_mask_item, _obs_item, np.nan)
            return obs_item
        else:
            tgt_item = np.where(obs_mask_item, np.nan, _obs_item)
            return tgt_item

    def setup(self, stage='test'):

        def select_paths_from_dates(files, times):
            # compute list of dates from domain
            if isinstance(times, list):
                dates = []
                times = []
                for _ in range(len(times)):
                    start = datetime.datetime.strptime(times.start, "%Y-%m-%d")
                    end = datetime.datetime.strptime(times.stop, "%Y-%m-%d")
                    dates.extend([(start + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(0, (end-start).days)])     
                    times.extend([start + datetime.timedelta(days=x) for x in range(0, (end-start).days)])
            else:
                start = datetime.datetime.strptime(times.start, "%Y-%m-%d")
                end = datetime.datetime.strptime(times.stop, "%Y-%m-%d")
                dates = [ (start + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(0, (end-start).days) ]
                times = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
            # subselection of paths
            files = np.sort([ f for f in files if any(s in f for s in dates) ])
            return files, np.array(times)

        post_fn = self.post_fn()
        post_fn_rand = self.post_fn_rand()

        train_asip_paths, train_times = select_paths_from_dates(self.asip_paths, self.domains['train']['time'])
        train_osisaf_paths, _ = select_paths_from_dates(self.osisaf_paths, self.domains['train']['time'])
        train_covariates_paths, _ = select_paths_from_dates(self.covariates_paths, self.domains['train']['time'])
        self.train_ds = XrDataset(
            train_asip_paths, 
            train_osisaf_paths,
            train_covariates_paths,
            self.covariates,
            self.mask,
            train_times,
            **self.xrds_kw, postpro_fn=post_fn_rand,
            res = self.res, 
            pad=self.pads[0],
            subsel_patch=True,
            subsel_patch_path=self.subsel_path+"/patch_in_ocean_"+self.domain_name+\
                              "_patch_"+str(self.xrds_kw["patch_dims"]["yc"])+"_"+\
                              str(self.xrds_kw["strides"]["yc"])+".txt"
        )
        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        if isinstance(self.domains['val']['time'], slice):
            val_asip_paths, val_times = select_paths_from_dates(self.asip_paths, self.domains['val']['time'])
            val_osisaf_paths, _ = select_paths_from_dates(self.osisaf_paths, self.domains['val']['time'])
            val_covariates_paths, _ = select_paths_from_dates(self.covariates_paths, self.domains['val']['time'])
            self.val_ds = XrDataset(
                val_asip_paths,
                val_osisaf_paths,
                val_covariates_paths,
                self.covariates,
                self.mask,
                val_times,
                **self.xrds_kw, postpro_fn=post_fn,
                res =self.res,
                pad=self.pads[1],
                stride_test=True,
                load_data=False,
                subsel_patch=True,
                subsel_patch_path=self.subsel_path+"/patch_in_ocean_test_"+self.domain_name+\
                                  "_patch_"+str(self.xrds_kw["patch_dims"]["yc"])+"_"+\
                                  str(self.xrds_kw["strides"]["yc"])+".txt"
            )
        else:
            self.val_ds = ConcatDataset([XrDataset(
                   select_paths_from_dates(self.asip_paths, sl)[0],
                   select_paths_from_dates(self.osisaf_paths, sl)[0],
                   select_paths_from_dates(self.covariates_paths, sl)[0],
                   self.covariates,
                   self.mask,
                   select_paths_from_dates(self.asip_paths, sl)[1],
                   **self.xrds_kw, postpro_fn=post_fn,
                   res = self.res, pad=self.pads[1],
                   stride_test=True,
                   load_data=False,
                   subsel_patch=True,
                   subsel_patch_path=self.subsel_path+"/patch_in_ocean_test_"+self.domain_name+\
                                     "_patch_"+str(self.xrds_kw["patch_dims"]["yc"])+"_"+\
                                     str(self.xrds_kw["strides"]["yc"])+".txt"
                   ) for sl in self.domains['val']['time']])

        test_asip_paths, test_times = select_paths_from_dates(self.asip_paths, self.domains['test']['time'])
        test_osisaf_paths, _ = select_paths_from_dates(self.osisaf_paths, self.domains['test']['time'])
        test_covariates_paths, _ = select_paths_from_dates(self.covariates_paths, self.domains['test']['time'])
        self.test_ds = XrDataset(
            test_asip_paths,
            test_osisaf_paths,
            test_covariates_paths,
            self.covariates,
            self.mask,
            test_times,
            **self.xrds_kw, postpro_fn=post_fn,
            res = self.res,
            pad=self.pads[2],
            stride_test=True,
            load_data=True,
            subsel_patch=True,
            subsel_patch_path=self.subsel_path+"/patch_in_ocean_test_"+self.domain_name+\
                              "_patch_"+str(self.xrds_kw["patch_dims"]["yc"])+"_"+\
                              str(self.xrds_kw["strides"]["yc"])+".txt"
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        cb_sampler = CustomBatchSampler(self.val_ds,
                                        batch_size=self.dl_kw["batch_size"])
        return torch.utils.data.DataLoader(self.val_ds, #shuffle=False, 
                                           batch_sampler=cb_sampler, 
                                           num_workers=self.dl_kw["num_workers"])#,**self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, 
                                           **self.dl_kw)

class ConcatDataModule(BaseDataModule):

    def setup(self, stage='test'):
        post_fn = self.post_fn()
        self.train_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn_rand,)
            for domain in self.domains['train']
        ])
        if self.aug_factor >= 1:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn_rand,)
            for domain in self.domains['val']
        ])
        self.test_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['test']
        ])
