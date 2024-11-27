import pytorch_lightning as pl
import numpy as np
import torch.utils.data
import xarray as xr
import itertools
import functools as ft
import tqdm
from collections import namedtuple
from torch.utils.data import  ConcatDataset
from random import sample 
import contrib
import datetime
import pyresample

TrainingItem = namedtuple(
    'TrainingItem', ['input', 'tgt', 'coarse']
)

TrainingItem_wgeo = namedtuple(
    'TrainingItem_wgeo', ['input', 'tgt', 'coarse', 'latv', 'lonv', 'land_mask', 'topo', 'fg_std']
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
            times,
            patch_dims, domain_limits=None, strides=None,
            strides_test=None,
            check_full_scan=False, check_dim_order=False,
            postpro_fn=None,
            res=0.05,
            pad = False,
            stride_test=False,
            limit_num_coords=None
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
        self.times = times
        self.patch_dims = patch_dims
        self.strides = strides or {}
        if stride_test:
            self.strides = strides_test or {}
        self.res = res
        self.pad = pad
        self.limit_num_coords = limit_num_coords

        # store coords 
        times = np.arange(len(self.asip_paths))
        xc_orig = xr.open_dataset(self.asip_paths[0]).sel(**(domain_limits or {})).xc.data
        yc_orig = xr.open_dataset(self.asip_paths[0]).sel(**(domain_limits or {})).yc.data
        self.xc = xc_orig
        self.yc = yc_orig

        # pad
        nt, ny, nx = (len(times),len(xc_orig),len(yc_orig))
        if self.pad:
            pad_x = find_pad(self.patch_dims['xc'], self.strides['xc'], nx)
            pad_y = find_pad(self.patch_dims['yc'], self.strides['yc'], ny)
            pad_ = {'xc':(pad_x[0],pad_x[1]),
                    'yc':(pad_y[0],pad_y[1])}
            dx = [pad_ *self.res for pad_ in pad_x]
            dy = [pad_ *self.res for pad_ in pad_y]
            new_xc = np.concatenate((np.linspace(xc_orig[0]-dx[0],
                                                 xc_orig[0],
                                                 pad_x[0],endpoint=False),
                                     xc_orig,
                                     np.linspace(xc_orig[-1]+self.res,
                                                 xc_orig[-1]+dx[1]+ self.res,
                                                 pad_x[1],endpoint=False))) 
            new_yc = np.concatenate((np.linspace(yc_orig[0]-dy[0],
                                                 yc_orig[0],
                                                 pad_y[0],endpoint=False),
                                     yc_orig,
                                     np.linspace(yc_orig[-1]+self.res,
                                                 yc_orig[-1]+dy[1]+ self.res,
                                                 pad_y[1],endpoint=False))) 
            yc_padded = np.round(new_yc,2)
            xc_padded = np.round(new_xc,2)
            da_dims = dict(time=nt, xc=len(xc_padded), yc=len(yc_padded))
            self.xc = xc_padded
            self.yc = yc_padded
        else:
            da_dims = dict(time=nt, xc=len(xc_orig), yc=len(yc_orig))
        print(da_dims)

        self.ds_size = {
            dim: max((da_dims[dim] - patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in patch_dims
        }

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
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords_old(self):
        self.return_coords = True
        coords = []
        try:
            if self.limit_num_coords is None:
                for i in range(len(self)):
                    coords.append(self[i])
            else:
                for i in sample(range(0,len(self)),self.limit_num_coords):
                    coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            if self.limit_num_coords is None:
                for i in range(len(self)):
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
            else:
                for i in sample(range(0,len(self)),self.limit_num_coords):
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

    def __getitem__(self, item, use_mf=False):
        sl = {
                dim: slice(self.strides.get(dim, 1) * idx,
                           self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                for dim, idx in zip(self.ds_size.keys(),
                                    np.unravel_index(item, tuple(self.ds_size.values())))
             }

        # read asip
        start = sl["time"].start
        end = sl["time"].stop
        if use_mf:
            asip = xr.open_mfdataset(self.asip_paths[start:end]).isel(xc=sl["xc"],
                                                                      yc=sl["yc"])
        else:
            asip = xr.concat([xr.open_dataset(self.asip_paths[t]).isel(xc=sl["xc"],
                                                                       yc=sl["yc"]) \
                              for t in range(start,end)],dim="time")

        # pad
        nt, ny, nx = tuple(self.patch_dims[d] for d in ['time', 'yc', 'xc'])
        print(asip)
        if ( (asip.sizes['yc'] !=ny) or (asip.sizes['xc']!=nx) ):
            padded_patch = xr.Dataset(
                        coords=dict(
                            xc=self.xc[sl["xc"].start:sl["xc"].stop],
                            yc=self.yc[sl["yc"].start:sl["yc"].stop],
                            time=asip.time,
                     ))
            asip = xr.align(patch,asip,join="left")[1]
            print(asip)

        lat = asip.lat.values
        lon = asip.lon.values
        asip_swath_def = pyresample.geometry.SwathDefinition(lons=lon, lats=lat)
        asip.close()

        # read osisaf
        if use_mf:
            osisaf = xr.open_mfdataset(self.osisaf_paths[start:end])
        else:
            osisaf = xr.concat([xr.open_dataset(self.osisaf_paths[t]) for t in range(start,end)],dim="time")
        osisaf_sic = osisaf.ice_conc.values
        osisaf_lon = osisaf.lon.values
        osisaf_lat = osisaf.lat.values
        osisaf_swath_def = pyresample.geometry.SwathDefinition(lons=osisaf_lon, lats=osisaf_lat)
        osisaf.close()

        # interpolate osisaf on asip
        asip_coarse = np.zeros(asip.sic.shape)
        for i in range(len(asip.time)):
            asip_coarse[i] = pyresample.kd_tree.resample_nearest(osisaf_swath_def, osisaf_sic[i], asip_swath_def, radius_of_influence=30000, fill_value=np.nan)
        asip = asip.update({"sic_coarse":(("time","yc","xc"),asip_coarse)})
 
        # create final item
        inp = asip.rename_vars({"sic":"input"}).transpose('time', 'yc', 'xc')
        tgt = asip.rename_vars({"sic":"tgt"}).transpose('time', 'yc', 'xc')
        coarse = asip.rename_vars({"sic_coarse":"coarse"}).transpose('time', 'yc', 'xc')

        item = inp
        item['tgt'] = tgt.tgt
        item['coarse'] = coarse.coarse
        item = item[[*contrib.DMI.ASIP_OSISAF.data.TrainingItem._fields]].to_array()

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item

    def reconstruct(self, batches, weight=None):
        """
        takes as input a list of np.ndarray of dimensions (b, *, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims

        batches: list of torch tensor correspondin to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting 
        """

        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, weight)

    def reconstruct_from_items(self, items, weight=None):
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

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, asip_paths, osisaf_paths, domains, xrds_kw, dl_kw, norm_stats,
                 aug_kw=None, res=0.05, pads=[False,False,False], **kwargs):
        
        super().__init__()
        self.asip_paths = asip_paths
        self.osisaf_paths = osisaf_paths
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.aug_kw = aug_kw if aug_kw is not None else {}
        self.res = res
        self.pads = pads
        self._norm_stats = norm_stats

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._post_fn = None

    def norm_stats(self):
        return self._norm_stats

    def post_fn(self):
        m, s = self.norm_stats()
        normalize = lambda item: (item - m) / s
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(input=normalize(item.input)),
            lambda item: item._replace(coarse=normalize(item.coarse)),
            lambda item: item._replace(tgt=normalize(item.tgt)),
        ])

    def post_fn_rand(self):
        m, s = self.norm_stats()
        normalize = lambda item: (item - m) / s
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(input=normalize(self.rand_obs(item.input-item.coarse))),
            lambda item: item._replace(coarse=normalize(item.coarse)),
            lambda item: item._replace(tgt=normalize(item.tgt-item.coarse)),
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
            return files, times

        post_fn = self.post_fn()
        post_fn_rand = self.post_fn_rand()

        train_asip_paths, train_times = select_paths_from_dates(self.asip_paths, self.domains['train']['time'])
        train_osisaf_paths, _ = select_paths_from_dates(self.osisaf_paths, self.domains['train']['time'])
        self.train_ds = XrDataset(
            train_asip_paths, 
            train_osisaf_paths,
            train_times,
            **self.xrds_kw, postpro_fn=post_fn_rand,
            res = self.res, limit_num_coords = 10000,
            pad=self.pads[0]
        )
        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        if isinstance(self.domains['val']['time'], slice):
            val_asip_paths, val_times = select_paths_from_dates(self.asip_paths, self.domains['val']['time'])
            val_osisaf_paths, _ = select_paths_from_dates(self.osisaf_paths, self.domains['val']['time'])
            self.val_ds = XrDataset(
                val_asip_paths,
                val_osisaf_paths,
                val_times,
                **self.xrds_kw, postpro_fn=post_fn,
                res =self.res,
                pad=self.pads[1]
            )
        else:
           self.val_ds =ConcatDataset([
              XrDataset(
                select_paths_from_dates(self.asip_paths, sl)[0],
                select_paths_from_dates(self.osisaf_paths, sl)[0],
                select_paths_from_dates(self.asip_paths, sl)[1],
                **self.xrds_kw, postpro_fn=post_fn,
                res = self.res, pad=self.pads[1]
              ) for sl in self.domains['val']['time'] ]
            )

        test_asip_paths, test_times = select_paths_from_dates(self.asip_paths, self.domains['test']['time'])
        test_osisaf_paths, _ = select_paths_from_dates(self.osisaf_paths, self.domains['test']['time'])
        self.test_ds = XrDataset(
            test_asip_paths,
            test_osisaf_paths,
            test_times,
            **self.xrds_kw, postpro_fn=post_fn,
            res = self.res,
            pad=self.pads[2],
            stride_test=True
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

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
