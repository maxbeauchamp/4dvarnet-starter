import pytorch_lightning as pl
import numpy as np
import torch.utils.data
import xarray as xr
import metpy.calc as mpcalc
import itertools
import functools as ft
import tqdm
from pyinterp.fill import gauss_seidel
from pyinterp.backends.xarray import Grid3D
from collections import namedtuple

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class IncompleteScanConfiguration(Exception):
    pass

class DangerousDimOrdering(Exception):
    pass

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
            self, da, patch_dims, domain_limits=None, strides=None,
            check_full_scan=False, check_dim_order=False,
            postpro_fn=None,
            resize_factor=1,
            res=0.05
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
        self.da = da.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        self.res = res

        # resizing
        if resize_factor!=1:
            self.da = self.da.coarsen(lon=resize_factor,boundary='trim').mean(skipna=True).coarsen(lat=resize_factor,boundary='trim').mean(skipna=True)
            self.res*=resize_factor

        # store coords 
        lon_orig = self.da.lon.data
        lat_orig = self.da.lat.data

        da_dims = dict(zip(self.da.dims, self.da.shape))
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

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {
                dim: slice(self.strides.get(dim, 1) * idx,
                           self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                for dim, idx in zip(self.ds_size.keys(),
                                    np.unravel_index(item, tuple(self.ds_size.values())))
                }
        item =  self.da.isel(**sl)

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

        da_shape = dict(zip(coords[0].dims, self.da.shape[-len(coords[0].dims):]))
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
    
        return rec_das

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

def compute_grad(da):
    return np.sqrt(da.differentiate("lon")**2 + da.differentiate("lat")**2)

def add_geo_attrs(da):
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")
    return da

def vort(da):
    return mpcalc.vorticity(
            *mpcalc.geostrophic_wind(
                da.pipe(add_geo_attrs).assign_attrs(units="m").metpy.quantify()
            )
        ).metpy.dequantify()

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, input_da, domains, xrds_kw, dl_kw, grad=False, aug_kw=None, resize_factor=1, res=0.05, norm_stats=None, **kwargs):
        super().__init__()
        self.input_da = input_da
        if grad:
            #self.input_da = compute_grad(self.input_da)
            self.input_da = vort(self.input_da)
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.grad = grad
        self.aug_kw = aug_kw if aug_kw is not None else {}
        self.resize_factor = resize_factor
        self.res = res
        self._norm_stats = norm_stats

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._post_fn = None

    def norm_stats(self):
        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std()
            print("Norm stats", self._norm_stats)
        return self._norm_stats

    def train_mean_std(self, variable='tgt'):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains['train'])
        return train_data.sel(variable=variable).pipe(lambda da: (da.mean().values.item(), da.std().values.item()))

    def post_fn(self):
        m, s = self.norm_stats()
        normalize = lambda item: (item - m) / s
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(input=normalize(item.input)),
            lambda item: item._replace(tgt=normalize(item.tgt)),
        ])

    def post_fn_rand(self):
        m, s = self.norm_stats()
        normalize = lambda item: (item - m) / s
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(input=normalize(self.rand_obs(item.input,obs=True))),
            lambda item: item._replace(tgt=normalize(item.tgt)),
        ])

    def rand_obs(self, gt_item, obs=True):
        obs_mask_item = ~np.isnan(gt_item)
        _obs_item = gt_item
        dtime = self.xrds_kw['patch_dims']['time']
        dlat = self.xrds_kw['patch_dims']['lat']
        dlon = self.xrds_kw['patch_dims']['lon']
        for t_ in range(dtime):
            obs_mask_item_t_ = obs_mask_item[t_]
            if np.sum(obs_mask_item_t_)>.25*dlat*dlon:
                obs_obj = .5*np.sum(obs_mask_item_t_)
                while  np.sum(obs_mask_item_t_)>= obs_obj:
                    half_patch_height = np.random.randint(2,10)
                    half_patch_width = np.random.randint(2,10)
                    idx_lat = np.random.randint(0,dlat)
                    idx_lon = np.random.randint(0,dlon)
                    obs_mask_item_t_[np.max([0,idx_lat-half_patch_height]):np.min([dlat,idx_lat+half_patch_height+1]),np.max([0,idx_lon-half_patch_width]):np.min([dlon,idx_lon+half_patch_width+1])] = 0
                obs_mask_item[t_] = obs_mask_item_t_
        obs_mask_item = obs_mask_item == 1
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
        dtime = self.xrds_kw.patch_dims.time
        dlat = self.xrds_kw.patch_dims.lat
        dlon = self.xrds_kw.patch_dims.lon

        # define random size of additional wholes
        half_patch_height = np.random.randint(2,10,(dtime,npatch))
        half_patch_width = np.random.randint(2,10,(dtime,npatch))
        idx_lat = np.random.randint(0,dlat,(dtime,npatch))
        idx_lon = np.random.randint(0,dlon,(dtime,npatch))

        # define objective of missing data (50% of the initial obs)
        obs_obj = .5 * np.sum(obs_mask_item,axis=(1,2))
        
        # define 3d-numpy array index of new mask 
        posy_start = (idx_lat-half_patch_height).clip(min=0)
        posy_stop = (idx_lat+half_patch_height).clip(max=dlat)
        posx_start = (idx_lon-half_patch_width).clip(min=0)
        posx_stop = (idx_lon+half_patch_width).clip(max=dlon)
        id_lat = [ [ np.arange(k,l) for k,l in zip(posy_start[t],posy_stop[t])] for t in range(dtime)]
        id_lon = [ [ np.arange(k,l) for k,l in zip(posx_start[t],posx_stop[t])] for t in range(dtime)]
        idx = np.concatenate([ np.array(np.meshgrid([t],id_lat[t][p],id_lon[t][p])).T.reshape(-1,3) \
                               for t, p in np.array(np.meshgrid(np.arange(dtime),np.arange(npatch))).T.reshape(-1,2) ])
        # clip the number of new missing data according to the objectives
        idx_t = [ np.where(idx[:,0]==t)[0][0] for t in range(dtime) ]
        idx_t.append(len(idx)-1)
        stop_idx = [ idx_t[t] + np.argwhere( np.cumsum(obs_mask_item[idx[idx_t[t]:idx_t[t+1],0],
                                                                     idx[idx_t[t]:idx_t[t+1],1],
                                                                     idx[idx_t[t]:idx_t[t+1],2]]==1) >= obs_obj[t])[0] \
                     if 2*obs_obj[t]/(dlat*dlon)>.4 else idx_t[t] for t in range(dtime) ]
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
        train_data = self.input_da.sel(self.domains['train'])
        post_fn = self.post_fn()
        post_fn_rand = self.post_fn_rand()
        #post_fn_rand = self.post_fn()

        self.train_ds = XrDataset(
            train_data, **self.xrds_kw, postpro_fn=post_fn_rand,
            resize_factor = self.resize_factor,
            res = self.res
        )
        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrDataset(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn,
            resize_factor = self.resize_factor,
            res =self.res
        )
        self.test_ds = XrDataset(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn,
            resize_factor = self.resize_factor,
            res = self.res
        )

    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

class ConcatDataModule(BaseDataModule):
    def train_mean_std(self):
        sum, count = 0, 0
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {}))
        for domain in self.domains['train']:
            _sum, _count = train_data.sel(domain).sel(variable='tgt').pipe(lambda da: (da.sum(), da.pipe(np.isfinite).sum()))
            sum += _sum
            count += _count

        mean = sum / count
        sum = 0
        for domain in self.domains['train']:
            _sum = train_data.sel(domain).sel(variable='tgt').pipe(lambda da: da - mean).pipe(np.square).sum()
            sum += _sum
        std = (sum / count)**0.5
        return mean.values.item(), std.values.item()

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

        
def load_altimetry_data(path, obs_from_tgt=False):
    ds =  (
        xr.open_dataset(path)
        # .assign(ssh=lambda ds: ds.ssh.coarsen(lon=2, lat=2).mean().interp(lat=ds.lat, lon=ds.lon))
        .load()
        .assign(
            input=lambda ds: ds.nadir_obs,
            tgt=lambda ds: remove_nan(ds.ssh),
        )    
    )

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))
    
    return (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )
    
def remove_nan(da):
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")

    da.transpose("lon", "lat", "time")[:, :] = gauss_seidel(
        Grid3D(da)
    )[1]
    return da

