from contrib.CROSCIM.data import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import numpy as np

def compute_cell_edges(arr):
    """
    Convert center-based coordinates (1D or 2D) to edge-based coordinates for pcolormesh.
    Returns array with shape+1 in each axis.
    """
    if arr.ndim == 1:
        # 1D version: compute edges between points
        edges = (arr[:-1] + arr[1:]) / 2
        first = arr[0] - (edges[0] - arr[0])
        last = arr[-1] + (arr[-1] - edges[-1])
        return np.concatenate([[first], edges, [last]])

    elif arr.ndim == 2:
        ny, nx = arr.shape
        edges = np.zeros((ny + 1, nx + 1), dtype=arr.dtype)

        # Interpolate internal edges
        edges[1:-1, 1:-1] = 0.25 * (
            arr[:-1, :-1] + arr[1:, :-1] + arr[:-1, 1:] + arr[1:, 1:]
        )

        # Extrapolate borders
        edges[0, 1:-1] = edges[1, 1:-1] - (edges[2, 1:-1] - edges[1, 1:-1])
        edges[-1, 1:-1] = edges[-2, 1:-1] + (edges[-2, 1:-1] - edges[-3, 1:-1])
        edges[1:-1, 0] = edges[1:-1, 1] - (edges[1:-1, 2] - edges[1:-1, 1])
        edges[1:-1, -1] = edges[1:-1, -2] + (edges[1:-1, -2] - edges[1:-1, -3])

        # Corners
        edges[0, 0] = edges[1, 0] - (edges[2, 0] - edges[1, 0])
        edges[0, -1] = edges[1, -1] - (edges[2, -1] - edges[1, -1])
        edges[-1, 0] = edges[-2, 0] + (edges[-2, 0] - edges[-3, 0])
        edges[-1, -1] = edges[-2, -1] + (edges[-2, -1] - edges[-3, -1])

        return edges

    else:
        raise ValueError("Unsupported dimension for computing edges.")


def plot_multires(batch_dict, var="asip_sic",
                           resolution_colors=None, title="Multi-resolution SIC"):
    """
    Affiche des champs multi-résolution avec gestion des discontinuités de longitude (±180°).
    """

    if resolution_colors is None:
        resolution_colors = {}

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
    ax.coastlines()

    pcm = None

    for key in sorted(batch_dict.keys(), key=lambda k: int(k.split("_x")[-1])):
        try:
            factor = int(key.split("_x")[-1])
        except ValueError:
            print(f"Ignoring malformed key: {key}")
            continue

        item = batch_dict[key]
        color = resolution_colors.get(factor, f"C{factor % 10}")

        try:
            data = getattr(item, var)
            lon = getattr(item, "lon")
            lat = getattr(item, "lat")
        except AttributeError:
            print(f"Variable manquante dans {key}.")
            continue

        if data.ndim == 3:
            data = np.nanmean(data, axis=0)

        # Convertir en coordonnées géographiques
        lon = denormalize_minmax(lon, -180, 180)
        lat = denormalize_minmax(lat, 50, 90)
        lon_edges = compute_cell_edges(lon)
        lat_edges = compute_cell_edges(lat)

        # Détection des sauts de longitude (wraps à ±180°)
        delta_lon_h = np.abs(np.diff(lon_edges, axis=1))
        jump_mask_h = delta_lon_h[:, :-1] > 180

        delta_lon_v = np.abs(np.diff(lon_edges, axis=0))
        jump_mask_v = delta_lon_v[:-1, :] > 180

        # Masque final pour cellules contenant un saut
        mask_bad = np.zeros_like(data, dtype=bool)
        mask_bad[:, :-1] |= jump_mask_h
        mask_bad[:-1, :] |= jump_mask_v

        data_masked = np.ma.array(data, mask=mask_bad)

        pcm = ax.pcolormesh(
            lon_edges, lat_edges, data_masked,
            transform=ccrs.PlateCarree(),
            cmap='Blues', shading='flat', alpha=0.8, zorder=1
        )

        # Tracer les contours du patch
        bottom = np.column_stack((lon_edges[0, :], lat_edges[0, :]))
        right = np.column_stack((lon_edges[:, -1], lat_edges[:, -1]))
        top = np.column_stack((lon_edges[-1, ::-1], lat_edges[-1, ::-1]))
        left = np.column_stack((lon_edges[::-1, 0], lat_edges[::-1, 0]))
        contour = np.vstack([bottom, right[1:], top[1:], left[1:], bottom[0:1]])

        ax.plot(contour[:, 0], contour[:, 1],
                transform=ccrs.PlateCarree(), color=color,
                linewidth=2, label=f"Patch x{factor}", zorder=2)

    ax.set_title(title)
    if pcm is not None:
        plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.5, label=var)
    ax.legend()
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()

def pad_dataset_with_coords(ds, pad_yc=0, pad_xc=0):
    import numpy as np
    import xarray as xr

    # Calcul des pas de grille régulière
    dy = float((ds.yc[1] - ds.yc[0]).item())
    dx = float((ds.xc[1] - ds.xc[0]).item())

    pad_yc_before = pad_yc // 2
    pad_yc_after = pad_yc - pad_yc_before
    pad_xc_before = pad_xc // 2
    pad_xc_after = pad_xc - pad_xc_before

    # Padding principal sur toutes les variables
    ds_padded = ds.pad(
        yc=(pad_yc_before, pad_yc_after),
        xc=(pad_xc_before, pad_xc_after),
        constant_values=np.nan
    )

    # Nouvelle coord yc/xc régulières
    new_yc = (
        ds.yc[0].item() - dy * np.arange(pad_yc_before, 0, -1)
    ).tolist() + ds.yc.values.tolist() + (
        ds.yc[-1].item() + dy * np.arange(1, pad_yc_after + 1)
    ).tolist()

    new_xc = (
        ds.xc[0].item() - dx * np.arange(pad_xc_before, 0, -1)
    ).tolist() + ds.xc.values.tolist() + (
        ds.xc[-1].item() + dx * np.arange(1, pad_xc_after + 1)
    ).tolist()

    ds_padded = ds_padded.assign_coords(
        yc=("yc", np.array(new_yc, dtype=ds.yc.dtype)),
        xc=("xc", np.array(new_xc, dtype=ds.xc.dtype))
    )

    # Extension de lon et lat (2D) par réplication des bords
    def pad_2d_variable(var, pad_y_before, pad_y_after, pad_x_before, pad_x_after):
        v = ds[var].values
        top = np.repeat(v[0:1, :], pad_y_before, axis=0)
        bottom = np.repeat(v[-1:, :], pad_y_after, axis=0)
        v_padded = np.concatenate([top, v, bottom], axis=0)

        left = np.repeat(v_padded[:, 0:1], pad_x_before, axis=1)
        right = np.repeat(v_padded[:, -1:], pad_x_after, axis=1)
        return np.concatenate([left, v_padded, right], axis=1)

    if "lon" in ds:
        lon_pad = pad_2d_variable("lon", pad_yc_before, pad_yc_after, pad_xc_before, pad_xc_after)
        ds_padded["lon"] = (("yc", "xc"), lon_pad)
    if "lat" in ds:
        lat_pad = pad_2d_variable("lat", pad_yc_before, pad_yc_after, pad_xc_before, pad_xc_after)
        ds_padded["lat"] = (("yc", "xc"), lat_pad)

    return ds_padded

class XrDatasetMultiResTrain(XrDataset):

    def __init__(self, multires=[1], *args, **kwargs):
        super().__init__(subsel_patch=True, *args, **kwargs)
        self.multires = multires

        # Precompute enlarged patch sizes per resolution
        self.enlarged_dims = {}
        for factor in self.multires:
            self.enlarged_dims[factor] = {
                'yc': self.patch_dims['yc'] * (factor//self.resize),
                'xc': self.patch_dims['xc'] * (factor//self.resize)
            }

    def coarsen_patch(self, patch, target_shape):
        """
        Coarsen a patch by adaptive average pooling to target shape.
        Input: (T, Y, X)
        """
        patch = torch.as_tensor(patch).float().unsqueeze(0)  # Add batch dim
        coarsened = F.adaptive_avg_pool2d(patch, target_shape)
        return coarsened.squeeze(0).numpy()

    def extract_enlarged_patch_from_datasets(self, sl, factor):
        """
        Extract a larger area from the original datasets (asip, cimr, cristal, covariates),
        coarsen ASIP, then interpolate other datasets onto coarsened grid.
        """

        y_center = (sl["yc"].start + sl["yc"].stop) // 2
        x_center = (sl["xc"].start + sl["xc"].stop) // 2

        enlarged_yc = self.enlarged_dims[factor*self.resize]['yc']
        enlarged_xc = self.enlarged_dims[factor*self.resize]['xc']

        y_start = max(0, y_center - enlarged_yc // 2)
        y_end = min(y_start + enlarged_yc, self.da_dims["yc"]-1)
        x_start = max(0, x_center - enlarged_xc // 2)
        x_end = min(x_start + enlarged_xc, self.da_dims["xc"]-1)

        item_mask = fast_pool(self.mask.isel(xc=slice(x_start, x_end), yc=slice(y_start, y_end)),
                              factor,factor,mode="binary")

        if self.load_data:
            asip_ds = self.full_asip.isel(time=sl["time"], xc=slice(x_start, x_end), yc=slice(y_start, y_end))
            cimr_ds = self.full_cimr.isel(time=sl["time"])
            cristal_ds = self.full_cristal.isel(time=sl["time"])
            covariate_ds = self.full_covs.isel(time=sl["time"])
        else:
            time_indices = np.arange(sl["time"].start, sl["time"].stop)
            slices = {"xc": slice(self.xc[x_start],self.xc[x_end]),
                      "yc": slice(self.yc[y_start],self.yc[y_end])
                     }
            type_coords = "coords"
            asip_ds = concatenate(self.asip_paths[time_indices], var_list=VAR_GROUPS["asip"],
                                  slices=slices, type_coords=type_coords,
                                  resize=factor*self.resize, domain_limits=self.domain_limits)
            cimr_ds = concatenate(self.cimr_paths[time_indices], var_list=VAR_GROUPS["cimr"], slices=None)
            cristal_ds = concatenate(self.cristal_paths[time_indices], var_list=VAR_GROUPS["cristal"], slices=None)
            covariate_ds = concatenate(self.covariates_paths[time_indices], var_list=self.covariates, slices=None)

        # padding if necessary
        expected_shape = (self.patch_dims['time'], self.patch_dims['yc'], self.patch_dims['xc'])
        actual_shape = asip_ds["sic"].shape
        if actual_shape != expected_shape:
            pad_t = expected_shape[0] - actual_shape[0]
            pad_y = expected_shape[1] - actual_shape[1]
            pad_x = expected_shape[2] - actual_shape[2]
            pad = {dim: (0, pad_) for dim, pad_ in zip(["time", "yc", "xc"], [pad_t, pad_y, pad_x])}
            # add mask
            asip_ds = asip_ds.update({"mask":(("yc","xc"),item_mask)})
            # pad
            asip_ds = pad_dataset_with_coords(asip_ds, pad_yc=pad_y, pad_xc=pad_x)
            asip_ds['mask'] = asip_ds['mask'].fillna(1)
            item_mask = asip_ds.mask.data

        lon_target = asip_ds.lon.values
        lat_target = asip_ds.lat.values

        coarsened_asip = {}
        for var in VAR_GROUPS["asip"]:
            coarsened_asip[f"asip_{var}"] = asip_ds[var].values

        #target_shape = (self.patch_dims['yc'], self.patch_dims['xc'])
        # Get target grid from coarsened ASIP
        #lon_target = F.adaptive_avg_pool2d(torch.from_numpy(asip_ds.lon.values).unsqueeze(0).float(), target_shape).squeeze(0).numpy()
        #lat_target = F.adaptive_avg_pool2d(torch.from_numpy(asip_ds.lat.values).unsqueeze(0).float(), target_shape).squeeze(0).numpy()

        # Interpolate CIMR, CRISTAL, COVARIATES
        if self.itrp_from_regular:
            target_grid=(asip_ds.xc.values, asip_ds.yc.values)
            coarsened_cimr = self.interpolate_dataset(target_grid, cimr_ds, VAR_GROUPS["cimr"],prefix="cimr")
            coarsened_cristal = self.interpolate_dataset(target_grid, cristal_ds, VAR_GROUPS["cristal"],prefix="cristal")
            coarsened_covs = self.interpolate_dataset(target_grid, covariate_ds, self.covariates)
        else:
            swath_def_target = pyresample.geometry.SwathDefinition(lons=lon_target, lats=lat_target)
            coarsened_cimr= self.interpolate_dataset(swath_def_target, cimr_ds, VAR_GROUPS["cimr"],prefix="cimr")
            coarsened_cristal = self.interpolate_dataset(swath_def_target, cristal_ds, VAR_GROUPS["cristal"],prefix="cristal")
            coarsened_covs = self.interpolate_dataset(swath_def_target, covariate_ds, self.covariates)

        # Assemble sample
        sample = {**coarsened_asip, **coarsened_cimr, **coarsened_cristal, **coarsened_covs}
        sample["land_mask"] = np.expand_dims(item_mask, axis=0)
        sample["lat"] = np.expand_dims(lat_target, axis=0)
        sample["lon"] = np.expand_dims(lon_target, axis=0)

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

    def __getitem__(self, idx):
        hr_sample = super().__getitem__(idx)
        if self.subsel_patch:
            idx = self.idx_patches_in_ocean[idx]
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx_dim,
                       self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
            for dim, idx_dim in zip(self.ds_size.keys(), np.unravel_index(idx, tuple(self.ds_size.values())))
        }

        out = {}
        out[f"patch_x{self.resize}"] = hr_sample
        for factor in self.multires[:-1]:
            enlarged_patch = self.extract_enlarged_patch_from_datasets(sl, factor//(self.resize))
            out[f"patch_x{factor}"] = enlarged_patch                  

        #plot_multires(out, var="asip_sic",
        #              resolution_colors={1: "blue", 10: "orange", 100: "red"},
        #              title="Sea Ice Concentration (Multi-resolution)"
        #              )
        return out

class XrDatasetMultiResTest:
    """
    Dataset pour le mode test : contient N XrDataset avec coarsening contrôlé par multires
    Chaque sortie est un TrainingItem avec suffixe de résolution.
    """
    """
    Dataset pour le mode test : contient N XrDataset avec coarsening contrôlé par multires
    """
    def __init__(self, multires=[1], *args, **kwargs):
        self.datasets = {}
        for res in multires:
            kwargs["resize"] = res
            self.datasets[res] = XrDataset(subsel_patch=(res==multires[-1]),
                                           load_data=True,
                                           *args, **kwargs)

    def get_dataloader_dict(self, batch_size=1, **loader_kwargs):
        """
        Retourne un dictionnaire de DataLoader pour chaque résolution
        """
        from torch.utils.data import DataLoader
        return {res: DataLoader(ds, batch_size=batch_size, **loader_kwargs)
                for res, ds in self.datasets.items()}

class BaseDataModuleMultiRes(BaseDataModule):
    
    def __init__(self, multires=[1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multires = multires
        self.resize = self.multires[-1]

    def save_batch_as_NetCDF_multires(self, batch_dict, ibatch, patch_dims_dict, save_dir="/dmidata/users/maxb/PREPROC/"):
        """
        Save a multiresolution batch dictionary as separate NetCDF files.
        batch_dict: dict of {f"patch_x{res}": TrainingItem}
        patch_dims_dict: dict of {res: {"time": ..., "yc": ..., "xc": ...}}
        """
        os.makedirs(save_dir, exist_ok=True)
    
        for key, batch in batch_dict.items():
            # Extrait le facteur de résolution (ex: x10 -> 10)
            try:
                factor = int(key.split("x")[-1])
            except:
                print(f"Warning: can't parse resolution factor in {key}")
                continue
    
            patch_dims = patch_dims_dict[factor]
    
            data_vars = {}
    
            # Variables satellites
            for group in VAR_GROUPS:
                for var in VAR_GROUPS[group]:
                    var_name = f"{group}_{var}"
                    if hasattr(batch, var_name):
                        tensor = getattr(batch, var_name)
                        if torch.is_tensor(tensor) and tensor.ndim == 4:
                            data_vars[var_name] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())
    
            # Covariates
            for cov in COVARIATES:
                if hasattr(batch, cov):
                    tensor = getattr(batch, cov)
                    if torch.is_tensor(tensor) and tensor.ndim == 4:
                        data_vars[cov] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())
    
            # Coordonnées et masque
            data_vars.update({
                'times': (('sample', 'time'), torch.squeeze(batch.time,dim=1).detach().cpu().numpy().astype("datetime64[s]")),
                'ycs': (('sample', 'yc'), torch.squeeze(batch.yc,dim=1).detach().cpu()),
                'xcs': (('sample', 'xc'), torch.squeeze(batch.xc,dim=1).detach().cpu()),
                'lat': (('sample', 'yc', 'xc'), torch.squeeze(batch.lat,dim=1).detach().cpu()),
                'lon': (('sample', 'yc', 'xc'), torch.squeeze(batch.lon,dim=1).detach().cpu()),
                'land_mask': (('sample', 'yc', 'xc'), torch.squeeze(batch.land_mask,dim=1).detach().cpu()),
            })
    
            # target variables
            for group, variables in VAR_GROUPS.items():
                for var in variables:
                    new_key = f"tgt_{var}"
                    if hasattr(batch, new_key):
                        tensor = getattr(batch, new_key)
                        if torch.is_tensor(tensor) and tensor.ndim == 4:
                            data_vars[new_key] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())

            # Coordonnées
            coords = {
                'sample': np.arange(list(data_vars.values())[0][1].shape[0]),
                'time': np.arange(patch_dims['time']),
                'yc': np.arange(patch_dims['yc']),
                'xc': np.arange(patch_dims['xc'])
            }

            # Construction et sauvegarde
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            save_path = os.path.join(save_dir, f"preproc_batch_{ibatch}_x{factor}.nc")
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
            
            if split=="test":
                XrDatasetMultiRes = XrDatasetMultiResTest
            else:
                XrDatasetMultiRes = XrDatasetMultiResTrain
            return XrDatasetMultiRes(
                multires=self.multires,
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
                stride_test=(split != 'train'),
                resize=self.resize,
                subsel_patch_path=f"{self.subsel_path}/patch_in_ocean_{split}_{self.domain_name}_patch_{self.xrds_kw['patch_dims']['yc']}_{self.xrds_kw['strides']['yc']}_resize_x{self.resize}.txt"
            )

        self.train_ds = create_dataset('train')
        #self.val_ds = create_dataset('val')
        #self.test_ds = create_dataset('test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        sampler = CustomBatchSampler(self.val_ds, batch_size=self.dl_kw["batch_size"])
        return torch.utils.data.DataLoader(self.val_ds, batch_sampler=sampler, num_workers=self.dl_kw["num_workers"])

    def test_dataloader(self):
        return { f"patch_x{res}": torch.utils.data.DataLoader(ds, shuffle=False, **self.dl_kw)
                 for res, ds in self.test_ds.datasets.items() }

