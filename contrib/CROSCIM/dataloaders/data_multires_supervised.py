from contrib.CROSCIM.dataloaders.data import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import numpy as np

# Models will be handled separately via models_vars parameter
DEFAULT_VAR_GROUPS = {
    "cimr": ["SIC", "SIT"],
    "cristal": ["SIT", "SSH"],
    "asip": ["sic"]
}


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

class XrDatasetMultiResTrainSupervised(XrDatasetSupervised):

    def __init__(self, multires=[1], satellite_vars=None, models_vars=None, models_paths=None,
                 target_vars=None, var_mapping=None, *args, **kwargs):
        
        # Store satellite_vars BEFORE calling super().__init__
        self.satellite_vars = satellite_vars or DEFAULT_VAR_GROUPS
        
        # Store models config
        self.models_vars = models_vars if models_vars is not None else []
        self.models_paths = models_paths if models_paths is not None else np.array([])

        # Store target configuration
        self.target_vars = target_vars if target_vars is not None else []
        self.var_mapping = var_mapping if var_mapping is not None else {}
        
              
        # Determine which sources are active (including models)
        self.active_sources = [src for src, vars in self.satellite_vars.items() if vars]
        if self.models_vars and len(self.models_vars) > 0: 
            self.active_sources.append('models')
        
        # ✅ PASS models_vars and models_paths to parent
        super().__init__(
            subsel_patch=True, 
            satellite_vars=satellite_vars,
            models_vars=models_vars, 
            models_paths=models_paths, 
            target_vars=target_vars, 
            var_mapping=var_mapping,  
            *args, 
            **kwargs
        )
        
        self.multires = multires

        # Precompute enlarged patch sizes per resolution
        self.enlarged_dims = {}
        for factor in self.multires:
            self.enlarged_dims[factor] = {
                'yc': self.patch_dims['yc'] * (factor // self.resize),
                'xc': self.patch_dims['xc'] * (factor // self.resize)
            }
        
        print(f"XrDatasetMultiResTrainSupervised initialized:")
        print(f"  Satellite vars: {self.satellite_vars}")
        print(f"  Models vars: {self.models_vars}")
        print(f"  Active sources: {self.active_sources}")
        print(f"  Multires: {self.multires}")

    def coarsen_patch(self, patch, target_shape):
        """
        Coarsen a patch by adaptive average pooling to target shape.
        Input: (T, Y, X)
        """
        patch = torch.as_tensor(patch).float().unsqueeze(0)  # Add batch dim
        # NaN-aware average pooling: pool(value*valid)/pool(valid) so NaN cells are
        # ignored (output valid as soon as >=1 finite cell in the window). Vectorised.
        valid = torch.isfinite(patch).float()
        num = F.adaptive_avg_pool2d(torch.nan_to_num(patch, nan=0.0), target_shape)
        den = F.adaptive_avg_pool2d(valid, target_shape)
        coarsened = num / den
        coarsened[den == 0] = float('nan')
        return coarsened.squeeze(0).numpy()

    def extract_enlarged_patch_from_datasets(self, sl, factor):
        """
        Extract a larger area from the original datasets,
        coarsen ASIP, then interpolate other datasets onto coarsened grid.
        Only loads data for active sources.
        """
        y_center = (sl["yc"].start + sl["yc"].stop) // 2
        x_center = (sl["xc"].start + sl["xc"].stop) // 2

        enlarged_yc = self.enlarged_dims[factor * self.resize]['yc']
        enlarged_xc = self.enlarged_dims[factor * self.resize]['xc']

        y_start = max(0, y_center - enlarged_yc // 2)
        y_end = min(y_start + enlarged_yc, self.da_dims["yc"] - 1)
        x_start = max(0, x_center - enlarged_xc // 2)
        x_end = min(x_start + enlarged_xc, self.da_dims["xc"] - 1)

        item_mask = fast_pool(self.mask.isel(xc=slice(x_start, x_end), yc=slice(y_start, y_end)),
                              factor, factor, mode="binary")

        # Load datasets based on active sources
        datasets = {}
        
        if self.load_data:
            # Load only active sources
            if 'asip' in self.active_sources:
                datasets['asip'] = self.full_asip.isel(
                    time=sl["time"], 
                    xc=slice(x_start, x_end), 
                    yc=slice(y_start, y_end)
                )
            if 'cimr' in self.active_sources:
                datasets['cimr'] = self.full_cimr.isel(time=sl["time"])
            if 'cristal' in self.active_sources:
                datasets['cristal'] = self.full_cristal.isel(time=sl["time"])
            
            #  Load models data
            if 'models' in self.active_sources:
                datasets['models'] = self.full_models.isel(time=sl["time"])
            
            # Load covariates if configured
            if hasattr(self, 'covariates') and self.covariates:
                datasets['covariates'] = self.full_covs.isel(time=sl["time"])
        else:
            time_indices = np.arange(sl["time"].start, sl["time"].stop)
            slices = {
                "xc": slice(self.xc[x_start], self.xc[x_end]),
                "yc": slice(self.yc[y_start], self.yc[y_end])
            }
            type_coords = "coords"
            
            # Load only active sources
            if 'asip' in self.active_sources:
                datasets['asip'] = concatenate(
                    self.asip_paths[time_indices], 
                    var_list=self.satellite_vars['asip'],
                    slices=slices, 
                    type_coords=type_coords,
                    resize=factor * self.resize, 
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
            
            #  Load models data
            if 'models' in self.active_sources:
                datasets['models'] = concatenate(
                    self.models_paths[time_indices], 
                    var_list=self.models_vars, 
                    slices=None
                )
            
            # Load covariates if configured
            if hasattr(self, 'covariates') and self.covariates:
                datasets['covariates'] = concatenate(
                    self.covariates_paths[time_indices], 
                    var_list=self.covariates, 
                    slices=None
                )

        # Get ASIP dataset as reference (must be present)
        if 'asip' not in datasets:
            raise ValueError("ASIP dataset is required as reference grid but not loaded")
        
        asip_ds = datasets['asip']

        # Padding if necessary
        expected_shape = (self.patch_dims['time'], self.patch_dims['yc'], self.patch_dims['xc'])
        # Use first available ASIP variable
        first_asip_var = self.satellite_vars['asip'][0]
        actual_shape = asip_ds[first_asip_var].shape
        
        if actual_shape != expected_shape:
            pad_t = expected_shape[0] - actual_shape[0]
            pad_y = expected_shape[1] - actual_shape[1]
            pad_x = expected_shape[2] - actual_shape[2]
            pad = {dim: (0, pad_) for dim, pad_ in zip(["time", "yc", "xc"], [pad_t, pad_y, pad_x])}
            # add mask
            asip_ds = asip_ds.update({"mask": (("yc", "xc"), item_mask)})
            # pad
            asip_ds = pad_dataset_with_coords(asip_ds, pad_yc=pad_y, pad_xc=pad_x)
            asip_ds['mask'] = asip_ds['mask'].fillna(1)
            item_mask = asip_ds.mask.data

        lon_target = asip_ds.lon.values
        lat_target = asip_ds.lat.values

        # Collect data from ASIP
        sample = {}
        for var in self.satellite_vars['asip']:
            sample[f"asip_{var}"] = asip_ds[var].values

        # Interpolate other satellite sources (only if active)
        if self.itrp_from_regular:
            target_grid = (asip_ds.xc.values, asip_ds.yc.values)
            if 'cimr' in datasets:
                interpolated = self.interpolate_dataset(
                    target_grid, 
                    datasets['cimr'], 
                    self.satellite_vars['cimr'],
                    prefix="cimr"
                )
                sample.update(interpolated)
            if 'cristal' in datasets:
                interpolated = self.interpolate_dataset(
                    target_grid, 
                    datasets['cristal'], 
                    self.satellite_vars['cristal'],
                    prefix="cristal"
                )
                sample.update(interpolated)
            
            # Interpolate models data
            if 'models' in datasets:
                interpolated = self.interpolate_dataset(
                    target_grid, 
                    datasets['models'], 
                    self.models_vars,
                    prefix="models"
                )
                sample.update(interpolated)
            
            if 'covariates' in datasets:
                interpolated = self.interpolate_dataset(
                    target_grid, 
                    datasets['covariates'], 
                    self.covariates
                )
                sample.update(interpolated)
        else:
            swath_def_target = pyresample.geometry.SwathDefinition(lons=lon_target, lats=lat_target)
            
            if 'cimr' in datasets:
                interpolated = self.interpolate_dataset(
                    swath_def_target, 
                    datasets['cimr'], 
                    self.satellite_vars['cimr'],
                    prefix="cimr"
                )
                sample.update(interpolated)
            
            if 'cristal' in datasets:
                interpolated = self.interpolate_dataset(
                    swath_def_target, 
                    datasets['cristal'], 
                    self.satellite_vars['cristal'],
                    prefix="cristal"
                )
                sample.update(interpolated)
            
            #  Interpolate models data
            if 'models' in datasets:
                interpolated = self.interpolate_dataset(
                    swath_def_target, 
                    datasets['models'], 
                    self.models_vars,
                    prefix="models"
                )
                sample.update(interpolated)
            
            if 'covariates' in datasets:
                interpolated = self.interpolate_dataset(
                    swath_def_target, 
                    datasets['covariates'], 
                    self.covariates
                )
                sample.update(interpolated)

        # Add metadata
        sample["land_mask"] = np.expand_dims(item_mask, axis=0)
        sample["lat"] = np.expand_dims(lat_target, axis=0)
        sample["lon"] = np.expand_dims(lon_target, axis=0)

        # Determine which resolution key to use based on factor
        res_key = f"patch_x{factor * self.resize}"
        
        # Get resolution-specific mapping
        if isinstance(self.var_mapping, dict) and res_key in self.var_mapping:
            mapping = self.var_mapping[res_key]
        else:
            # Fallback to base var_mapping (for backward compatibility)
            mapping = self.var_mapping
        
        # Initialize targets from their sources
        for target_var, source_var in mapping.items():
            # If target starts with models_XXX, create tgt_XXX from models_XXX
            if target_var.startswith('models_') and '_' in target_var:
                suffix = target_var.split('_', 1)[1]  # Extract XXX from models_XXX
                tgt_var = f"tgt_{suffix}"
                if target_var in sample:
                    sample[tgt_var] = sample[target_var].copy()
                    print(f"  {res_key}: Created {tgt_var} <- {target_var}")
                else:
                    print(f"  WARNING ({res_key}): Could not find {target_var} to create {tgt_var}")
            # If target starts with tgt_XXX, use the source_var directly
            elif target_var.startswith('tgt_') and source_var in sample:
                sample[target_var] = sample[source_var].copy()
                print(f"  {res_key}: Mapped {target_var} <- {source_var}")
            elif target_var.startswith('tgt_'):
                print(f"  WARNING ({res_key}): Could not map {target_var} from {source_var}. Available keys: {list(sample.keys())}")

        # Keep track of the coordinates
        sample["time"] = np.expand_dims(
            np.array([np.datetime64(t, "s").astype('float64') for t in asip_ds.time.values]),
            axis=0
        )
        sample["xc"] = np.expand_dims(asip_ds.xc.values, axis=0)
        sample["yc"] = np.expand_dims(asip_ds.yc.values, axis=0)

        # Apply postprocessing WITH resolution key
        if self.postpro_fn is not None:
            # Check if postpro_fn expects resolution_key argument
            import inspect
            sig = inspect.signature(self.postpro_fn)
            if 'resolution_key' in sig.parameters:
                sample = self.postpro_fn(sample, resolution_key=res_key)
            else:
                # Fallback for old-style postpro_fn
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
            enlarged_patch = self.extract_enlarged_patch_from_datasets(sl, factor // (self.resize))
            out[f"patch_x{factor}"] = enlarged_patch

        return out

class XrDatasetMultiResTestSupervised:
    """
    Dataset pour le mode test : contient N XrDataset avec coarsening contrôlé par multires
    Chaque sortie est un TrainingItem avec suffixe de résolution.
    """
    def __init__(self, multires=[1], patch_dims_dict=None, strides_test_dict=None,
                 satellite_vars=None, covariates=None, models_vars=None, models_paths=None,
                 target_vars=None, var_mapping=None, *args, **kwargs):
        self.datasets = {}
        self.satellite_vars = satellite_vars or DEFAULT_VAR_GROUPS
        self.covariates = covariates if covariates is not None else []
        self.models_vars = models_vars if models_vars is not None else []
        self.models_paths = models_paths if models_paths is not None else np.array([])
        self.target_vars = target_vars if target_vars is not None else []
        self.var_mapping = var_mapping if var_mapping is not None else {}    

        # Handle patch_dims_dict (new) vs patch_dims (legacy)
        if patch_dims_dict is not None:
            self.patch_dims_dict = patch_dims_dict
            # Validate: all resolutions must have patch_dims
            for res in multires:
                if res not in patch_dims_dict:
                    raise ValueError(f"patch_dims_dict missing entry for resolution {res}")
        else:
            # Fallback: use same patch_dims for all resolutions
            default_patch_dims = kwargs.get('patch_dims', {'time': 15, 'yc': 256, 'xc': 256})
            self.patch_dims_dict = {res: default_patch_dims.copy() for res in multires}
            print(f" No patch_dims_dict provided, using {default_patch_dims} for all resolutions")

        # Handle strides_test_dict
        if strides_test_dict is not None:
            self.strides_test_dict = strides_test_dict
            for res in multires:
                if res not in strides_test_dict:
                    raise ValueError(f"strides_test_dict missing entry for resolution {res}")
        else:
            # Fallback: use strides_test from kwargs if present, else compute from patch_dims
            default_strides_test = kwargs.get('strides_test', None)
            if default_strides_test:
                self.strides_test_dict = {res: default_strides_test.copy() for res in multires}
            else:
                # Auto-compute: non-overlapping strides = patch_dims
                self.strides_test_dict = {
                    res: self.patch_dims_dict[res].copy() 
                    for res in multires
                }
            print(f"No strides_test_dict provided, using non-overlapping strides")

        print(f"\n{'='*60}")
        print(f"XrDatasetMultiResTest configuration:")
        print(f"  Multires: {multires}")
        print(f"{'='*60}")

        for res in multires:
            # Get resolution-specific patch_dims
            patch_dims = self.patch_dims_dict[res]
            strides_test = self.strides_test_dict[res]
 
            # Create resolution-specific kwargs
            res_kwargs = kwargs.copy()
            res_kwargs['patch_dims'] = patch_dims
            res_kwargs['strides_test'] = strides_test
            res_kwargs['resize'] = res
            
            #  Remove models_paths and models_vars before passing to XrDataset
            res_kwargs_clean = {k: v for k, v in res_kwargs.items() if k not in ['models_paths', 'models_vars']}
            
            print(f"  Resolution x{res}:")
            print(f"    patch_dims: time={patch_dims['time']}, yc={patch_dims['yc']}, xc={patch_dims['xc']}")


            self.datasets[res] = XrDatasetSupervised(
                satellite_vars=satellite_vars,
                covariates=covariates,
                models_vars=models_vars,  
                models_paths=models_paths, 
                target_vars=target_vars,
                var_mapping=var_mapping,
                subsel_patch=(res == multires[-1]) and len(multires) > 1,
                load_data=True,
                *args, 
                **res_kwargs 
            )

    def get_dataloader_dict(self, batch_size=1, **loader_kwargs):
        """
        Retourne un dictionnaire de DataLoader pour chaque résolution
        """
        from torch.utils.data import DataLoader
        return {
            res: DataLoader(ds, batch_size=batch_size, **loader_kwargs)
            for res, ds in self.d
        }

class BaseDataModuleMultiRes(BaseDataModule):
    def __init__(self, 
                 asip_paths, 
                 cimr_paths, 
                 cristal_paths,
                 covariates_paths,
                 target_vars, 
                 covariates=None, 
                 models_paths=None,
                 models_vars=None,
                 satellite_vars=None,
                 var_mapping=None,
                 multires=[50, 10, 2],
                 norm_stats_models=None,
                 rand_obs=False,
                 **kwargs):
        """
        Multi-resolution data module extending BaseDataModule.
        
        Args:
            models_paths: Paths to numerical model files (optional)
            models_vars: List of model variable names (optional)
            multires: List of resolution factors [coarsest, ..., finest]
            All other args passed to BaseDataModule
        """

        # Store models configuration
        self.models_paths = models_paths if models_paths is not None else []
        self.models_vars = models_vars if models_vars is not None else []        
        self._norm_stats_models = norm_stats_models

        # Extract patch_dims_dict / dl_kw from xrds_kw if present (only for test)
        xrds_kw = kwargs.get('xrds_kw', {})
        self.patch_dims_dict = xrds_kw.pop('patch_dims_dict', None)
        self.strides_test_dict = xrds_kw.pop('strides_test_dict', None)
        # dl_kw may be nested inside xrds_kw (test configs) — hoist it up so
        # BaseDataModule receives it as a top-level arg, not forwarded to XrDataset
        if 'dl_kw' in xrds_kw and 'dl_kw' not in kwargs:
            kwargs['dl_kw'] = xrds_kw.pop('dl_kw')
        else:
            xrds_kw.pop('dl_kw', None)  # remove duplicate if already present

        # Update kwargs with modified xrds_kw
        if 'xrds_kw' in kwargs:
            kwargs['xrds_kw'] = xrds_kw

        # Call parent with explicit required arguments (including rand_obs)
        super().__init__(
            asip_paths=asip_paths,
            cimr_paths=cimr_paths,
            cristal_paths=cristal_paths,
            covariates_paths=covariates_paths,
            covariates=covariates,
            target_vars=target_vars,
            satellite_vars=satellite_vars,
            var_mapping=var_mapping,
            models_vars=self.models_vars,
            rand_obs=rand_obs,
            **kwargs
        )
        
        # Store multi-resolution configuration
        self.multires = multires
        self.resize = self.multires[-1]

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
        print(f"TrainingItem fields: {TrainingItem._fields}")

        print(f"\n{'='*60}")
        print(f"BaseDataModuleMultiRes configuration:")
        print(f"{'='*60}")
        print(f"  Multi-resolution factors: {self.multires}")
        print(f"  Satellite vars: {self.satellite_vars}")
        print(f"  Models vars: {self.models_vars}")  # ✅ NEW
        print(f"  Target vars: {self.target_vars}")
        if self.patch_dims_dict:
            print(f"  Test patch_dims per resolution:")
            for res, dims in self.patch_dims_dict.items():
                print(f"    x{res}: time={dims['time']}, yc={dims['yc']}, xc={dims['xc']}")
        else:
            print(f"  Using same patch_dims for all resolutions (train behavior)")
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
        
    def save_batch_as_NetCDF_multires(self, batch_dict, ibatch, patch_dims_dict, 
                                      save_dir="/dmidata/users/maxb/PREPROC/"):
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

            # Helper function to check if a field is non-empty
            def is_valid_field(tensor):
                """Check if tensor is valid (not empty array from None replacement)."""
                if torch.is_tensor(tensor):
                    return tensor.numel() > 0 and tensor.ndim == 4
                elif isinstance(tensor, np.ndarray):
                    return tensor.size > 0
                return False

            # Variables satellites (dynamically based on config)
            for source in self.active_sources:
                for var in self.satellite_vars.get(source, []):
                    var_name = f"{source}_{var}"
                    if hasattr(batch, var_name):
                        tensor = getattr(batch, var_name)
                        if is_valid_field(tensor):
                            data_vars[var_name] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())

            # Model variables
            for var in self.models_vars:
                var_name = f"models_{var}"
                if hasattr(batch, var_name):
                    tensor = getattr(batch, var_name)
                    if is_valid_field(tensor):
                        data_vars[var_name] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())

            # Covariates
            for cov in self.covariates:
                if hasattr(batch, cov):
                    tensor = getattr(batch, cov)
                    if is_valid_field(tensor):
                        data_vars[cov] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())

            # Coordonnées et masque (always present, not filtered)
            data_vars.update({
                'times': (('sample', 'time'), torch.squeeze(batch.time, dim=1).detach().cpu().numpy().astype("datetime64[s]")),
                'ycs': (('sample', 'yc'), torch.squeeze(batch.yc, dim=1).detach().cpu()),
                'xcs': (('sample', 'xc'), torch.squeeze(batch.xc, dim=1).detach().cpu()),
                'lat': (('sample', 'yc', 'xc'), torch.squeeze(batch.lat, dim=1).detach().cpu()),
                'lon': (('sample', 'yc', 'xc'), torch.squeeze(batch.lon, dim=1).detach().cpu()),
                'land_mask': (('sample', 'yc', 'xc'), torch.squeeze(batch.land_mask, dim=1).detach().cpu()),
            })

            # Target variables (dynamically based on config and resolution)
            # Get target_vars for this specific resolution
            if isinstance(self.target_vars, dict):
                res_key = f"patch_x{factor}"
                target_vars_for_res = self.target_vars.get(res_key, [])
            else:
                target_vars_for_res = self.target_vars
            
            # Build list of all target variables to save (including tgt_XXX variants)
            all_target_vars = []
            for target_var in target_vars_for_res:
                all_target_vars.append(target_var)
                # If target_var is models_XXX or other prefix, also save tgt_XXX
                if '_' in target_var:
                    prefix, suffix = target_var.split('_', 1)
                    tgt_var = f"tgt_{suffix}"
                    if tgt_var not in all_target_vars:
                        all_target_vars.append(tgt_var)
            
            for target_var in all_target_vars:
                if hasattr(batch, target_var):
                    tensor = getattr(batch, target_var)
                    if is_valid_field(tensor):
                        data_vars[target_var] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())

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
            print(f"Saved: {save_path}")

    def post_fn(self, rand_obs=False):
        """
        Post-processing with resolution-dependent target initialization.
        
        Args:
            rand_obs: Either a boolean (apply to all satellite vars if True) 
                     or a list of variable names to apply random masking to
        """
        
        norm_sats = self._norm_stats
        norm_covs = self._norm_stats_covs
        norm_models = self._norm_stats_models
        
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
    
        def apply_norm(item, resolution_key):
            """
            Normalize batch and initialize targets based on resolution.
            
            Args:
                item: Raw batch dictionary
                resolution_key: One of 'patch_x2', 'patch_x10', 'patch_x50'
            """
            
            # Get resolution-specific var_mapping
            if isinstance(self.var_mapping, dict) and resolution_key in self.var_mapping:
                mapping = self.var_mapping[resolution_key]
            else:
                # Fallback to single mapping (backward compatibility)
                mapping = self.var_mapping
            
            # Get resolution-specific target_vars
            if isinstance(self.target_vars, dict) and resolution_key in self.target_vars:
                target_vars_for_res = self.target_vars[resolution_key]
            else:
                # Fallback: use all target vars if not resolution-specific
                target_vars_for_res = self._get_all_target_vars()
            
            # Initialize target variables from their RESOLUTION-SPECIFIC sources
            #for target_var, source_var in mapping.items():
            #    if source_var in item:
            #        item[target_var] = item[source_var].copy()
            #        print(f"  {resolution_key}: Initialized {target_var} from {source_var}")
            
            # Create TrainingItem - handle missing fields with None defaults
            item_with_defaults = {field: item.get(field, None) for field in TrainingItem._fields}
            data = TrainingItem(**item_with_defaults)
            
            # ==================================================================
            # NORMALIZATION LOGIC
            # ==================================================================
            
            # Build a mapping to determine which stats to use for each variable
            # This avoids double normalization
            stats_to_use = {}
            
            # Check var_mapping to determine if satellite variables should use model stats
            for target_var, source_var in mapping.items():
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
            for target_var, source_var in mapping.items():
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
            # These are typically created in extract_enlarged_patch_from_datasets
            for target_var in target_vars_for_res:
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
            
            # Filter out None fields by creating a new TrainingItem with only present fields
            # Keep only fields that are not None (indicating they exist at this resolution)
            data_dict = data._asdict()
            filtered_dict = {}
            for field in data._fields:
                if data_dict[field] is not None:
                    filtered_dict[field] = data_dict[field]
                else:
                    # Replace None with empty array so TrainingItem stays same class (for pickling)
                    # These will be filtered during NetCDF save and model processing
                    filtered_dict[field] = np.array([], dtype=np.float32)
            
            return type(data)(**filtered_dict)
        
        # Return a function that applies normalization with resolution context
        def post_fn_with_resolution(batch_dict, resolution_key=None):
            """Apply post-processing to multi-resolution batch.
            
            Args:
                batch_dict: Either a multi-res dict with patch_xXX keys, or a single sample dict
                resolution_key: Optional resolution key to use for single sample (e.g., 'patch_x50')
            """
            
            # Generate resolution keys dynamically from self.multires
            res_keys = [f'patch_x{factor}' for factor in self.multires]
            
            # Check if this is a multi-resolution batch by looking for any of the resolution keys
            is_multires_batch = isinstance(batch_dict, dict) and any(key in batch_dict for key in res_keys)
            
            if is_multires_batch:
                # Multi-resolution batch
                processed = {}
                for res_key in res_keys:
                    if res_key in batch_dict:
                        processed[res_key] = apply_norm(batch_dict[res_key], res_key)
                return processed
            else:
                # Single resolution batch (fallback)
                # Use provided resolution_key or fallback to first resolution
                if resolution_key is None:
                    resolution_key = res_keys[0] if res_keys else 'patch_x1'
                return apply_norm(batch_dict, resolution_key)
        
        return post_fn_with_resolution

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
            
            for source in self.active_sources:
                source_paths_attr = f"{source}_paths"
                if hasattr(self, source_paths_attr):
                    fmt = "%Y%m%d" if source == "asip" else "%Y-%m-%d"
                    paths, times = select_paths(
                        getattr(self, source_paths_attr), 
                        self.domains[split]['time'],
                        fmt=fmt
                    )
                    paths_dict[f"{source}_paths"] = paths
            
            # Ensure all required paths are present (even if empty)
            for source in ['asip', 'cimr', 'cristal']:
                if f"{source}_paths" not in paths_dict:
                    paths_dict[f"{source}_paths"] = np.array([])
        
            # Add models paths if configured
            if self.models_vars is not None and len(self.models_vars) > 0:
                models_paths, _ = select_paths(
                    self.models_paths,
                    self.domains[split]['time'],
                    fmt="%Y-%m-%d"  # Adjust format as needed
                )
                paths_dict['models_paths'] = models_paths
            
            # Get covariate paths if configured
            if self.covariates and len(self.covariates) > 0:
                cov_paths, _ = select_paths(
                    self.covariates_paths,
                    self.domains[split]['time'],
                    fmt="%Y-%m-%d"
                )
                paths_dict['covariates_paths'] = cov_paths

            # TEST: Use XrDatasetMultiResTest with patch_dims_dict
            if split == "test":
                return XrDatasetMultiResTestSupervised(
                    multires=self.multires,
                    patch_dims_dict=self.patch_dims_dict,
                    strides_test_dict=self.strides_test_dict, 
                    satellite_vars=self.satellite_vars,
                    covariates=self.covariates,
                    models_vars=self.models_vars,
                    target_vars=self.target_vars,
                    var_mapping=self.var_mapping,
                    **paths_dict,
                    mask=self.mask,
                    times=times,
                    **self.xrds_kw,
                    postpro_fn=self.post_fn(rand_obs=self.rand_obs),
                    res=self.res,
                    pad=self.pads[2],
                    stride_test=True,
                    resize=self.resize,
                    subsel_patch_path=f"{self.subsel_path}/patch_in_ocean_{split}_{self.domain_name}_patch_{self.xrds_kw['patch_dims']['yc']}_{self.xrds_kw['strides']['yc']}_resize_x{self.resize}.txt"
                )
            
            # TRAIN/VAL: Use XrDatasetMultiResTrain
            else:
                return XrDatasetMultiResTrainSupervised(
                    multires=self.multires,
                    satellite_vars=self.satellite_vars,
                    covariates=self.covariates,
                    models_vars=self.models_vars,
                    target_vars=self.target_vars,
                    var_mapping=self.var_mapping,
                    **paths_dict,
                    mask=self.mask,
                    times=times,
                    **self.xrds_kw,
                    postpro_fn=self.post_fn(rand_obs=(split == 'train')),
                    res=self.res,
                    pad=self.pads[0 if split == 'train' else 1],
                    stride_test=False,
                    resize=self.resize,
                    subsel_patch_path=f"{self.subsel_path}/patch_in_ocean_{split}_{self.domain_name}_patch_{self.xrds_kw['patch_dims']['yc']}_{self.xrds_kw['strides']['yc']}_resize_x{self.resize}.txt"
                )
            
        self.train_ds = create_dataset('train')
        self.val_ds = create_dataset('val')
        #self.test_ds = create_dataset('test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        sampler = CustomBatchSampler(self.val_ds, batch_size=self.dl_kw["batch_size"])
        return torch.utils.data.DataLoader(self.val_ds, batch_sampler=sampler, num_workers=self.dl_kw["num_workers"])

    def test_dataloader(self):
        return {
            f"patch_x{res}": torch.utils.data.DataLoader(ds, shuffle=False, **self.dl_kw)
            for res, ds in self.test_ds.datasets.items()
        }