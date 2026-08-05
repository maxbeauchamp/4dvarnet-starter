import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

from glob import glob
import datetime
import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import as_strided
from joblib import Parallel, delayed
#import cupy as cp
#from cupy.lib.stride_tricks import as_strided as as_strided_cp


# Default values (can be overridden)
DEFAULT_VAR_GROUPS = {
    "cimr": ["SIC", "SIT"],
    "cristal": ["SIT", "SSH"],
    "asip": ["sic"],
}

DEFAULT_COVARIATES = ["msl", "t2m", "u10", "v10"]

# Native pixel spacing (meters) per satellite source, used to pick the
# reference/grid-defining source when ASIP is not active.
SOURCE_RESOLUTION_M = {"asip": 500, "cimr": 5000, "cristal": 5000}
# Fixed priority in case of a resolution tie (cimr and cristal are both
# documented as "already at 5km", see the load_mfdata comment below).
_RESOLUTION_TIEBREAK_ORDER = ["cimr", "cristal"]


def resolve_reference_source(active_sources, preferred="asip", override=None):
    """Choose which source's grid/time axis is used as reference.

    - `override` (explicit config value) wins if set.
    - Else `preferred` ("asip") wins if active.
    - Else the active source with the smallest SOURCE_RESOLUTION_M (finest
      resolution) wins; ties broken by _RESOLUTION_TIEBREAK_ORDER.
    """
    if override is not None:
        if override not in active_sources:
            raise ValueError(f"reference_source override {override!r} is not in active_sources={active_sources}")
        return override
    if preferred in active_sources:
        return preferred
    candidates = [s for s in active_sources if s in SOURCE_RESOLUTION_M]
    if not candidates:
        raise ValueError(f"No known-resolution source among active_sources={active_sources} to use as reference grid")
    best_res = min(SOURCE_RESOLUTION_M[s] for s in candidates)
    tied = [s for s in candidates if SOURCE_RESOLUTION_M[s] == best_res]
    for s in _RESOLUTION_TIEBREAK_ORDER:
        if s in tied:
            return s
    return tied[0]


# Static, pre-built reference-grid files (contrib/CROSCIM/scripts/build_grid_reference.py),
# one per nominal `multires` level (e.g. gridref_x50.nc for level 50 -> 25km).
# They hold the fixed asip-derived xc/yc/lon/lat grid used as the interpolation
# target for every active source, independently of which sources are active —
# so the spatial grid never depends on `satellite_vars`/`reference_source`.
_GRIDREF_DIR_DEFAULT = "/Odyssey/private/m19beauc/4dvarnet-starter/contrib/CROSCIM/gridref"


def gridref_path(level, gridref_dir=None):
    """Path to the static reference-grid file for a given nominal multires level."""
    return os.path.join(gridref_dir or _GRIDREF_DIR_DEFAULT, f"gridref_x{level}.nc")


def denormalize_minmax(norm_data, min_val, max_val):
    return norm_data * (max_val - min_val) + min_val

def summarize_lonlat(lon, lat):
    for name, arr in zip(['lon', 'lat'], [lon, lat]):
        arr = np.asarray(arr)
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        center = arr[arr.shape[0] // 2, arr.shape[1] // 2] if arr.ndim == 2 else arr[len(arr) // 2]
        print(f"{name.upper()}: min={vmin:.4f}, max={vmax:.4f}, center={center:.4f}")

def fast_pool(var, fy, fx, mode="mean"):
    arr = var.values
    *leading, ny, nx = arr.shape
    if ny % fy != 0 or nx % fx != 0:
        arr = arr[..., :ny - (ny % fy), :nx - (nx % fx)]
    shape = (*leading, ny // fy, fy, nx // fx, fx)
    strides = (*arr.strides[:-2], arr.strides[-2]*fy, arr.strides[-2], arr.strides[-1]*fx, arr.strides[-1])
    blocks = as_strided(arr, shape=shape, strides=strides)
    if mode=="mean":
        return np.nanmean(blocks, axis=(-1, -3))
    else:
        return ((np.nanmean(blocks, axis=(-1, -3)))==1.).astype(np.float32)
    
def fast_pool_gpu(var, fy, fx, mode="mean", device=1):
    with cp.cuda.Device(device):
        arr = cp.asarray(var.values if hasattr(var, "values") else var)
        *leading, ny, nx = arr.shape

        if ny % fy != 0 or nx % fx != 0:
            arr = arr[..., :ny - (ny % fy), :nx - (nx % fx)]

        shape = (*leading, ny // fy, fy, nx // fx, fx)
        strides = (
            *arr.strides[:-2],
            arr.strides[-2] * fy,
            arr.strides[-2],
            arr.strides[-1] * fx,
            arr.strides[-1]
        )
        blocks = as_strided_cp(arr, shape=shape, strides=strides)

        if mode == "mean":
            out = cp.nanmean(blocks, axis=(-1, -3))
        else:
            out = ((cp.nanmean(blocks, axis=(-1, -3))) == 1.).astype(cp.float32)

        out = cp.asnumpy(out)
        return out 

def fast_coarsen_xr(ds, factor_y=2, factor_x=2, dims=('yc', 'xc'), mode="mean", gpu="False",device=1):
    
    out = {}
    for var in ds.data_vars:
        if all(d in ds[var].dims for d in dims):
            if gpu=="True":
                out[var] = (ds[var].dims, fast_pool_gpu(ds[var], factor_y, factor_x, mode=mode, device=device))
            else:   
                out[var] = (ds[var].dims, fast_pool(ds[var], factor_y, factor_x, mode=mode))
        else:
            out[var] = ds[var]

    new_coords = {}
    for d, factor in zip(dims, [factor_y, factor_x]):
        coord = ds.coords[d].values
        coord = coord[: len(coord) - (len(coord) % factor)]
        coord_new = coord.reshape(-1, factor).mean(axis=1)
        new_coords[d] = coord_new

    for c in ds.coords:
        if c not in dims and c not in ['lat', 'lon']:
            new_coords[c] = ds.coords[c]

    # lat/lon coarsening
    for var in ['lon', 'lat']:
        if var in ds.coords:
            if gpu=="True":
                pooled = fast_pool_gpu(ds[var], factor_y, factor_x, device=device)
            else:
                pooled = fast_pool(ds[var], factor_y, factor_x)
            new_coords[var] = (dims, pooled)

    return xr.Dataset(out, coords=new_coords)

def fast_coarsen_xr_array(da, factor_y=2, factor_x=2, dims=('yc', 'xc'), mode="mean"):
    """
    Coarsen a DataArray along two spatial dimensions (e.g., 'yc', 'xc') using pooling.

    Parameters:
    - da: xarray.DataArray
    - factor_y: int, coarsening factor along the first dimension (e.g., 'yc')
    - factor_x: int, coarsening factor along the second dimension (e.g., 'xc')
    - dims: tuple of two str, names of the dimensions to coarsen (e.g., ('yc', 'xc'))
    - mode: str, reduction method ('mean', 'sum', 'max', etc.)

    Returns:
    - xarray.DataArray with coarsened data and updated coordinates
    """

    # Apply pooling
    pooled = fast_pool(da, factor_y, factor_x, mode=mode)

    # Handle coordinates
    new_coords = {}
    for d, factor in zip(dims, [factor_y, factor_x]):
        coord = da.coords[d].values
        coord = coord[: len(coord) - (len(coord) % factor)]
        coord_new = coord.reshape(-1, factor).mean(axis=1)
        new_coords[d] = coord_new

    # Preserve other coordinates
    for c in da.coords:
        if c not in dims and c not in ['lat', 'lon']:
            new_coords[c] = da.coords[c]

    # Optionally coarsen lat/lon
    for var in ['lon', 'lat']:
        if var in da.coords:
            pooled_coord = fast_pool(da.coords[var], factor_y, factor_x)
            new_coords[var] = (dims, pooled_coord)

    return xr.DataArray(
        pooled,
        dims=da.dims,
        coords=new_coords,
        attrs=da.attrs
    )


def load_data(paths={"asip":"/Odyssey/public/CROSCIM_dataset/ASIP_L3",
                     "cimr":"/Odyssey/public/CROSCIM_dataset/data_noise",
                     "cristal":"/Odyssey/public/CROSCIM_dataset/out_CRISTAL",
                     "covariates":"/Odyssey/public/CROSCIM_dataset/atm_data",
                     "models":"/Odyssey/public/CROSCIM_dataset/out_MOD"},
                     type="asip"):
    
    """Load file paths for a given data type."""
    if type == "asip":
        return glob(paths["asip"] + "/*nc")
    elif type == "cimr":
        return glob(paths["cimr"] + "/CIMR5km_*nc")
    elif type == "cristal":
        return glob(paths["cristal"] + "/CRISTAL5km_*nc")
    elif type == "models":  
        return glob(paths["models"] + "/MOD5km_*nc")
        return glob(paths["covariates"] + "/atm5km_*.nc")


def concatenate(paths, var_list, slices=None, type_coords="index", resize=1, domain_limits=None):
    
    import xarray as xr
    from omegaconf import ListConfig
    # Convert OmegaConf ListConfig to Python list
    if isinstance(var_list, ListConfig):
        var_list = list(var_list)
    # initialize with 1st Dataset
    ds = xr.open_dataset(paths[0])
    if domain_limits is not None:
        ds = ds.sel(**(domain_limits or {}))
    times = [ds.time[0].data]
    ds = ds[var_list]
    if slices is not None:
        if type_coords == "index":
            ds = ds.isel(**slices)
        else:
            ds = ds.sel(**slices)
    if resize!=1:
        ds = fast_coarsen_xr(ds, factor_x=resize, factor_y=resize)
    #summarize_lonlat(ds["lon"].data, ds["lat"].data)
    ds_vars = {}
    for var in var_list:
        if var in ds:
            ds_vars[var] = np.squeeze(ds[var].data)
    # Handle coords properly
    coords = ds.coords
    dims = ds.sizes
    ds.close()

    data_vars = {var: [ds_vars[var]] for var in ds_vars}

    for path in paths[1:]:
        ds = xr.open_dataset(path)
        if domain_limits is not None:
            ds = ds.sel(**(domain_limits or {}))
        times.append(ds.time[0].data)
        ds = ds[var_list]
        if slices is not None:
            if type_coords == "index":
                ds = ds.isel(**slices)
            else:
                ds = ds.sel(**slices)
        if resize!=1:
            ds = fast_coarsen_xr(ds, factor_x=resize, factor_y=resize)
        for var in var_list:
            if var in ds:
                selected = ds[var].data
                data_vars[var].append(np.squeeze(selected))
        ds.close()

    for var in data_vars:
        data_vars[var] = np.stack(data_vars[var], axis=0)

    if "yc" in dims:
        concat = xr.Dataset(
            data_vars={var: (("time", "yc", "xc"), data_vars[var]) for var in data_vars},
            coords=dict(
                time=times,
                xc=coords["xc"],
                yc=coords["yc"],
                lon=coords["lon"],
                lat=coords["lat"]
            )
        )
    else:
        concat = xr.Dataset(
            data_vars={var: (("time", "latitude", "longitude"), data_vars[var]) for var in data_vars},
            coords=dict(
                time=times,
                latitude=coords["latitude"],
                longitude=coords["longitude"]
            )
        )

    return concat

def process_single_file(path, var_list, slices, type_coords, resize, domain_limits, 
                        return_coords=False):
    """Process a single netCDF file and return the data."""
    from omegaconf import ListConfig
    # Convert OmegaConf ListConfig to Python list
    if isinstance(var_list, ListConfig):
        var_list = list(var_list)
    ds = xr.open_dataset(path)
    if domain_limits is not None:
        ds = ds.sel(**(domain_limits or {}))
    time = ds.time[0].data

    ds = ds[var_list]

    if slices is not None:
        if type_coords == "index":
            ds = ds.isel(**slices)
        else:
            ds = ds.sel(**slices)
    if resize != 1:
        ds = fast_coarsen_xr(ds, factor_x=resize, factor_y=resize)
    
    result = {}
    for var in var_list:
        if var in ds:
            result[var] = np.squeeze(ds[var].data)
    
    if return_coords:
        coords = ds.coords
        dims = ds.sizes
        ds.close()
        return time, result, coords, dims
    else:
        ds.close()
        return time, result

def concatenate_parallel(paths, var_list, 
                        slices=None, type_coords="index",
                        resize=1, domain_limits=None, n_jobs=15):

    print(slices)
    
    # Process first file separately to get coordinates
    time0, result0, coords, dims = process_single_file(
        paths[0], var_list, slices, type_coords, resize, domain_limits, return_coords=True
    )
    
    # Parallel processing of remaining files
    if len(paths) > 1:
        # Use 'loky' backend instead of 'threading' to avoid HDF5 thread-safety issues
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(process_single_file)(path, var_list, slices, type_coords, resize, domain_limits, return_coords=False)
            for path in paths[1:]
        )
        # Combine first result with parallel results
        all_results = [(time0, result0)] + results
    else:
        all_results = [(time0, result0)]
    
    # Unpack results
    times = [r[0] for r in all_results]
    data_vars = {var: [] for var in var_list}
    for time, result in all_results:
        for var in var_list:
            if var in result:
                data_vars[var].append(result[var])
    
    # Stack arrays
    for var in data_vars:
        if data_vars[var]:
            data_vars[var] = np.stack(data_vars[var], axis=0)

    # Create dataset
    if "yc" in dims:
        concat = xr.Dataset(
            data_vars={var: (("time", "yc", "xc"), data_vars[var]) for var in data_vars if len(data_vars[var]) > 0},
            coords=dict(
                time=times,
                xc=coords["xc"],
                yc=coords["yc"],
                lon=coords["lon"],
                lat=coords["lat"]
            )
        )
    else:
        concat = xr.Dataset(
            data_vars={var: (("time", "latitude", "longitude"), data_vars[var]) for var in data_vars if len(data_vars[var]) > 0},
            coords=dict(
                time=times,
                latitude=coords["latitude"],
                longitude=coords["longitude"]
            )
        )

    return concat

def load_mfdata(times,
                satellite_vars=None,
                covariates=None,
                models_vars=None,
                slices=None,
                path_loaders=None,
                type_coords="index",
                resize=1,
                domain_limits=None,
                reference_source=None):
    """
    Load multi-source data with configurable variables.
    Only loads data for satellites that are actually used.

    Args:
        times: Time range(s) to load
        satellite_vars: dict of {source: [var_list]}, e.g., {"cimr": ["SIC", "SIT"], "asip": ["sic"]}
                        If None, uses DEFAULT_VAR_GROUPS
        covariates: list of covariate names, e.g., ["u10", "v10"]
                    If None, uses DEFAULT_COVARIATES
        models_vars: list of model variable names, e.g., ["t2m", "msl", "sic", "sit"]
                     If None or empty, no model data is loaded
        slices: unused (kept for signature compatibility) — every source is
                loaded at native resolution and regridded downstream onto
                the static gridref target grid.
        path_loaders: dict of {source: list_of_paths}
        type_coords: "index" or "values"
        resize: unused (kept for signature compatibility), see `slices` above.
        domain_limits: Optional domain limits dict
        reference_source: unused (kept for signature compatibility) — the
                           spatial grid no longer depends on which source is
                           "reference"; see gridref_path().

    Returns:
        dict of {source: xr.Dataset} for each active source
    """
    # Use defaults if not provided
    if satellite_vars is None:
        satellite_vars = DEFAULT_VAR_GROUPS.copy()
    if covariates is None:
        covariates = DEFAULT_COVARIATES.copy()
    if models_vars is None:
        models_vars = []
    
    def select_paths_from_dates(files, times, fmt="%Y%m%d"):
        if isinstance(times, list):
            dates = []
            for t in times:
                start = datetime.datetime.strptime(t.start, "%Y-%m-%d")
                end = datetime.datetime.strptime(t.stop, "%Y-%m-%d")
                dates.extend([(start + datetime.timedelta(days=x)).strftime(fmt) 
                             for x in range((end-start).days)])
        else:
            start = datetime.datetime.strptime(times.start, "%Y-%m-%d")
            end = datetime.datetime.strptime(times.stop, "%Y-%m-%d")
            dates = [(start + datetime.timedelta(days=x)).strftime(fmt) 
                    for x in range((end-start).days)]
        return np.sort([f for f in files if any(s in f for s in dates)])
    
    # Date format for each source
    date_formats = {
        "asip": "%Y%m%d",
        "cimr": "%Y-%m-%d",
        "cristal": "%Y-%m-%d",
        "models": "%Y-%m-%d", 
    }
    
    # Cap parallelism to what's actually available — concatenate_parallel's
    # own default (n_jobs=15) oversubscribes when the process is confined to
    # fewer CPUs than the machine's total (e.g. a cgroup-limited Jupyter
    # kernel on a shared node), which is much slower than running fewer,
    # uncontended workers. os.cpu_count() reports the machine's total
    # logical CPUs and ignores cgroup/affinity limits — os.sched_getaffinity
    # reflects what this process can actually use. A dedicated compute node
    # (e.g. a sbatch job) with no such restriction still gets up to 15.
    try:
        n_available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        n_available_cpus = os.cpu_count() or 1
    n_jobs = min(15, n_available_cpus)

    # Load only required satellite data
    datasets = {}

    for source, vars_list in satellite_vars.items():
        if not vars_list:  # Skip if empty list
            continue
            
        print(f"Loading {source} data for variables: {vars_list}")
        
        # Get paths for this source
        all_paths = path_loaders.get(source, [])
        if len(all_paths) == 0:
            print(f"  Warning: No path_loaders configured for {source}")
            continue
            
        selected_paths = select_paths_from_dates(all_paths, times, fmt=date_formats.get(source, "%Y-%m-%d"))
        
        if len(selected_paths) == 0:
            print(f"  Warning: No files found for {source}")
            continue
        
        # Pre-shrink each source to roughly its own resolution-appropriate
        # pixel count before it's held in memory for the whole eager-loaded
        # period (this is a memory/perf optimization only — the exact target
        # grid is still handled later by XrDataset.interpolate_dataset, via
        # the static gridref, regardless of this factor). `resize` is
        # expressed in units of asip's native 500m spacing; convert it to the
        # raw pixel-binning factor for THIS source's own native spacing, so
        # e.g. asip (500m) at resize=50 doesn't get loaded at full 500m
        # resolution for the whole test period just because a coarser
        # multires level (e.g. patch_x50) also needs cimr/cristal (5km, for
        # which resize=50 already means no coarsening at all).
        source_native_m = SOURCE_RESOLUTION_M.get(source)
        if source_native_m is not None and resize != 1:
            source_factor = max(1, round(resize * SOURCE_RESOLUTION_M["asip"] / source_native_m))
        else:
            source_factor = 1
        datasets[source] = concatenate_parallel(
            selected_paths, vars_list, None, type_coords,
            resize=source_factor,
            domain_limits=domain_limits,
            n_jobs=n_jobs
        )
        
        print(f"  Loaded {source}: {list(datasets[source].data_vars)}, shape: {datasets[source].dims}")
    
    if models_vars:
        print(f"Loading model data for variables: {models_vars}")
        models_paths = path_loaders.get("models", [])
        
        if len(models_paths) > 0:
            selected_model_paths = select_paths_from_dates(
                models_paths, times, fmt=date_formats["models"]
            )
            
            if len(selected_model_paths) > 0:
                # Models are assumed to be at same resolution as CIMR/CRISTAL (5km)
                datasets['models'] = concatenate_parallel(
                    selected_model_paths, models_vars, None, type_coords,
                    domain_limits=domain_limits,
                    n_jobs=n_jobs
                )
                print(f"  Loaded models: {list(datasets['models'].data_vars)}, shape: {datasets['models'].dims}")
            else:
                print("  Warning: No model files found for specified time range")
        else:
            print("  Warning: No model path_loaders configured")
    
    # Load covariates if requested
    if covariates:
        print(f"Loading covariates: {covariates}")
        covariates_paths = path_loaders.get("covariates", [])
        
        if len(covariates_paths) > 0:
            selected_cov_paths = select_paths_from_dates(covariates_paths, times, fmt="%Y-%m-%d")
            
            if len(selected_cov_paths) > 0:
                datasets['covariates'] = concatenate_parallel(
                    selected_cov_paths, covariates, None, type_coords,
                    domain_limits=domain_limits,
                    n_jobs=n_jobs
                )
                print(f"  Loaded covariates: {list(datasets['covariates'].data_vars)}, shape: {datasets['covariates'].dims}")
            else:
                print("  Warning: No covariate files found")
        else:
            print("  Warning: No covariate path_loaders configured")
    
    return datasets

def get_paths_for_source(source,
                        paths={"asip":"/Odyssey/public/CROSCIM_dataset/ASIP_L3",
                            "cimr":"/Odyssey/public/CROSCIM_dataset/data_noise",
                            "cristal":"/Odyssey/public/CROSCIM_dataset/out_CRISTAL",
                            "covariates":"/Odyssey/public/CROSCIM_dataset/atm_data",
                            "models":"/Odyssey/public/CROSCIM_dataset/out_MOD"}):
    """Get all paths for a given data source."""
    path_map = {
        "asip": paths["asip"]+'/*nc',
        "cimr": paths["cimr"]+'/CIMR5km_*nc',
        "cristal": paths["cristal"]+'/CRISTAL5km_*nc',
        "covariates": paths["covariates"]+'/atm5km_*.nc',
        "models": paths["models"]+'/MOD5km_*nc'
    }
    return glob(path_map.get(source, ""))
