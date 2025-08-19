from glob import glob
import datetime
import numpy as np
import xarray as xr
import pyresample
from numpy.lib.stride_tricks import as_strided

VAR_GROUPS = {
    "cimr": ["SIC", "SIT", "Tsurf", "SICnoise", "SITnoise", "Tsurfnoise"],
    "cristal": ["HS", "SIT", "SSH", "HSnoise", "SITnoise", "SSHnoise"],
    "asip": ["sic", "standard_deviation_sic", "status_flag"]
}

COVARIATES = ["msl", "t2m", "u10", "v10", "tcc", "d2m", "ssrd", "strd", "tp"]

VAR_GROUPS = {
    "cimr": ["SIC", "SIT"],
    "cristal": ["SIT", "SSH"],
    "asip": ["sic"]
}

COVARIATES = ["msl", "t2m", "u10", "v10"]

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

def fast_coarsen_xr(ds, factor_y=2, factor_x=2, dims=('yc', 'xc'), mode="mean"):
    
    out = {}
    for var in ds.data_vars:
        if all(d in ds[var].dims for d in dims):
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

def load_data(type="asip"):
    if type == "asip":
        return glob('/dmidata/users/maxb/ASIP_OSISAF_dataset/ASIP_L3/*nc')
    elif type == "cimr":
        return glob('/dmidata/users/maxb/CROSCIM_dataset/out_CIMR/CIMR5km_*nc')
    elif type == "cristal":
        return glob('/dmidata/users/maxb/CROSCIM_dataset/out_CRISTAL/CRISTAL5km_*nc')
    else:
        return glob('/dmidata/users/maxb/CROSCIM_dataset/atm_data/atm5km_*.nc')

def concatenate(paths, var_list, slices=None, type_coords="index", resize=1, domain_limits=None):
    
    import xarray as xr
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

def load_mfdata(asip_paths, cimr_paths, cristal_paths,
                covariates_paths, covariates,
                times, slices=None, type_coords="index",resize=1):
    def select_paths_from_dates(files, times, fmt="%Y%m%d"):
        if isinstance(times, list):
            dates = []
            for t in times:
                start = datetime.datetime.strptime(t.start, "%Y-%m-%d")
                end = datetime.datetime.strptime(t.stop, "%Y-%m-%d")
                dates.extend([(start + datetime.timedelta(days=x)).strftime(fmt) for x in range((end-start).days)])
        else:
            start = datetime.datetime.strptime(times.start, "%Y-%m-%d")
            end = datetime.datetime.strptime(times.stop, "%Y-%m-%d")
            dates = [(start + datetime.timedelta(days=x)).strftime(fmt) for x in range((end-start).days)]
        return np.sort([f for f in files if any(s in f for s in dates)])

    sel_asip = select_paths_from_dates(asip_paths, times)
    sel_cimr = select_paths_from_dates(cimr_paths, times, fmt="%Y-%m-%d")
    sel_cristal = select_paths_from_dates(cristal_paths, times, fmt="%Y-%m-%d")
    sel_covariates = select_paths_from_dates(covariates_paths, times, fmt="%Y-%m-%d")

    asip = concatenate(sel_asip, VAR_GROUPS["asip"], slices, type_coords, resize=resize)
    cimr = concatenate(sel_cimr, VAR_GROUPS["cimr"], None, type_coords)
    cristal = concatenate(sel_cristal, VAR_GROUPS["cristal"], None, type_coords)
    covs = concatenate(sel_covariates, covariates, None, type_coords)

    return asip, cimr, cristal, covs
