from glob import glob
import datetime
import numpy as np
import xarray as xr
import pyresample

def load_data(type="asip"):
    if type=="asip":
        path = glob('/dmidata/users/maxb/ASIP_OSISAF_dataset/ASIP_L3/*nc')
    elif type=="osisaf":
        path = glob('/dmidata/users/maxb/ASIP_OSISAF_dataset/OSISAF_NRT/*/*/*nh*amsr2_????????1200.nc')
    else:
        path = glob('/dmidata/users/maxb/ERA5_DAILY/ERA5_20*.nc')
    return path

def concatenate(paths, var, slices=None, type_coords="index"):
    # initialize with 1st Dataset
    if type_coords=="index":
        ds0 = xr.open_dataset(paths[0]).isel(**(slices or {}))
    else:
        ds0 = xr.open_dataset(paths[0]).sel(**(slices or {}))
    dims = ds0.sizes
    if "yc" in dims:
        data = np.zeros((len(paths),dims["yc"],dims["xc"]))
    else:
        data = np.zeros((len(paths),dims["latitude"],dims["longitude"]))
    data[0] = ds0[var].data
    times = [ds0.time[0].data]
    coords = ds0.coords
    ds0.close()
    # loop 
    for i in range(1,len(paths)):
        if type_coords=="index":
            dsi = xr.open_dataset(paths[i]).isel(**(slices or {}))
        else:
            dsi = xr.open_dataset(paths[i]).sel(**(slices or {}))
        data[i] = dsi[var].data
        times.append(dsi.time[0].data)
        dsi.close()
    # concatenate
    if "yc" in dims:
        concat = xr.Dataset(data_vars={var:(("time", "yc", "xc"), data)},
                        coords=dict(
                            time=times,
                            xc=coords["xc"],
                            yc=coords["yc"],
                            lon=coords["lon"],
                            lat=coords["lat"],
                            ))
    else:
        concat = xr.Dataset(data_vars={var:(("time", "latitude", "longitude"), data)},
                        coords=dict(
                            time=times,
                            latitude=coords["latitude"],
                            longitude=coords["longitude"],
                            ))
    return concat

def load_mfdata(asip_paths, osisaf_paths, 
                covariates_paths, covariates,
                times, slices=None, type_coords="index"):
    def select_paths_from_dates(files, times):
        # compute list of dates from domain
        if isinstance(times, list):
            dates = []
            new_times = []
            for _ in range(len(times)):
                start = datetime.datetime.strptime(times.start, "%Y-%m-%d")
                end = datetime.datetime.strptime(times.stop, "%Y-%m-%d")
                dates.extend([(start + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(0, (end-start).days)])
                new_times.extend([start + datetime.timedelta(days=x) for x in range(0, (end-start).days)])
        else:
            start = datetime.datetime.strptime(times.start, "%Y-%m-%d")
            end = datetime.datetime.strptime(times.stop, "%Y-%m-%d")
            dates = [ (start + datetime.timedelta(days=x)).strftime("%Y%m%d") for x in range(0, (end-start).days) ]
            new_times = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
        # subselection of paths
        files = np.sort([ f for f in files if any(s in f for s in dates) ])
        return files, new_times
    sel_asip, _ = select_paths_from_dates(asip_paths,times)
    sel_osisaf, _ = select_paths_from_dates(osisaf_paths,times)
    sel_covariates, _ = select_paths_from_dates(covariates_paths,times)
    asip = concatenate(sel_asip, "sic", slices, type_coords)   
    #asip = xr.concat([xr.open_dataset(path) for path in sel_asip],dim="time")
    osisaf = concatenate(sel_osisaf, "ice_conc", slices, type_coords) 
    #osisaf = xr.concat([xr.open_dataset(path) for path in sel_osisaf],dim="time")
    covs = []
    for i in range(len(covariates)):
        covs.append(concatenate(sel_covariates,
                                var=covariates[i])
                    )

    return asip, osisaf, covs

