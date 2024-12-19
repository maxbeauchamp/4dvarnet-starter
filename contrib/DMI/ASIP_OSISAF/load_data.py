from glob import glob
import datetime
import numpy as np
import xarray as xr
import pyresample

def load_data(type="asip"):
    if type=="asip":
        path = glob('/dmidata/users/maxb/ASIP_OSISAF_dataset/ASIP_L3/*nc')
    else:
        path = glob('/dmidata/users/maxb/ASIP_OSISAF_dataset/OSISAF_NRT/*/*/*nh*amsr2_????????1200.nc')
    return path

def concatenate(paths, var, slices=None):
    # initialize with 1st Dataset
    ds0 = xr.open_dataset(paths[0]).isel(**(slices or {}))
    dims = ds0.sizes
    data = np.zeros((len(paths),dims["yc"],dims["xc"]))
    data[0] = ds0[var].data
    times = [ds0.time[0].data]
    coords = ds0.coords
    ds0.close()
    # loop 
    for i in range(1,len(paths)):
        dsi = xr.open_dataset(paths[i]).isel(**(slices or {}))
        data[i] = dsi[var].data
        times.append(dsi.time[0].data)
        dsi.close()
    # concatenate
    concat = xr.Dataset(data_vars={var:(("time", "yc", "xc"), data)},
                        coords=dict(
                            time=times,
                            xc=coords["xc"],
                            yc=coords["yc"],
                            lon=coords["lon"],
                            lat=coords["lat"],
                            ))
    return concat

def load_mfdata(asip_paths, osisaf_paths, times):
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
    sel_asip, _ = select_paths_from_dates(asip_paths,times)
    sel_osisaf, _ = select_paths_from_dates(osisaf_paths,times)
    asip = concatenate(sel_asip, "sic")   
    #asip = xr.concat([xr.open_dataset(path) for path in sel_asip],dim="time")
    osisaf = concatenate(sel_osisaf, "ice_conc") 
    #osisaf = xr.concat([xr.open_dataset(path) for path in sel_osisaf],dim="time")

    return asip, osisaf

