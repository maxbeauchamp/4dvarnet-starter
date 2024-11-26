import xarray as xr
import src
import numpy as np
import pickle

def load_data_wcov(path_files=["/Odyssey/private/m19beauc/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                               "/Odyssey/private/m19beauc/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc"],
                   var_names=["nadir_obs","ssh"],
                   new_names=["input","tgt"],
                   path_mask=None,
                   mask_var=None,
                   load_geo=False):

    lon_min, lon_max, lat_min, lat_max = (-68,-52,30,45)
    lon = xr.open_dataset(path_files[0]).sel(lon=slice(lon_min, lon_max),
                                             lat=slice(lat_min,lat_max)).lon.data
    lat = xr.open_dataset(path_files[0]).sel(lon=slice(lon_min, lon_max),
                                              lat=slice(lat_min,lat_max)).lat.data
    data = xr.merge([ xr.open_dataset(path_files[i]).sel(lon=slice(lon_min, lon_max),
                                                         lat=slice(lat_min,lat_max)).assign_coords(lon=lon,
                                                             lat=lat).rename_vars({var_names[i]:new_names[i]})[new_names[i]] for i in range(len(path_files))],
                    compat='override').transpose('time', 'lat', 'lon')

    if path_mask is None:
        mask = np.ones(data[new_names[0]][0].shape)
    else:
        mask = np.isnan(xr.open_dataset(path_mask)[mask_var][0].values.astype(int))

    if load_geo:
        data = data.update({'latv':(('lat','lon'),data.lat.broadcast_like(data[new_names[0]][0]).data),
                            'lonv':(('lat','lon'),data.lon.broadcast_like(data[new_names[0]][0]).data),
                            'mask':(('lat','lon'),mask)})
    return data

def load_data_sst(path, path_mask="/Odyssey/private/m19beauc/IMT/data/mask_sst_gf_ext.nc",
                  var_gt='sst', var_mask='mask_sst'):

    ds = xr.open_dataset(path).rename_vars({var_gt:"tgt"})
    ds_mask = xr.open_dataset(path_mask)
    lon_min = np.min(ds_mask.lon.data)
    lat_min = np.min(ds_mask.lat.data)
    lon_max = np.max(ds_mask.lon.data)+.05
    lat_max = np.max(ds_mask.lat.data)+.05
    ds = ds.sel(lat= slice(lat_min,lat_max), lon=slice(lon_min, lon_max))
    ds = ds.update({'input':(('time','lat','lon'),np.where(ds_mask[var_mask]!=0, ds.tgt.data, np.nan))})[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon')

    #ds = ds.update({"tgt":(("time","lat","lon"),remove_nan(ds.tgt))})

    return ds

def load_data_ssh_sst(path_ssh, path_sst,
                      var_obs_ssh="nadir_obs", var_gt_ssh='ssh',
                      path_mask_sst="/Odyssey/private/m19beauc/IMT/data/mask_sst_gf_ext.nc",
                      var_gt_sst='sst', var_mask_sst='mask_sst'):

    # sst
    ds_sst = xr.open_dataset(path_sst).rename_vars({var_gt_sst:"tgt_sst"})
    ds_mask = xr.open_dataset(path_mask_sst)
    lon_min = np.min(ds_mask.lon.data)
    lat_min = np.min(ds_mask.lat.data)
    lon_max = np.max(ds_mask.lon.data)+.05
    lat_max = np.max(ds_mask.lat.data)+.05
    ds_sst = ds_sst.sel(lat= slice(lat_min,lat_max), lon=slice(lon_min, lon_max))
    ds_sst = ds_sst.update({'input_sst':(('time','lat','lon'),np.where(ds_mask[var_mask_sst]!=0, ds_sst.tgt_sst.data, np.nan))})[[*src.data.TrainingItem_sst._fields]].transpose('time', 'lat', 'lon')

    # ssh
    ds_ssh = xr.merge([
             xr.open_dataset(path_ssh).rename_vars({var_obs_ssh:"input_ssh"}),
             xr.open_dataset(path_ssh).rename_vars({var_gt_ssh:"tgt_ssh"})]
           ,compat='override')[[*src.data.TrainingItem_ssh._fields]].transpose('time', 'lat', 'lon') 
    ds_ssh = ds_ssh.sel(lat= slice(lat_min,lat_max), lon=slice(lon_min, lon_max))

    # merge
    ds = xr.merge([ds_ssh,ds_sst])

    return ds

def mask_input(da, mask_list):
    i = np.random.randint(0, len(mask_list))
    mask = mask_list[i]
    da = np.where(np.isfinite(mask), da, np.empty_like(da).fill(np.nan)).astype(np.float32)
    return da

def open_glorys12_data(path, masks_path, domain, variables="zos", masking=True, test_cut=None):
    """
        Function to load glorys data

        path: path to glorys .nc file
        masks_path: path to nadir-like observation masks with dimensions matching glorys dataset size. pickled np array list.
        domain: lat and long extremities to cut data
        variables: variable to load
        masking: whether to mask the input data using the masks in masks_path
        test_cut: if not None, {'time': slice(time1, time2)}, speeding up the loading by pre-cutting the loaded data
    """

    print("LOADING input data")
    # DROPPING DEPTH !!
    ds =  (
        xr.open_dataset(path).drop_vars('depth')
    )
    
    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})


    if test_cut is not None:
        ds = ds.sel(time=test_cut)

    ds = (
        ds
        .load()
        .assign(
            input = lambda ds: ds[variables],
            tgt= lambda ds: ds[variables]
        )
    )
    print("done.")

    if masking:
        print("OPENING mask list")
        with open(masks_path, 'rb') as masks_file:
            mask_list = pickle.load(masks_file)
        mask_list = np.array(mask_list)
        print("done.")

        print("MASKING input data")
        ds= ds.assign(
            input=xr.apply_ufunc(mask_input, ds.input, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], kwargs={"mask_list": mask_list}, dask="allowed", vectorize=True)
            )
        print("done.")
    
    ds = ds.sel(domain)
    ds = (
        ds[[*src.data.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        #.to_array()
    )

    return ds
