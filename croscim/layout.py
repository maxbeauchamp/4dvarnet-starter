"""
layout.py
=========
Constants shared by the package and the conversion script, without heavy
imports (no torch): grid projection and extent, source variables and Zarr
store names.
"""

import datetime as dt

# ASIP's native 500 m grid projection (EPSG:3411)
PROJ4 = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs"
HALF_EXTENT_M = 3_200_000
STORES = {"asip": "ASIP_L3_zarr", "cimr": "data_noise_zarr", "cristal": "out_CRISTAL_zarr", "era5": "atm_data_zarr"}
SOURCES = {"cimr": ("SIC", "SIT"), "cristal": ("SIT", "HS"), "era5": ("t2m", "msl", "u10", "v10")}
EPOCH = dt.date(1970, 1, 1)
