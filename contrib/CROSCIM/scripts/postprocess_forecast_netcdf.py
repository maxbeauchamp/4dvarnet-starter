#!/usr/bin/env python3
"""Post-process a CROSCIM forecast/test NetCDF (SIC or SIT) for external
distribution — CMEMS, colleagues using QGIS/Panoply, etc.

Addresses feedback received on the CROSCIM SIC product (equally applicable
to SIT):

1. CF-compliant grid mapping. The raw files don't declare their polar-
   stereographic projection, so GIS tools can't georeference them
   automatically — the workaround was manually deriving the geotransform
   from xc/yc. This adds a `polar_stereographic` grid_mapping variable
   (EPSG:3411 parameters — the exact projection CROSCIM's own notebooks use
   to plot this grid, see DATA_CRS in
   Notebooks/CROSCIM/Notebook_Benchmark_CROSCIM_SI{C,T}.ipynb), a
   `crs_wkt`/`spatial_ref` for GDAL-based tools, correct CF attributes on
   xc/yc/lon/lat, and a `grid_mapping` attribute on every gridded variable.

2. Variable documentation. `pred_*`, `models_*`, and `tgt_*` get `long_name`/
   `comment` attributes. `tgt_*` is config-dependent (see
   contrib/CROSCIM/dataloaders/data.py's var_mapping handling): it can be a
   literal copy of `models_*` (numerical-model reference) OR an observation-
   based field (e.g. `tgt_SIC` = `asip_sic`) depending on the experiment's
   `var_mapping`. This script can't know which one produced a given file, so
   pass --tgt-description to state it explicitly and have it embedded.

3. SOD / floe size / confidence roadmap: out of scope for a post-processing
   script — that's a product-planning answer, not something derivable from
   the NetCDF.

4. Acquisition times: the reconstructed field has one common valid/reference
   time per output step (that's the point of the 4DVarNet fusion — see
   models.py's aggregate_batches), not the underlying satellite acquisition
   time(s). This script documents that explicitly on the `time` coordinate.
   Actually recovering per-observation acquisition times would require a
   change upstream, at data-loading time (contrib/CROSCIM/dataloaders/), not
   a post-processing step on the already-produced file — the raw per-pixel
   acquisition timestamps aren't retained past that point today.

The input file covers the full analysis+forecast window; by default only the
last 3 time steps (the forecast leads — lead-0/1/2, same N_FORECAST=3
convention used throughout the benchmark notebooks) are kept in the output.
Pass --n-forecast to change that, or a value >= the file's time length to
keep everything.

------------------------------------------------------------------------
How to run
------------------------------------------------------------------------
Minimal (keeps only the last 3 forecast time steps, default):

    python postprocess_forecast_netcdf.py IN.nc OUT.nc

Document what tgt_<var> actually is for this file's experiment config
(recommended — see point 2 above):

    python postprocess_forecast_netcdf.py IN.nc OUT.nc \\
        --tgt-description "tgt_SIC = asip_sic (ASIP obs), not models_SIC"

Keep a different number of trailing time steps, or the whole file:

    python postprocess_forecast_netcdf.py IN.nc OUT.nc --n-forecast 5
    python postprocess_forecast_netcdf.py IN.nc OUT.nc --n-forecast 999999
"""
from __future__ import annotations

import argparse
import datetime as _dt

import xarray as xr

# EPSG:3411 — NSIDC Sea Ice Polar Stereographic North (Hughes 1980 ellipsoid).
# Same projection used to plot this grid in the CROSCIM benchmark notebooks:
# ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70,
#     globe=ccrs.Globe(semimajor_axis=6378273.0, semiminor_axis=6356889.449))
_PROJ_ATTRS = {
    "grid_mapping_name": "polar_stereographic",
    "straight_vertical_longitude_from_pole": -45.0,
    "latitude_of_projection_origin": 90.0,
    "standard_parallel": 70.0,
    "false_easting": 0.0,
    "false_northing": 0.0,
    "semi_major_axis": 6378273.0,
    "semi_minor_axis": 6356889.449,
    "epsg_code": "EPSG:3411",
}


def _crs_wkt() -> str:
    """WKT for EPSG:3411, via pyproj if available — falls back to a hardcoded
    WKT (identical to what pyproj produces) so georeferencing still works on
    machines without pyproj installed."""
    try:
        from pyproj import CRS
        return CRS.from_epsg(3411).to_wkt()
    except Exception:
        return (
            'PROJCS["NSIDC Sea Ice Polar Stereographic North",'
            'GEOGCS["Unspecified datum based upon the Hughes 1980 ellipsoid",'
            'DATUM["Not_specified_based_on_Hughes_1980_ellipsoid",'
            'SPHEROID["Hughes 1980",6378273,257.2891366225505]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Polar_Stereographic"],'
            'PARAMETER["latitude_of_origin",70],'
            'PARAMETER["central_meridian",-45],'
            'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
            'UNIT["metre",1],AXIS["Easting",SOUTH],AXIS["Northing",SOUTH],'
            'AUTHORITY["EPSG","3411"]]'
        )


_VAR_DOC = {
    "pred_": "Model prediction (4DVarNet-CROSCIM reconstruction/forecast for this variable)",
    "models_": "Numerical sea-ice model reference field, used as evaluation ground truth",
    "asip_": "ASIP satellite observation (input to the reconstruction, not a prediction)",
    "cimr_": "CIMR satellite observation (input to the reconstruction, not a prediction)",
    "cristal_": "CRISTAL satellite observation (input to the reconstruction, not a prediction)",
}

_TGT_DEFAULT_COMMENT = (
    "Training/loss target for this resolution. Depending on the experiment's "
    "var_mapping config, this is EITHER the numerical-model reference field "
    "(identical to models_<var> when that's the mapping) OR an observation-"
    "based field (e.g. tgt_SIC = asip_sic) — check the xp config that "
    "produced this file, or re-run this script with --tgt-description to "
    "document it explicitly."
)


def _var_comment(name: str, tgt_description: str | None) -> str | None:
    if name.startswith("tgt_"):
        return tgt_description or _TGT_DEFAULT_COMMENT
    for prefix, doc in _VAR_DOC.items():
        if name.startswith(prefix):
            return doc
    return None


def postprocess(in_path: str, out_path: str, tgt_description: str | None = None,
                 n_forecast: int = 3) -> None:
    ds = xr.open_dataset(in_path)

    # ── 0. Keep only the last n_forecast time steps (forecast leads) ───────
    if "time" in ds.dims and n_forecast < ds.sizes["time"]:
        ds = ds.isel(time=slice(-n_forecast, None))

    # ── 1. CF-compliant georeferencing ─────────────────────────────────────
    ds["polar_stereographic"] = xr.DataArray(0, attrs=dict(_PROJ_ATTRS))
    ds["polar_stereographic"].attrs["crs_wkt"] = _crs_wkt()
    ds["spatial_ref"] = ds["polar_stereographic"]  # name some GDAL/rioxarray readers look for

    if "xc" in ds.coords:
        ds["xc"].attrs.update(standard_name="projection_x_coordinate", units="m", axis="X")
    if "yc" in ds.coords:
        ds["yc"].attrs.update(standard_name="projection_y_coordinate", units="m", axis="Y")
    if "lon" in ds.variables:
        ds["lon"].attrs.update(standard_name="longitude", units="degrees_east")
    if "lat" in ds.variables:
        ds["lat"].attrs.update(standard_name="latitude", units="degrees_north")

    # ── 4. Document the "common reference time" convention ─────────────────
    if "time" in ds.coords:
        ds["time"].attrs["comment"] = (
            "Common valid/reference time of the reconstructed field. CROSCIM "
            "fuses asynchronous satellite passes onto this single time grid — "
            "per-observation acquisition times are not preserved in this file."
        )

    # ── 2. Variable documentation + grid_mapping reference ─────────────────
    for var in ds.data_vars:
        if var in ("polar_stereographic", "spatial_ref"):
            continue
        if {"xc", "yc"}.issubset(ds[var].dims):
            ds[var].attrs["grid_mapping"] = "polar_stereographic"
        comment = _var_comment(var, tgt_description)
        if comment:
            ds[var].attrs.setdefault("long_name", comment.split(".")[0].split(" — ")[0])
            ds[var].attrs["comment"] = comment

    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["history"] = (
        ds.attrs.get("history", "").strip() + "\n"
        f"{_dt.datetime.now(_dt.timezone.utc).isoformat()}: postprocess_forecast_netcdf.py"
        " — added CF grid_mapping and variable documentation."
    ).strip()

    ds.to_netcdf(out_path)
    ds.close()
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_path", help="Input CROSCIM forecast/test NetCDF (SIC or SIT)")
    ap.add_argument("out_path", help="Output path for the post-processed NetCDF")
    ap.add_argument(
        "--tgt-description", default=None,
        help="What tgt_<var> actually is in THIS file's config, e.g. "
             "'tgt_SIC = asip_sic (ASIP obs, not models_SIC)'. "
             "Embedded verbatim into tgt_* variable comments.")
    ap.add_argument(
        "--n-forecast", type=int, default=3,
        help="Number of trailing time steps to keep (default: 3, the "
             "forecast leads — same N_FORECAST convention as the benchmark "
             "notebooks). Use a value >= the file's time length to keep everything.")
    args = ap.parse_args()
    postprocess(args.in_path, args.out_path, args.tgt_description, args.n_forecast)


if __name__ == "__main__":
    main()
