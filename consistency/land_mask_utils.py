import os

import cartopy
import cartopy.feature as cfeature
import geopandas as gpd
import numpy as np
from geopandas import GeoSeries

_CARTOPY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cartopy')
cartopy.config['pre_existing_data_dir'] = _CARTOPY_DIR
cartopy.config['data_dir'] = _CARTOPY_DIR

_land_gdf = None


def _land_polygons():
    global _land_gdf
    if _land_gdf is None:
        land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m')
        _land_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=list(land_50m.geometries()))
    return _land_gdf


def land_mask_from_latlon(lat, lon):
    """Boolean array, True where (lat, lon) falls on land (Natural Earth 50m polygons).

    lon is wrapped to [-180, 180) before the lookup: the ASIP/OSISAF grid stores
    longitudes in [0, 360), which otherwise misclassifies points against the
    EPSG:4326 land polygons (this is why the dataset's precomputed `land_mask`
    is wrong).
    """
    lat = np.asarray(lat)
    lon = ((np.asarray(lon) + 180) % 360) - 180
    pts = gpd.GeoDataFrame(
        geometry=GeoSeries(gpd.points_from_xy(lon.ravel(), lat.ravel())),
        crs='epsg:4326',
    )
    joined = gpd.sjoin(pts, _land_polygons(), how='left', predicate='within')
    is_land = joined['index_right'].notnull().to_numpy()
    return is_land.reshape(lat.shape)
