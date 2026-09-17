import os

import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy
import cmcrameri as cmc

# Offline cartopy data (no internet access) -- same convention as
# Notebooks/CROSCIM/Notebook_Benchmark_CROSCIM_SIT.ipynb. Resolved relative to
# this file so it works regardless of the current working directory.
_CARTOPY_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cartopy')
cartopy.config['pre_existing_data_dir'] = _CARTOPY_DATA_DIR
cartopy.config['data_dir'] = _CARTOPY_DATA_DIR

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def highres_rectangle(extent, n_points_per_side=50):
    """
    Crée un rectangle en coordonnées géographiques avec plus de points par côté
    pour éviter les déformations lors de la projection.

    extent : [lon_min, lon_max, lat_min, lat_max]
    n_points_per_side : nombre de segments par côté
    """
    lon_min, lon_max, lat_min, lat_max = extent

    # côtés
    top = np.column_stack([np.linspace(lon_min, lon_max, n_points_per_side), np.full(n_points_per_side, lat_max)])
    right = np.column_stack([np.full(n_points_per_side, lon_max), np.linspace(lat_max, lat_min, n_points_per_side)])
    bottom = np.column_stack([np.linspace(lon_max, lon_min, n_points_per_side), np.full(n_points_per_side, lat_min)])
    left = np.column_stack([np.full(n_points_per_side, lon_min), np.linspace(lat_min, lat_max, n_points_per_side)])

    # concaténer et fermer le polygone
    coords = np.vstack([top, right, bottom, left, top[0:1]])
    return coords[:,0], coords[:,1]

# -------- util: masque les cellules qui “wrap” (sauts de longitude) --------
def z_masked_overlap(axe, X, Y, Z, source_projection=None):
    if not hasattr(axe, 'projection'):
        return X, Y, Z
    if not isinstance(axe.projection, ccrs.Projection):
        return X, Y, Z
    if (X.ndim != 2) or (Y.ndim != 2):
        return X, Y, Z

    if (source_projection is not None and isinstance(source_projection, ccrs.Geodetic)):
        tp = axe.projection.transform_points(source_projection, X, Y)
        ptx, pty = tp[..., 0], tp[..., 1]
    else:
        ptx, pty = X, Y

    with np.errstate(invalid='ignore'):
        d0 = np.hypot(ptx[1:, 1:] - ptx[:-1, :-1], pty[1:, 1:] - pty[:-1, :-1])
        d1 = np.hypot(ptx[1:, :-1] - ptx[:-1, 1:], pty[1:, :-1] - pty[:-1, 1:])
        half_span = abs(axe.projection.x_limits[1] - axe.projection.x_limits[0]) / 2
        to_mask = (d0 > half_span) | np.isnan(d0) | (d1 > half_span) | np.isnan(d1)

        # si Z est à la même taille que to_mask, étend le masque au bord
        if (to_mask.shape[0] == Z.shape[0] - 1) and (to_mask.shape[1] == Z.shape[1] - 1):
            ext = np.zeros_like(Z, dtype=bool)
            ext[:-1, :-1] = to_mask
            ext[-1, :] = ext[-2, :]
            ext[:, -1] = ext[:, -2]
            to_mask = ext

        Zm = np.ma.masked_where(to_mask, Z)
        return ptx, pty, Zm

# -------- util: pcolormesh géodésique + masque wrap --------
def masked_pcolormesh(ax, lon2d, lat2d, data2d, **kwargs):
    X, Y, Zm = z_masked_overlap(ax, lon2d, lat2d, data2d, source_projection=ccrs.Geodetic())
    # ICI: X,Y sont déjà dans la projection de l’axe
    return ax.pcolormesh(X, Y, Zm, transform=ax.projection, shading="auto", **kwargs)

# -------- util: extent serré d’un (lon,lat) 2D --------
def tight_lonlat_extent(lon2d, lat2d, margin=0.0):
    lon2d = np.asarray(lon2d)
    lat2d = np.asarray(lat2d)
    # Antimeridian-aware: a patch near the pole can span e.g. 179 deg to
    # -179 deg (a narrow strip crossing +-180), which raw min/max would read
    # as an almost-360-deg-wide extent -- a "rectangle" that wide becomes a
    # full circle once projected in polar stereographic. Detect that case
    # and unwrap by shifting negative longitudes by +360 before taking
    # min/max, so the extent reflects the patch's true (narrow) width.
    # Left unwrapped (may exceed 180) on purpose: ax.set_extent accepts
    # dateline-crossing extents in that form.
    if np.nanmax(lon2d) - np.nanmin(lon2d) > 180:
        lon2d = np.where(lon2d < 0, lon2d + 360, lon2d)
    lon_min = np.nanmin(lon2d); lon_max = np.nanmax(lon2d)
    lat_min = np.nanmin(lat2d); lat_max = np.nanmax(lat2d)
    dl = (lon_max - lon_min) * margin
    dphi = (lat_max - lat_min) * margin
    return [lon_min - dl, lon_max + dl, lat_min - dphi, lat_max + dphi]
    #return [-180,180,70,90]

# -------- util: cercle pour dôme polaire --------
def set_polar_circle(ax):
    if not hasattr(ax, "set_boundary"):
        return
    theta = np.linspace(0, 2*np.pi, 200)
    center = [0.5, 0.5]
    radius = 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    if verts.size == 0:
        return
    ax.set_boundary(mpath.Path(verts * radius + center), transform=ax.transAxes)

def denormalize_minmax(norm_data, min_val, max_val):
    return norm_data * (max_val - min_val) + min_val

def denormalize_zscore(norm_data, mean, std):
    return norm_data * std + mean

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# SIC colormap: cmcrameri's perceptually-uniform "oslo" (dark-to-light blue),
# truncated to its upper 80% so the darkest (near-black) end -- hard to read
# against the land/coastline overlay -- is dropped.
_sic_cmap = truncate_colormap(cmc.cm.oslo, minval=0.2, maxval=1.0, n=100)

# ====================== PER-VARIABLE DISPLAY STYLE ======================
# Normalisation stats sourced from
# config/xp/CROSCIM/FM_Swin_solvers/base_arctic_croscim_wpreproc_sit_supervised_wbounds.yaml
# (norm_stats / norm_stats_covs blocks). These describe the shared CROSCIM
# preprocessing and are the same across experiments regardless of which
# variable is the target.
#
# These are the raw SATELLITE inputs (this script visualises observations,
# not model/target fields): asip_sic (ASIP, 0-100 percent scale -- unlike
# models_SIC/cimr.SIC which are 0-1 fractions), cimr_SIT.
NORM_STATS = {
    "asip_sic":    {"type": "minmax", "min": 0.0, "max": 100.0},
    "cimr_SIT":    {"type": "zscore", "mean": 0.5695980677516013, "std": 0.8216149733058172},
    "cristal_SSH": {"type": "zscore", "mean": 0.1912636630889516, "std": 0.41697635927509086},
    "u10":         {"type": "zscore", "mean": 0.6010106010949151, "std": 4.545559141871505},
}

# Colour scale per variable: SIC = SIC-notebook convention, SIT = SIT-notebook
# convention, SSH/u10 = diverging (signed quantities), distinct colormaps so
# adjacent rows aren't visually confusable.
VAR_STYLE = {
    "asip_sic":    dict(cmap=_sic_cmap, vmin=0.0,  vmax=1.0,  label="Sea Ice Concentration"),
    "cimr_SIT":    dict(cmap="plasma",  vmin=0.0,  vmax=4.0,  label="Sea Ice Thickness (m)"),
    "cristal_SSH": dict(cmap="RdBu_r",  vmin=-1.0, vmax=1.0,  label="Sea Surface Height anomaly (m)"),
    "u10":         dict(cmap="PuOr",    vmin=-15., vmax=15.,  label="10 m zonal wind speed (m/s)"),
}

# Row title prefix per variable, used as "{PREFIX} (x{res})".
VAR_DISPLAY = {
    "asip_sic": "SIC",
    "cimr_SIT": "SIT",
    "cristal_SSH": "SSH",
    "u10": "u10",
}

def denormalize_var(var, data):
    """Denormalise `data` for `var` using NORM_STATS; returns `data` unchanged
    if `var` has no entry (e.g. an unrecognised variable passed by a caller)."""
    stats = NORM_STATS.get(var)
    if stats is None:
        return data
    if stats["type"] == "zscore":
        return denormalize_zscore(data, stats["mean"], stats["std"])
    if stats["type"] == "minmax":
        return denormalize_minmax(data, stats["min"], stats["max"])
    return data

# ====================== MAIN PLOTTER ======================
def plot_multires_polar(ncfiles_by_res, multires, vars_to_plot, time_index=0,
                        proj=ccrs.NorthPolarStereo()):
    """
    ncfiles_by_res: dict {res: path_to_nc}
    multires: list comme [50,10,2] (du plus large au plus fin)
    vars_to_plot: liste de variables à tracer (lignes)
    time_index: index temporel à tracer
    """
    # charge datasets
    dss = {res: xr.open_dataset(ncfiles_by_res[res]).isel(sample=0) for res in multires}

    nrows = len(vars_to_plot)
    ncols = len(multires)

    fig, axes = plt.subplots(nrows, ncols,
                             subplot_kw={"projection": proj},
                             figsize=(4*ncols, 3*nrows))
    if nrows == 1: axes = np.expand_dims(axes, axis=0)
    if ncols == 1: axes = np.expand_dims(axes, axis=1)

    # boucle variables (lignes)
    for i, var in enumerate(vars_to_plot):
        style = VAR_STYLE.get(var, dict(cmap="viridis", vmin=None, vmax=None, label=var))
        var_label = VAR_DISPLAY.get(var, var)

        # boucle résolutions (colonnes)
        for j, res in enumerate(multires):
            ax = axes[i, j]
            ds = dss[res]

            # récup lon/lat (2D) + data au temps choisi -- normalisés minmax
            # dans le pipeline de preprocessing (contrib/CROSCIM/dataloaders/data.py,
            # ~L1211-1212: normalize_var(lat, min=50,max=90) / normalize_var(lon,
            # min=-180,max=180)), donc bien besoin de dénormaliser ici.
            lon = ds["lon"].values
            lat = ds["lat"].values
            lon = denormalize_minmax(lon, -180, 180)
            lat = denormalize_minmax(lat, 50, 90)

            if "time" in ds[var].dims:
                da = ds[var].isel(time=time_index).values
            else:
                da = ds[var].values
            da = denormalize_var(var, da)

            # fond carte & extent serré
            #ax.add_feature(cfeature.OCEAN, color='midnightblue', zorder=0)
            ax.add_feature(cfeature.LAND, color='silver', zorder=1)
            ax.add_feature(cfeature.COASTLINE, zorder=3)
            ax.gridlines(draw_labels=False, x_inline=False, y_inline=False)

            # plot principal
            im = masked_pcolormesh(ax, lon, lat, da, cmap=style["cmap"], vmin=style["vmin"], vmax=style["vmax"])
            ax.set_title(f"{var_label} (x{res})")

            # extent sur la zone couverte par cette résolution
            ax.set_extent(tight_lonlat_extent(lon, lat, margin=0.02), crs=ccrs.PlateCarree())
            #set_polar_circle(ax)

            # inset: j -> j+1 (si existe)
            if j < ncols - 1:
                res_f = multires[j+1]
                ds_f = dss[res_f]
                lon_f = ds_f["lon"].values
                lat_f = ds_f["lat"].values
                lon_f = denormalize_minmax(lon_f, -180, 180)
                lat_f = denormalize_minmax(lat_f, 50, 90)
                if "time" in ds_f[var].dims:
                    da_f = ds_f[var].isel(time=time_index).values
                else:
                    da_f = ds_f[var].values
                da_f = denormalize_var(var, da_f)

                # emprise du fin pour zoom (lon/lat, antimeridian-safe --
                # voir tight_lonlat_extent). Note : reste une boîte
                # englobante autour du "losange" réel du patch en
                # projection polaire, donc un peu plus grande que le patch
                # -- un fix en coordonnées natives xc/yc a été tenté puis
                # abandonné : nécessitait de connaître la projection EXACTE
                # ayant produit xc/yc dans ces fichiers, non vérifiable
                # sans accès aux données/à la doc ASIP source.
                zoom_extent_ll = tight_lonlat_extent(lon_f, lat_f, margin=0.00)

                rect_lon, rect_lat = highres_rectangle(zoom_extent_ll, n_points_per_side=100)
                ax.plot(rect_lon, rect_lat, transform=ccrs.PlateCarree(),
                        color="red", lw=1.0, zorder=4)

                # positionne un petit axes en haut-droite de ax
                bbox = ax.get_position()
                iw, ih = bbox.width * 0.45, bbox.height * 0.45
                ix0, iy0 = bbox.x1 - iw*0.95, bbox.y1 - ih*0.95
                axins = fig.add_axes([ix0, iy0, iw, ih], projection=proj)

                # fond & extent de l’inset
                #axins.add_feature(cfeature.OCEAN, color='midnightblue', zorder=0)
                axins.add_feature(cfeature.LAND, color='silver', zorder=1)
                axins.add_feature(cfeature.COASTLINE, zorder=3)
                masked_pcolormesh(axins, lon_f, lat_f, da_f, cmap=style["cmap"], vmin=style["vmin"], vmax=style["vmax"])
                axins.set_extent(zoom_extent_ll, crs=ccrs.PlateCarree())
                #set_polar_circle(axins)

        # une seule colorbar par ligne (colonne la plus à droite), avec légende
        cax = fig.add_axes([axes[i, -1].get_position().x1 + 0.01,
                            axes[i, -1].get_position().y0,
                            0.015,
                            axes[i, -1].get_position().height])
        fig.colorbar(axes[i, -1].collections[0], cax=cax, orientation="vertical").set_label(style["label"], fontsize=10)

    #plt.tight_layout()
    return fig, axes

# ====================== EXEMPLE D’USAGE ======================
if __name__ == "__main__":
    multires = [50, 10, 2]  # du plus large au plus fin
    ncfiles = {
        50: "/Odyssey/public/CROSCIM_dataset/preproc_CROSCIM_x50.nc",
        10: "/Odyssey/public/CROSCIM_dataset/preproc_CROSCIM_x10.nc",
        2:  "/Odyssey/public/CROSCIM_dataset/preproc_CROSCIM_x2.nc",
    }
    vars_to_plot = ["asip_sic", "cimr_SIT", "cristal_SSH", "u10"]

    fig, axes = plot_multires_polar(ncfiles, multires, vars_to_plot, time_index=7,
                                    proj=ccrs.NorthPolarStereo())
    fig.savefig("multires_polar_insets.png", dpi=300, bbox_inches="tight")
