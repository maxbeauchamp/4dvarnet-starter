import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
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
    dss = {res: xr.open_dataset(ncfiles_by_res[res]).isel(record=10,
                                                     sample=0) for res in multires}

    nrows = len(vars_to_plot)
    ncols = len(multires)

    fig, axes = plt.subplots(nrows, ncols,
                             subplot_kw={"projection": proj},
                             figsize=(4*ncols, 3*nrows))
    if nrows == 1: axes = np.expand_dims(axes, axis=0)
    if ncols == 1: axes = np.expand_dims(axes, axis=1)

    # boucle variables (lignes)
    for i, var in enumerate(vars_to_plot):
        # boucle résolutions (colonnes)
        for j, res in enumerate(multires):
            ax = axes[i, j]
            ds = dss[res]

            # récup lon/lat (2D) + data au temps choisi
            lon = ds["lon"].values
            lat = ds["lat"].values
            lon = denormalize_minmax(lon, -180, 180)
            lat = denormalize_minmax(lat, 50, 90)

            if "time" in ds[var].dims:
                da = ds[var].isel(time=time_index).values
            else:
                da = ds[var].values

            # fond carte & extent serré
            #ax.add_feature(cfeature.OCEAN, color='midnightblue', zorder=0)
            ax.add_feature(cfeature.LAND, color='silver', zorder=1)
            ax.add_feature(cfeature.COASTLINE, zorder=3)
            ax.gridlines(draw_labels=False, x_inline=False, y_inline=False)

            # plot principal
            im = masked_pcolormesh(ax, lon, lat, da)
            ax.set_title(f"{var} - x{res}")

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

                # emprise du fin pour zoom
                zoom_extent_ll = tight_lonlat_extent(lon_f, lat_f, margin=0.00)

                # rectangle indicateur (en lon/lat)
                rect_lon = [zoom_extent_ll[0], zoom_extent_ll[1], zoom_extent_ll[1], zoom_extent_ll[0], zoom_extent_ll[0]]
                rect_lat = [zoom_extent_ll[2], zoom_extent_ll[2], zoom_extent_ll[3], zoom_extent_ll[3], zoom_extent_ll[2]]
                lon_min = float(lon_f.min())
                lon_max = float(lon_f.max())
                lat_min = float(lat_f.min())
                lat_max = float(lat_f.max())
                extent_zoom = [lon_min, lon_max, lat_min, lat_max]
                rect_lon, rect_lat = highres_rectangle(extent_zoom, n_points_per_side=100)
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
                masked_pcolormesh(axins, lon_f, lat_f, da_f)
                axins.set_extent(zoom_extent_ll, crs=ccrs.PlateCarree())
                #set_polar_circle(axins)

        # une seule colorbar par ligne (colonne la plus à droite)
        cax = fig.add_axes([axes[i, -1].get_position().x1 + 0.01,
                            axes[i, -1].get_position().y0,
                            0.015,
                            axes[i, -1].get_position().height])
        fig.colorbar(axes[i, -1].collections[0], cax=cax, orientation="vertical")

    #plt.tight_layout()
    return fig, axes

# ====================== EXEMPLE D’USAGE ======================
if __name__ == "__main__":
    multires = [50, 10, 2]  # du plus large au plus fin
    ncfiles = {
        50: "/dmidata/users/maxb/PREPROC/preproc_CROSCIM_x50.nc",
        10: "/dmidata/users/maxb/PREPROC/preproc_CROSCIM_x10.nc",
        2:  "/dmidata/users/maxb/PREPROC/preproc_CROSCIM_x2.nc",
    }
    vars_to_plot = ["tgt_sic", "tgt_SIT", "cristal_SSH", "u10"]

    fig, axes = plot_multires_polar(ncfiles, multires, vars_to_plot, time_index=7,
                                    proj=ccrs.NorthPolarStereo())
    fig.savefig("multires_polar_insets.png", dpi=300, bbox_inches="tight")

