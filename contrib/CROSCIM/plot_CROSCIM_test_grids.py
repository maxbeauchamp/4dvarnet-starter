import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

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

def extract_contour_lonlat(lon2d, lat2d):
    """
    Retourne les coordonnées lon/lat du contour complet d'une grille 2D.
    """
    # Bas (ligne 0, colonnes 0 -> fin)
    lon_bottom = lon2d[0, :]
    lat_bottom = lat2d[0, :]

    # Droite (colonne -1, lignes 1 -> fin)
    lon_right = lon2d[1:, -1]
    lat_right = lat2d[1:, -1]

    # Haut (ligne -1, colonnes -2 -> 0, sens inverse)
    lon_top = lon2d[-1, -2::-1]
    lat_top = lat2d[-1, -2::-1]

    # Gauche (colonne 0, lignes -2 -> 1, sens inverse)
    lon_left = lon2d[-2:0:-1, 0]
    lat_left = lat2d[-2:0:-1, 0]

    # Concaténer
    poly_lon = np.concatenate([lon_bottom, lon_right, lon_top, lon_left, lon_bottom[:1]])
    poly_lat = np.concatenate([lat_bottom, lat_right, lat_top, lat_left, lat_bottom[:1]])

    return poly_lon, poly_lat

def multires_polar_test_grids(test_dataloader):

    # Exemple : couleurs associées aux résolutions
    colors_by_res = {
        "patch_x50": "red",
        "patch_x10": "blue",
        "patch_x2": "green"
    }
    
    # Projection principale (orthographique vue polaire nord)
    map_proj = ccrs.Orthographic(central_longitude=10, central_latitude=60)
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection=map_proj))
    
    # Décor
    #ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, color='gainsboro', zorder=0, edgecolor='black')
    ax.gridlines(draw_labels=False)
    ax.set_global()
    
    # Domaine global sur lequel on zoom
    lon_min, lon_max = -180, 180
    lat_min, lat_max = 50, 90
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Rectangle principal
    poly_lon = [
        -180,               # coin bas-gauche
        180,                # bas-droit
        180,                # haut-droit
        -180,               # haut-gauche
        -180                # retour au début
    ]
    poly_lat = [
        50,
        50,
        90,
        90,
        50
    ]
    ax.plot(poly_lon, poly_lat,
            transform=ccrs.PlateCarree(),
            color='darkred',
            linestyle="dashed",
            alpha=0.8,
            zorder=2)
    
    # Récupération des coordonnées par résolution
    # multires_patches est un dict du type :
    # {
    #   "patch_x50": list_of_coords_for_res50,
    #   "patch_x10": list_of_coords_for_res10,
    #   "patch_x2":  list_of_coords_for_res2
    # }
    multires_patches = {}
    for res in ["patch_x50", "patch_x10", "patch_x2"]:
        multires_patches[res] = test_dataloader[res].dataset.get_coords()  # à adapter selon ton datamodule
    
    # Boucle sur résolutions
    for res, coords_list in multires_patches.items():
        color = colors_by_res.get(res, "black")
        for c in coords_list:
            lon2d = c.lon.data
            lat2d = c.lat.data
            lon2d, lat2d, _ = z_masked_overlap(ax, lon2d, lat2d, np.zeros_like(lon2d),
                                               source_projection=ccrs.Geodetic())

            poly_lon, poly_lat = extract_contour_lonlat(lon2d, lat2d)
            
            ax.plot(poly_lon, poly_lat,
                color=color,
                transform=ccrs.PlateCarree(),
                lw=1.0,
                alpha=0.9,
                zorder=3)

    plt.savefig("multires_polar_test_grids.png", dpi=300, bbox_inches="tight")

