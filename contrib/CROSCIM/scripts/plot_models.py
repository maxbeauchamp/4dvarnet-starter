import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.path import Path

# -----------------------------
# Load dataset
# -----------------------------
file_path = "/data/users/maxb/CROSCIM_dataset/out_MOD/MOD5km_2021-03-18.nc"
ds = xr.open_dataset(file_path)

lat = ds["lat"]
lon = ds["lon"]
sic = ds["SIC"]
sit = ds["SIT"]

sic = sic.where(np.isfinite(sic))
sit = sit.where(np.isfinite(sit))

# -----------------------------
# Circular boundary
# -----------------------------
theta = np.linspace(0, 2 * np.pi, 200)
center = [0.5, 0.5]
radius = 0.5

verts = np.vstack([
    np.sin(theta) * radius + center[0],
    np.cos(theta) * radius + center[1]
]).T

circle_path = Path(verts)

# -----------------------------
# Figure
# -----------------------------
fig = plt.figure(figsize=(12, 6))
proj = ccrs.NorthPolarStereo()

# -----------------------------
# SIC plot
# -----------------------------
ax1 = plt.subplot(1, 2, 1, projection=proj)
ax1.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
ax1.coastlines()
ax1.add_feature(cfeature.LAND)
ax1.set_boundary(circle_path, transform=ax1.transAxes)

sic_plot = ax1.pcolormesh(
    lon, lat, sic,
    transform=ccrs.PlateCarree(),
    cmap="Blues",
    vmin=0, vmax=1
)

cbar1 = plt.colorbar(
    sic_plot,
    ax=ax1,
    orientation='horizontal',
    pad=0.05,
    shrink=0.8,
    location='top'
)

cbar1.set_label(r"Sea Ice Concentration ([%])", fontsize=10)

ax1.set_title("SIC")

# -----------------------------
# SIT plot
# -----------------------------
ax2 = plt.subplot(1, 2, 2, projection=proj)
ax2.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.LAND)
ax2.set_boundary(circle_path, transform=ax2.transAxes)

sit_plot = ax2.pcolormesh(
    lon, lat, sit,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    vmin=0,
    vmax=5
)

cbar2 = plt.colorbar(
    sit_plot,
    ax=ax2,
    orientation='horizontal',
    pad=0.05,
    shrink=0.8,
    location='top'
)

cbar2.set_label(r"Sea Ice Thickness ($m$)", fontsize=10)

ax2.set_title("SIT")

# -----------------------------
# Text box
# -----------------------------
textstr = """Training Datasets
HYCOM-CICE 3 years dataset simulation
CIMR and CRISTAL model-based pseudo-observations
ASIP-based ROSE-L observations  

Inputs
CIMR, CRISTAL, ASIP Sea Ice Thickness and Concentrations 
Atmospheric conditions in the assimilation window

Outputs 
Sea Ice Thickness and Concentrations for nowcasting and forecasting up to 3 days"""

fig.text(
    0.5, 0.02, textstr,
    ha='center', va='bottom',
    fontsize=9,
    bbox=dict(
        boxstyle="square,pad=0.5",
        facecolor="lightblue",
        edgecolor="darkblue",
        linewidth=2
    )
)

# -----------------------------
# Layout
# -----------------------------
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()
