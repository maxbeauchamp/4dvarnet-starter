import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable


dataset = xr.open_dataset("data/ERA5_SST_2019.nc").isel(valid_time=slice(0,1400,1))
temperature = dataset['sst']

domain = {"longitude": slice(-65,-55),
          "latitude": slice(30, 40),}
dataset = xr.open_dataset("data/GLO12_2019.nc").sel(**domain).isel(depth=0)
temperature = dataset['thetao'] + 273.15

# Extract latitude, longitude, and time steps
lats = dataset['latitude'].values
lons = dataset['longitude'].values
#times = dataset['valid_time'].values
times = dataset['time'].values

# Select the first 10 time steps for the sequence
num_steps = 6
temperature_sequence = temperature[:num_steps, :, :]

# Apply a Gaussian filter to simulate diffusion (e.g., smoothing the data)
def apply_diffusion(temp_data, sigma=1):
    return temp_data + np.reshape(np.random.normal(0,i,np.prod(temp_data.shape)),
                                  temp_data.shape) #gaussian_filter(temp_data, sigma=sigma)

# Diffusion
# Create figure with subplots
fig, axes = plt.subplots(2, num_steps, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(num_steps):
    print(i)
    # Original temperature field
    ax1 = axes[0, i]
    #ax1.set_title(f"Original {np.datetime_as_string(times[i], unit='h')}")
    im1 = ax1.pcolormesh(lons, lats, temperature_sequence[i], cmap='coolwarm', shading="auto")
    
    # Diffused temperature field
    ax2 = axes[1, i]
    #ax2.set_title(f"Diffused {np.datetime_as_string(times[i], unit='h')}")
    diffused_temp = apply_diffusion(temperature_sequence[num_steps-1],sigma=(i/2)**2)
    im2 = ax2.pcolormesh(lons, lats, diffused_temp, cmap='coolwarm')
# Add colorbars
#fig.colorbar(im1, ax=axes[0, :], orientation='horizontal', pad=0.25, fraction=0.05, label="Temperature (K)")
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig("diffusion.png",dpi=300,bbox_inches='tight',pad_inches = 0,transparent = True)

# Dyffusion
# Create figure with subplots
fig, axes = plt.subplots(2, num_steps, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(num_steps):
    print(i)
    # Original temperature field
    ax1 = axes[0, i]
    #ax1.set_title(f"Original {np.datetime_as_string(times[i], unit='h')}")
    im1 = ax1.pcolormesh(lons, lats, temperature_sequence[i], cmap='coolwarm', shading="auto")
    
    # Diffused temperature field
    ax2 = axes[1, i]
    #ax2.set_title(f"Diffused {np.datetime_as_string(times[i], unit='h')}")
    diffused_temp = temperature_sequence[i]
    im2 = ax2.pcolormesh(lons, lats, diffused_temp, cmap='coolwarm')
# Add colorbars
plt.subplots_adjust(wspace=0.5, hspace=0.5)
#fig.colorbar(im1, ax=axes[0, :], orientation='horizontal', pad=0.25, fraction=0.05, label="Temperature (K)")
plt.savefig("dyffusion.png",dpi=300,bbox_inches='tight',pad_inches = 0,transparent = True)

# ST diffusion
def plot_st_diffusion(sigma=0):

    # Create figure and main axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Define shifts for subplots
    x_shift = 0.06  # Shift in x-direction
    y_shift = 0.05  # Shift in y-direction

    # Plot each 2D map as a shifted subplot
    for i in range(num_steps-1,0,-1):
        left = 0.05 + (num_steps-i) * x_shift  # X position shift
        bottom = 0.05 + i * y_shift  # Y position shift
        width = 0.4
        height = 0.4

        ax_sub = fig.add_axes([left, bottom, width, height])
        temp_data = temperature_sequence[i]
        im = ax_sub.imshow(temp_data +np.reshape(np.random.normal(0,sigma,np.prod(temp_data.shape)),
                                                               temp_data.shape),
                            origin='lower', cmap='coolwarm')
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])
    
        # Add colorbar only for the last subplot
        #if i == 0:
        #    divider = make_axes_locatable(ax_sub)
        #    cax = divider.append_axes("right", size="5%", pad=0.05)
        #    fig.colorbar(im, cax=cax, orientation='vertical')  
    plt.tight_layout()
    plt.savefig("st_diffusion_"+str(sigma)+".png",dpi=300,bbox_inches='tight',pad_inches = 0,transparent = True)
    
# ST diffusion (stacked in one row)
def plot_st_diffusion2(sigma=0):
    fig, axes = plt.subplots(1, num_steps-1, figsize=(3*(num_steps-1), 4), constrained_layout=True)

    # Ensure axes is iterable
    if num_steps-1 == 1:
        axes = [axes]

    for idx, i in enumerate(range(num_steps-1, 0, -1)):
        temp_data = temperature_sequence[i]
        noisy_data = temp_data + np.reshape(
            np.random.normal(0, sigma, np.prod(temp_data.shape)),
            temp_data.shape
        )

        im = axes[idx].imshow(noisy_data, origin="lower", cmap="coolwarm")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

        # Optional: title with step index
        axes[idx].set_title(f"Step {i}")

    # Add a single shared colorbar
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)

    plt.savefig(
        f"st_diffusion_{sigma}_2.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True
    )
    plt.close(fig)


plot_st_diffusion2(sigma=0)
plot_st_diffusion2(sigma=3)
plot_st_diffusion2(sigma=6)
plot_st_diffusion2(sigma=10)


# Pseudo-observations (SST cloud cover / altimetry tracks)
def generate_cloud_mask(shape, coverage=0.3, sigma=5, seed=None):
    """Boolean mask, True where an SST obs is available (cloud-free)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape)
    smooth = gaussian_filter(noise, sigma=sigma)
    thresh = np.quantile(smooth, 1 - coverage)
    return smooth >= thresh


def generate_track_mask(shape, angle_deg=20, spacing=60, width=4, wide=False, seed=None):
    """Boolean mask, True where an altimetry obs is available.

    wide=False mimics a thin nadir track (Jason/Envisat-like), repeated every
    `spacing` pixels. wide=True mimics a SWOT-like swath with a nadir gap.
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    angle = np.deg2rad(angle_deg)
    perp = xx * np.cos(angle) + yy * np.sin(angle)
    offset = rng.uniform(0, spacing)
    band = (perp + offset) % spacing

    track_width = spacing * 0.5 if wide else width
    mask = band < track_width
    if wide:
        gap_width = track_width * 0.15
        nadir_gap = np.abs(band - track_width / 2) < gap_width / 2
        mask = mask & ~nadir_gap
    return mask


def plot_st_pseudo_obs(obs_type="sst", seed=0, **mask_kwargs):
    # Create figure and main axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Define shifts for subplots
    x_shift = 0.06  # Shift in x-direction
    y_shift = 0.05  # Shift in y-direction

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")  # missing obs -> opaque white, not transparent

    rng = np.random.default_rng(seed)

    # Plot each masked 2D map as a shifted subplot
    for i in range(num_steps-1,0,-1):
        left = 0.05 + (num_steps-i) * x_shift  # X position shift
        bottom = 0.05 + i * y_shift  # Y position shift
        width = 0.4
        height = 0.4

        ax_sub = fig.add_axes([left, bottom, width, height])
        temp_data = np.asarray(temperature_sequence[i])
        step_seed = int(rng.integers(1_000_000))

        if obs_type == "sst":
            mask = generate_cloud_mask(temp_data.shape, seed=step_seed, **mask_kwargs)
        elif obs_type == "altimetry_nadir":
            mask = generate_track_mask(temp_data.shape, wide=False, seed=step_seed, **mask_kwargs)
        elif obs_type == "altimetry_swot":
            mask = generate_track_mask(temp_data.shape, wide=True, seed=step_seed, **mask_kwargs)
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

        obs_data = np.ma.masked_invalid(np.where(mask, temp_data, np.nan))
        im = ax_sub.imshow(obs_data, origin='lower', cmap=cmap)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])

    plt.tight_layout()
    plt.savefig(
        f"st_pseudo_obs_{obs_type}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)


plot_st_pseudo_obs(obs_type="sst", coverage=0.3, sigma=5)
plot_st_pseudo_obs(obs_type="altimetry_nadir", spacing=60, width=4)
plot_st_pseudo_obs(obs_type="altimetry_swot", spacing=60)

