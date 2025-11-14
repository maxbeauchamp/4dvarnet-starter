from random import sample
import contrib
from contrib.CROSCIM.load_data import *
from contrib.CROSCIM.data_simple import *
import datetime
import pyresample
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature 
import shapely.geometry as sgeom
import os
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def highres_rectangle(extent, n_points_per_side=50):
    """
    Crée un rectangle en coordonnées géographiques avec plus de points par côté
    pour éviter les déformations lors de la projection.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    
    top = np.column_stack([np.linspace(lon_min, lon_max, n_points_per_side), 
                           np.full(n_points_per_side, lat_max)])
    right = np.column_stack([np.full(n_points_per_side, lon_max), 
                             np.linspace(lat_max, lat_min, n_points_per_side)])
    bottom = np.column_stack([np.linspace(lon_max, lon_min, n_points_per_side), 
                              np.full(n_points_per_side, lat_min)])
    left = np.column_stack([np.full(n_points_per_side, lon_min), 
                            np.linspace(lat_min, lat_max, n_points_per_side)])
    
    coords = np.vstack([top, right, bottom, left, top[0:1]])
    return coords[:,0], coords[:,1]


def z_masked_overlap(axe, X, Y, Z, source_projection=None):
    """Masque les cellules qui "wrap" (sauts de longitude)."""
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

        if (to_mask.shape[0] == Z.shape[0] - 1) and (to_mask.shape[1] == Z.shape[1] - 1):
            ext = np.zeros_like(Z, dtype=bool)
            ext[:-1, :-1] = to_mask
            ext[-1, :] = ext[-2, :]
            ext[:, -1] = ext[:, -2]
            to_mask = ext

        Zm = np.ma.masked_where(to_mask, Z)
        return ptx, pty, Zm


def masked_pcolormesh(ax, lon2d, lat2d, data2d, **kwargs):
    """Pcolormesh géodésique avec masque wrap."""
    X, Y, Zm = z_masked_overlap(ax, lon2d, lat2d, data2d, source_projection=ccrs.Geodetic())
    return ax.pcolormesh(X, Y, Zm, transform=ax.projection, shading="auto", **kwargs)


def tight_lonlat_extent(lon2d, lat2d, margin=0.0):
    """Calcule l'extent serré d'un (lon,lat) 2D."""
    lon_min = np.nanmin(lon2d)
    lon_max = np.nanmax(lon2d)
    lat_min = np.nanmin(lat2d)
    lat_max = np.nanmax(lat2d)
    dl = (lon_max - lon_min) * margin
    dphi = (lat_max - lat_min) * margin
    return [lon_min - dl, lon_max + dl, lat_min - dphi, lat_max + dphi]


def denormalize_minmax(norm_data, min_val, max_val):
    """Dénormalise les données normalisées en min-max."""
    return norm_data * (max_val - min_val) + min_val

class XrDatasetMultiRes_simplify(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    
    def __init__(self, paths, split, multires, build_batch, input_vars=None):
        'Initialization'
        self.multires = multires
        self.input_vars = input_vars  # Store which variables are available
        self.db = {}
        self.plot_frequency = 20 
        self.debug_dir = "./debug_plots"

        for res in self.multires:
            self.db[f"patch_x{res}"] = xr.open_dataset(paths[f"patch_x{res}"]).isel(record=split)
        
        self.build_batch = build_batch

    def __len__(self):
        'Denotes the total number of samples'
        res_min = self.multires[-1]
        # Use first available variable instead of hardcoded 'asip_sic'
        first_var = list(self.db[f"patch_x{res_min}"].data_vars)[0]
        return len(self.db[f"patch_x{res_min}"][first_var])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem22__(self, idx):
        out = {}
        for res in self.multires:
            item = self.db[f"patch_x{res}"].isel(record=idx, sample=0)
            
            # Only select variables that exist in TrainingItem
            available_fields = [f for f in TrainingItem._fields if f in item.data_vars or f in item.coords]
            item = item[available_fields]
            
            var_dict = {var: item[var].values for var in item.data_vars}
            var_dict["time"] = item.time.data
            var_dict["xc"] = item.xc.data
            var_dict["yc"] = item.yc.data
            
            item = self.build_batch(var_dict)
            out[f"patch_x{res}"] = item
        
        return out

    def _plot_multires_overlay(self, items_by_res, idx, var_to_plot="asip_sic"):
        """
        Plot multi-resolution overlay with domain boundaries (adapted from plot_CROSCIM_multires).
        
        Args:
            items_by_res: Dict of {res: xr.Dataset} (raw items from netcdf)
            idx: Sample index
            var_to_plot: Variable to plot (default: "asip_sic")
        """
        try:
            # Get finest resolution as reference
            finest_res = self.multires[-1]
            
            # Find first available variable if var_to_plot doesn't exist
            if var_to_plot not in items_by_res[finest_res]:
                for var in items_by_res[finest_res].data_vars:
                    if var not in ['time', 'xc', 'yc', 'lat', 'lon', 'land_mask']:
                        var_to_plot = var
                        break
            
            if var_to_plot not in items_by_res[finest_res]:
                print("⚠️  No plottable variable found")
                return
            
            # Determine time index (middle timestep)
            hr_item = items_by_res[finest_res]
            if "time" in hr_item[var_to_plot].dims:
                time_data = hr_item[var_to_plot].values
                t_idx = time_data.shape[0] // 2
            else:
                t_idx = 0
            
            # Create figure with polar projection
            proj = ccrs.NorthPolarStereo()
            fig, ax = plt.subplots(1, 1, subplot_kw={"projection": proj}, figsize=(12, 10))
            
            # Map features
            ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
            ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, zorder=2)
            
            # Plot finest resolution data
            ds_finest = items_by_res[finest_res]
            lon_finest = ds_finest["lon"].values
            lat_finest = ds_finest["lat"].values
            lon_finest = denormalize_minmax(lon_finest, -180, 180)
            lat_finest = denormalize_minmax(lat_finest, 50, 90)
            
            if "time" in ds_finest[var_to_plot].dims:
                data_finest = ds_finest[var_to_plot].isel(time=t_idx).values
            else:
                data_finest = ds_finest[var_to_plot].values
            
            # Plot main data
            im = masked_pcolormesh(ax, lon_finest, lat_finest, data_finest, 
                                  cmap='viridis', alpha=0.8, zorder=3)
            
            # Set extent to coarsest resolution
            coarsest_res = self.multires[0]
            lon_coarse = items_by_res[coarsest_res]["lon"].values
            lat_coarse = items_by_res[coarsest_res]["lat"].values
            lon_coarse = denormalize_minmax(lon_coarse, -180, 180)
            lat_coarse = denormalize_minmax(lat_coarse, 50, 90)
            
            global_extent = tight_lonlat_extent(lon_coarse, lat_coarse, margin=0.05)
            ax.set_extent(global_extent, crs=ccrs.PlateCarree())
            
            # Draw boundaries for each resolution
            colors = ['red', 'orange', 'cyan', 'magenta', 'lime']
            linewidths = [3.0, 2.5, 2.0, 1.5, 1.5]
            legend_elements = []
            
            for i, res in enumerate(self.multires):
                ds = items_by_res[res]
                lon = ds["lon"].values
                lat = ds["lat"].values
                lon = denormalize_minmax(lon, -180, 180)
                lat = denormalize_minmax(lat, 50, 90)
                
                # Create high-res rectangle
                lon_min = float(np.nanmin(lon))
                lon_max = float(np.nanmax(lon))
                lat_min = float(np.nanmin(lat))
                lat_max = float(np.nanmax(lat))
                extent = [lon_min, lon_max, lat_min, lat_max]
                
                rect_lon, rect_lat = highres_rectangle(extent, n_points_per_side=100)
                
                # Draw boundary
                line = ax.plot(rect_lon, rect_lat, 
                              transform=ccrs.PlateCarree(),
                              color=colors[i % len(colors)], 
                              linewidth=linewidths[i % len(linewidths)],
                              label=f'x{res} domain',
                              zorder=10 + i)
                
                legend_elements.append(line[0])
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation='vertical')
            cbar.set_label(var_to_plot, fontsize=12)
            
            # Legend
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                     framealpha=0.9, title='Resolution domains')
            
            # Title
            title = (f'Multi-Resolution Domains Overlay\n'
                    f'{var_to_plot} | Sample {idx} | Time {t_idx}')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Save
            os.makedirs(self.debug_dir, exist_ok=True)
            filename = f'multires_overlay_{idx:04d}.png'
            filepath = os.path.join(self.debug_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ Saved overlay: {filename}")
            
        except Exception as e:
            print(f"⚠️  Failed to plot overlay for sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    def _plot_multiresolution_grid(self, items_by_res, idx):
        """
        Plot data side-by-side for each resolution with nested domain boundaries.
        Shows boundaries of finer resolutions overlaid on coarser resolution plots.
        
        Args:
            items_by_res: Dict of {res: xr.Dataset}
            idx: Sample index
        """
        try:
            # Find first available variable
            first_var = None
            for var in items_by_res[self.multires[-1]].data_vars:
                if var not in ['time', 'xc', 'yc', 'lat', 'lon', 'land_mask']:
                    first_var = var
                    break
            
            if first_var is None:
                return
            
            # Number of resolutions
            n_res = len(self.multires)
            
            # Create figure with Cartopy projection for each subplot
            proj = ccrs.NorthPolarStereo()
            fig, axes = plt.subplots(1, n_res, 
                                    subplot_kw={"projection": proj},
                                    figsize=(6*n_res, 6))
            if n_res == 1:
                axes = [axes]
            
            # Collect data and find common colorscale
            all_data = []
            t_idx = None
            for res in self.multires:
                item = items_by_res[res]
                if first_var in item:
                    data = item[first_var].values
                    if data.ndim == 3:
                        if t_idx is None:
                            t_idx = data.shape[0] // 2
                        data = data[t_idx]
                    all_data.append(data)
            
            vmin = np.nanpercentile(np.concatenate([d.flatten() for d in all_data]), 2)
            vmax = np.nanpercentile(np.concatenate([d.flatten() for d in all_data]), 98)
            
            # Colors for nested boundaries
            boundary_colors = {
                self.multires[0]: 'red',      # x50 -> red
                self.multires[1]: 'orange',   # x10 -> orange  
                self.multires[2]: 'cyan'      # x2 -> cyan
            }
            
            # Plot each resolution
            for i, (res, ax) in enumerate(zip(self.multires, axes)):
                item = items_by_res[res]
                
                if first_var in item:
                    # Get coordinates
                    lon = item["lon"].values
                    lat = item["lat"].values
                    lon = denormalize_minmax(lon, -180, 180)
                    lat = denormalize_minmax(lat, 50, 90)
                    
                    # Get data
                    data = item[first_var].values
                    if data.ndim == 3:
                        data = data[t_idx]
                    
                    # Add map features
                    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
                    ax.gridlines(draw_labels=False, x_inline=False, y_inline=False, zorder=2)
                    
                    # Plot data with masked_pcolormesh
                    im = masked_pcolormesh(ax, lon, lat, data,
                                        cmap='viridis', 
                                        vmin=vmin, vmax=vmax,
                                        alpha=0.8, zorder=3)
                    
                    # Set extent to current resolution
                    extent = tight_lonlat_extent(lon, lat, margin=0.02)
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    
                    # ✅ Draw boundaries of finer resolutions (nested domains)
                    legend_elements = []
                    for j in range(i, n_res):
                        finer_res = self.multires[j]
                        finer_item = items_by_res[finer_res]
                        
                        finer_lon = finer_item["lon"].values
                        finer_lat = finer_item["lat"].values
                        finer_lon = denormalize_minmax(finer_lon, -180, 180)
                        finer_lat = denormalize_minmax(finer_lat, 50, 90)
                        
                        # Create high-res rectangle for boundary
                        lon_min = float(np.nanmin(finer_lon))
                        lon_max = float(np.nanmax(finer_lon))
                        lat_min = float(np.nanmin(finer_lat))
                        lat_max = float(np.nanmax(finer_lat))
                        extent_rect = [lon_min, lon_max, lat_min, lat_max]
                        
                        rect_lon, rect_lat = highres_rectangle(extent_rect, n_points_per_side=100)
                        
                        # Line style: solid for current res, dashed for finer
                        linestyle = '-' if j == i else '--'
                        linewidth = 3.0 if j == i else 2.0
                        
                        # Draw boundary
                        line = ax.plot(rect_lon, rect_lat,
                                    transform=ccrs.PlateCarree(),
                                    color=boundary_colors.get(finer_res, 'white'),
                                    linewidth=linewidth,
                                    linestyle=linestyle,
                                    label=f'x{finer_res} domain',
                                    zorder=10 + j)
                        
                        legend_elements.append(line[0])
                    
                    # Title with resolution info
                    ax.set_title(f'x{res} Resolution\n{data.shape[0]}×{data.shape[1]} pixels',
                            fontsize=12, fontweight='bold')
                    
                    # Add legend for nested domains
                    if legend_elements:
                        ax.legend(handles=legend_elements, 
                                loc='upper right', 
                                fontsize=9,
                                framealpha=0.9,
                                title='Domains')
            
            # Colorbar
            fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label=first_var)
            
            # Title
            fig.suptitle(
                f'Multi-Resolution Grid with Nested Domains\n'
                f'Sample {idx} | Timestep {t_idx if t_idx else 0} | Variable: {first_var}',
                fontsize=14, fontweight='bold'
            )
            
            plt.tight_layout()
            
            # Save
            os.makedirs(self.debug_dir, exist_ok=True)
            filename = f'multires_grid_{idx:04d}.png'
            filepath = os.path.join(self.debug_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ Saved grid: {filename}")
            
        except Exception as e:
            print(f"⚠️  Failed to plot grid for sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    def __getitem__(self, idx):
        out = {}
        items_by_res = {}  # Store raw xr.Dataset for plotting
        
        for res in self.multires:
            # Get raw item from netcdf
            item = self.db[f"patch_x{res}"].isel(record=idx, sample=0)
            
            # Store for plotting
            items_by_res[res] = item
            
            # Only select variables that exist in TrainingItem
            available_fields = [f for f in TrainingItem._fields if f in item.data_vars or f in item.coords]
            item_filtered = item[available_fields]
            
            var_dict = {var: item_filtered[var].values for var in item_filtered.data_vars}
            var_dict["time"] = item_filtered.time.data
            var_dict["xc"] = item_filtered.xc.data
            var_dict["yc"] = item_filtered.yc.data
            
            item_processed = self.build_batch(var_dict)
            out[f"patch_x{res}"] = item_processed
        
        # ✅ Generate plots if enabled
        if idx % self.plot_frequency == 0:
            # Plot 1: Overlay with domain boundaries (like plot_CROSCIM_multires)
            self._plot_multires_overlay(items_by_res, idx, var_to_plot="asip_sic")
            
            # Plot 2: Grid comparison
            self._plot_multiresolution_grid(items_by_res, idx)
        
        return out

class BaseDataModuleMultiRes_simplify(pl.LightningDataModule):
    def __init__(self, 
                 croscim_preproc_paths,
                 multires,
                 split_train,
                 split_val,
                 split_test,
                 norm_stats,
                 norm_stats_covs,
                 satellite_vars=None,  # NEW: From config
                 covariates=None,      # NEW: From config
                 target_vars=None,     # NEW: From config
                 **kwargs):

        super().__init__()
        self.croscim_preproc_paths = croscim_preproc_paths
        self.multires = multires
        self.split_train = split_train
        self.split_val = split_val
        self.split_test = split_test
        self._norm_stats = norm_stats
        self._norm_stats_covs = norm_stats_covs
        
        # Store variable configuration
        self.satellite_vars = satellite_vars or DEFAULT_VAR_GROUPS
        self.covariates = covariates or DEFAULT_COVARIATES
        self.target_vars = target_vars or ["tgt_sic", "tgt_SIT"]
        
        # Construct input_vars automatically
        self.input_vars = self._construct_input_vars()
        
        # Determine which sources are actually needed
        self.active_sources = [src for src, vars in self.satellite_vars.items() if vars]
        
        print(f"\n{'='*60}")
        print(f"DataModule configuration:")
        print(f"{'='*60}")
        print(f"  Satellite vars: {self.satellite_vars}")
        print(f"  Active sources: {self.active_sources}")
        print(f"  Covariates: {self.covariates}")
        print(f"  Target vars: {self.target_vars}")
        print(f"  Input vars: {self.input_vars}")
        print(f"  Multires: {self.multires}")
        print(f"{'='*60}\n")

    def _construct_input_vars(self):
        """Construct input variable names from satellite_vars + covariates"""
        input_vars = []
        
        # Add satellite variables with source prefix
        for source, vars in self.satellite_vars.items():
            for var in vars:
                input_vars.append(f"{source}_{var}")
        
        # Add covariates
        if self.covariates:
            input_vars.extend(self.covariates)
        
        return input_vars

    @property
    def norm_stats(self):
        return self._norm_stats

    @property
    def norm_stats_covs(self):
        return self._norm_stats_covs

    def build_batch(self, item_dict):
        """
        Build batch from item_dict, handling only available fields.
        """
        # Extract only the fields defined in TrainingItem that are present in item_dict
        fields = {k: v for k, v in item_dict.items() if k in TrainingItem._fields}
        return TrainingItem(**fields)

    def setup(self, stage='test'):
        """
        Setup datasets for train/val/test.
        Validates that preprocessed data contains the required variables.
        """
        build_batch = self.build_batch
        
        # Validate that preprocessed files contain required variables
        self._validate_preprocessed_data()
        
        self.train_ds = XrDatasetMultiRes_simplify(
            self.croscim_preproc_paths, 
            self.split_train, 
            self.multires, 
            build_batch,
            input_vars=self.input_vars
        )
        
        self.val_ds = XrDatasetMultiRes_simplify(
            self.croscim_preproc_paths,
            self.split_val, 
            self.multires, 
            build_batch,
            input_vars=self.input_vars
        )
        
        self.test_ds = XrDatasetMultiRes_simplify(
            self.croscim_preproc_paths,
            self.split_test, 
            self.multires, 
            build_batch,
            input_vars=self.input_vars
        )
        
        print(f"Datasets ready:")
        print(f"  Train: {len(self.train_ds)} samples")
        print(f"  Val: {len(self.val_ds)} samples")
        print(f"  Test: {len(self.test_ds)} samples")

    def _validate_preprocessed_data(self):
        """
        Validate that preprocessed data files contain the required variables.
        """
        for res in self.multires:
            path_key = f"patch_x{res}"
            if path_key not in self.croscim_preproc_paths:
                raise ValueError(f"Missing path for {path_key} in croscim_preproc_paths")
            
            # Open one file to check variables
            try:
                ds = xr.open_dataset(self.croscim_preproc_paths[path_key])
                available_vars = list(ds.data_vars)
                
                # Check if all required input vars are present
                missing_vars = []
                for var in self.input_vars:
                    if var not in available_vars:
                        missing_vars.append(var)
                
                # Check if all target vars are present
                for var in self.target_vars:
                    if var not in available_vars:
                        missing_vars.append(var)
                
                if missing_vars:
                    print(f"\n Warning for {path_key}:")
                    print(f"  Missing variables: {missing_vars}")
                    print(f"  Available variables: {available_vars}")
                    print(f"  This may cause errors during training!")
                else:
                    print(f"{path_key}: All required variables present")
                
                ds.close()
                
            except Exception as e:
                print(f"❌ Error validating {path_key}: {e}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, 
            shuffle=True, 
            batch_size=2,
            num_workers=5, 
            persistent_workers=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, 
            shuffle=False, 
            batch_size=2,
            num_workers=5, 
            persistent_workers=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, 
            shuffle=False, 
            batch_size=2,
            num_workers=5, 
            persistent_workers=True
        )