#!/usr/bin/env python3
"""
Create MOD5km files by merging CRISTAL and CIMR data.
Uses parallel processing for maximum speed.
"""

import os
import xarray as xr
from glob import glob
from joblib import Parallel, delayed
from pathlib import Path
import datetime

# Paths
CRISTAL_DIR = "/Odyssey/public/CROSCIM_dataset/out_CRISTAL"
CIMR_DIR = "/Odyssey/public/CROSCIM_dataset/data_noise"
OUTPUT_DIR = "/Odyssey/public/CROSCIM_dataset/out_MOD"

# Variables to extract from each source
CRISTAL_VARS = ["HS_model", "SSH_model", "SIT_model"]
CIMR_VARS = ["SIC"]
COORDS = ["lat", "lon", "xc", "yc", "time"]

def extract_date_from_filename(filename):
    """Extract date string from CRISTAL5km_no_N_YYYY-MM-DD.nc format."""
    basename = os.path.basename(filename)
    # Extract YYYY-MM-DD from CRISTAL5km_no_N_YYYY-MM-DD.nc
    # Split by '_' and take the last part before .nc
    parts = basename.replace(".nc", "").split("_")
    date_str = parts[-1]  # Last part is the date
    return date_str

def process_single_date(cristal_file):
    """Process a single date: merge CRISTAL and CIMR data."""
    try:
        date_str = extract_date_from_filename(cristal_file)
        
        print(f"  Processing date: {date_str}")
        
        # Find corresponding CIMR file
        cimr_file = os.path.join(CIMR_DIR, f"CIMR5km_{date_str}_mod.nc")
        
        if not os.path.exists(cimr_file):
            print(f"  ⚠ Skipping {date_str}: CIMR file not found at {cimr_file}")
            return None
        
        # Open datasets (only read needed variables)
        ds_cristal = xr.open_dataset(cristal_file)
        ds_cimr = xr.open_dataset(cimr_file)
        
        # Extract coordinates (from CRISTAL)
        coords_data = {coord: ds_cristal[coord] for coord in COORDS if coord in ds_cristal}
        
        # Extract variables
        data_vars = {}
        
        # From CRISTAL - remove _model suffix when storing
        for var in CRISTAL_VARS:
            if var in ds_cristal:
                # Remove _model suffix for output
                var_name = var.replace("_model", "")
                data_vars[var_name] = ds_cristal[var]
        
        # From CIMR
        for var in CIMR_VARS:
            if var in ds_cimr:
                data_vars[var] = ds_cimr[var]
        
        # Create merged dataset
        ds_merged = xr.Dataset(data_vars, coords=coords_data)
        
        # Output file
        output_file = os.path.join(OUTPUT_DIR, f"MOD5km_{date_str}.nc")
        
        # Save with compression for faster I/O
        encoding = {var: {"zlib": True, "complevel": 4} for var in ds_merged.data_vars}
        ds_merged.to_netcdf(output_file, encoding=encoding)
        
        # Close datasets
        ds_cristal.close()
        ds_cimr.close()
        ds_merged.close()
        
        return date_str
        
    except Exception as e:
        print(f"  ✗ Error processing {cristal_file}: {e}")
        return None

def main():
    print("="*70)
    print("Creating MOD5km files from CRISTAL + CIMR data")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Get all CIMR files (base reference)
    cimr_files = sorted(glob(os.path.join(CIMR_DIR, "CIMR5km_*.nc")))
    print(f"\nFound {len(cimr_files)} CIMR files")
    
    if len(cimr_files) == 0:
        print("No CIMR files found! Check path.")
        return
    
    # Extract dates from CIMR files
    dates = []
    for cimr_file in cimr_files:
        basename = os.path.basename(cimr_file)
        date_str = basename.replace("CIMR5km_", "").replace("_mod.nc", "")
        dates.append(date_str)
    
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Build list of CRISTAL files to process (based on CIMR dates)
    cristal_files_to_process = []
    for date_str in dates:
        # CRISTAL files have format: CRISTAL5km_no_N_YYYY-MM-DD.nc
        # We need to find the file with this date (no_N can vary)
        cristal_pattern = os.path.join(CRISTAL_DIR, f"CRISTAL5km_no_N_{date_str}.nc")
        print(cristal_pattern)
        matching_cristal = glob(cristal_pattern)
        
        if len(matching_cristal) > 0:
            cristal_files_to_process.append(matching_cristal[0])  # Take first match
        else:
            print(f"  ⚠ Warning: CRISTAL file missing for {date_str}")
    
    print(f"\nWill process {len(cristal_files_to_process)} dates (where both CRISTAL and CIMR exist)")
    
    if len(cristal_files_to_process) == 0:
        print("No matching files found!")
        return
    
    print(f"\nProcessing with parallel jobs...")
    start_time = datetime.datetime.now()
    
    # Parallel processing (use all available cores)
    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(process_single_date)(cristal_file) 
        for cristal_file in cristal_files_to_process
    )
    
    # Count successes
    successful = [r for r in results if r is not None]
    
    end_time = datetime.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print(f"✓ Completed: {len(successful)}/{len(cristal_files_to_process)} files")
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    print(f"  Speed: {len(successful)/elapsed:.2f} files/second")
    print("="*70)

if __name__ == "__main__":
    main()
