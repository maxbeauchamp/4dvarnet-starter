"""
Extract and list all files required by load_mfdata for a specific date range.
"""
import sys
sys.path.append('../..')

from contrib.CROSCIM.dataloaders.load_data import get_paths_for_source, DEFAULT_VAR_GROUPS, DEFAULT_COVARIATES
from glob import glob
import datetime
import numpy as np
import shutil
from pathlib import Path


def select_paths_from_dates(files, times, fmt="%Y%m%d"):
    """
    Select file paths matching the given time range(s).
    This is the same logic as in load_mfdata.
    """
    if isinstance(times, list):
        dates = []
        for t in times:
            start = datetime.datetime.strptime(t.start, "%Y-%m-%d")
            end = datetime.datetime.strptime(t.stop, "%Y-%m-%d")
            dates.extend([(start + datetime.timedelta(days=x)).strftime(fmt) 
                         for x in range((end-start).days)])
    elif isinstance(times, slice):
        start = datetime.datetime.strptime(times.start, "%Y-%m-%d")
        end = datetime.datetime.strptime(times.stop, "%Y-%m-%d")
        dates = [(start + datetime.timedelta(days=x)).strftime(fmt) 
                for x in range((end-start).days)]
    else:
        raise ValueError(f"Unsupported times type: {type(times)}")
    
    return np.sort([f for f in files if any(s in f for s in dates)])


def extract_files_for_dates(start_date, end_date, output_dir=None, copy_files=False):
    """
    Extract list of required files for a date range using the same logic as load_mfdata.
    
    Args:
        start_date: str, e.g., '2022-02-01'
        end_date: str, e.g., '2022-02-15'
        output_dir: Optional path to copy files to
        copy_files: If True, copy files to output_dir
    
    Returns:
        dict with {source: [file_paths]}
    """
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING FILES FOR {start_date} to {end_date}")
    print("="*70)
    
    # Create time slice (same format as load_mfdata expects)
    times = slice(start_date, end_date)
    
    # Use default variable configuration
    satellite_vars = DEFAULT_VAR_GROUPS
    covariates = DEFAULT_COVARIATES
    
    # Path loaders for each source (same as in load_mfdata)
    path_loaders = {
        "asip": lambda: glob('/dmidata/users/maxb/ASIP_OSISAF_dataset/ASIP_L3/*nc'),
        "cimr": lambda: glob('/dmidata/users/maxb/CROSCIM_dataset/out_CIMR/CIMR5km_*nc'),
        "cristal": lambda: glob('/dmidata/users/maxb/CROSCIM_dataset/out_CRISTAL/CRISTAL5km_*nc'),
        "models": lambda: glob('/dmidata/users/maxb/CROSCIM_dataset/out_MOD/MOD5km_*nc'),
    }
    
    # Date format for each source (same as in load_mfdata)
    date_formats = {
        "asip": "%Y%m%d",
        "cimr": "%Y-%m-%d",
        "cristal": "%Y-%m-%d",
        "models": "%Y-%m-%d",
    }
    
    required_files = {}
    
    # ✅ Extract satellite file paths (same logic as load_mfdata)
    for source, vars_list in satellite_vars.items():
        if not vars_list:  # Skip if empty list
            print(f"\n{source.upper()}: Skipped (no variables configured)")
            continue
        
        print(f"\n{source.upper()}: Variables = {vars_list}")
        
        # Get all paths for this source
        all_paths = path_loaders[source]()
        print(f"  Total files available: {len(all_paths)}")
        
        # Select paths matching date range
        selected_paths = select_paths_from_dates(all_paths, times, fmt=date_formats[source])
        required_files[source] = list(selected_paths)
        
        print(f"  Files matching date range: {len(selected_paths)}")
        
        if len(selected_paths) > 0:
            print(f"  First file: {Path(selected_paths[0]).name}")
            print(f"  Last file:  {Path(selected_paths[-1]).name}")
        else:
            print(f"  ⚠️  No files found!")
    
    # ✅ Extract covariate file paths
    if covariates:
        print(f"\nCOVARIATES: Variables = {covariates}")
        
        covariates_paths = glob('/dmidata/users/maxb/CROSCIM_dataset/atm_data/atm5km_*.nc')
        print(f"  Total files available: {len(covariates_paths)}")
        
        selected_cov_paths = select_paths_from_dates(covariates_paths, times, fmt="%Y-%m-%d")
        required_files['covariates'] = list(selected_cov_paths)
        
        print(f"  Files matching date range: {len(selected_cov_paths)}")
        
        if len(selected_cov_paths) > 0:
            print(f"  First file: {Path(selected_cov_paths[0]).name}")
            print(f"  Last file:  {Path(selected_cov_paths[-1]).name}")
        else:
            print(f"  ⚠️  No files found!")
    
    # ✅ Extract model output file paths
    print(f"\nMODELS: Output files from out_MOD")
    models_paths = glob('/dmidata/users/maxb/CROSCIM_dataset/out_MOD/MOD5km_*.nc')
    print(f"  Total files available: {len(models_paths)}")
    selected_mod_paths = select_paths_from_dates(models_paths, times, fmt="%Y-%m-%d")
    required_files['models'] = list(selected_mod_paths)
    print(f"  Files matching date range: {len(selected_mod_paths)}")
    if len(selected_mod_paths) > 0:
        print(f"  First file: {Path(selected_mod_paths[0]).name}")
        print(f"  Last file:  {Path(selected_mod_paths[-1]).name}")
    else:
        print(f"  ⚠️  No files found!")
    
    # Print summary
    print("\n" + "="*70)
    print("REQUIRED FILES SUMMARY")
    print("="*70)
    total_files = 0
    for source, files in required_files.items():
        print(f"\n{source.upper()}: {len(files)} files")
        total_files += len(files)
        if len(files) > 0:
            for f in files[:3]:  # Show first 3
                print(f"  - {Path(f).name}")
            if len(files) > 3:
                print(f"  ... and {len(files)-3} more")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {total_files} files")
    print("="*70)
    
    # Copy files if requested
    if copy_files and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"COPYING FILES TO {output_dir}")
        print("="*70)
        
        total_copied = 0
        total_skipped = 0
        
        for source, files in required_files.items():
            if not files:
                print(f"\n{source.upper()}: No files to copy")
                continue
            
            source_dir = output_path / source
            source_dir.mkdir(exist_ok=True)
            
            print(f"\n{source.upper()}: Copying {len(files)} files...")
            for i, src_file in enumerate(files):
                src_path = Path(src_file)
                dst_file = source_dir / src_path.name
                
                if not dst_file.exists():
                    try:
                        shutil.copy2(src_file, dst_file)
                        if i < 3 or i == len(files) - 1:  # Print first few and last
                            print(f"  ✓ Copied {i+1}/{len(files)}: {src_path.name}")
                        total_copied += 1
                    except Exception as e:
                        print(f"  ✗ Failed {i+1}/{len(files)}: {src_path.name} - {e}")
                else:
                    if i < 3 or i == len(files) - 1:  # Print first few and last
                        print(f"  - Skipped {i+1}/{len(files)}: {src_path.name} (exists)")
                    total_skipped += 1
        
        print(f"\n{'='*70}")
        print(f"✅ Copy complete! Copied: {total_copied}, Skipped: {total_skipped}")
        print("="*70)
    
    return required_files


def save_file_list(required_files, output_file="required_files.txt"):
    """Save file list to text file."""
    with open(output_file, 'w') as f:
        f.write(f"# Generated on {datetime.datetime.now()}\n")
        total = sum(len(files) for files in required_files.values())
        f.write(f"# Total files: {total}\n\n")
        
        for source, files in required_files.items():
            f.write(f"# {source.upper()} ({len(files)} files)\n")
            for file_path in files:
                f.write(f"{file_path}\n")
            f.write("\n")
    
    total_files = sum(len(files) for files in required_files.values())
    print(f"\n✅ File list saved to {output_file} ({total_files} total files)")


if __name__ == "__main__":
    # ✅ Configure your date range here
    START_DATE = "2022-02-01"
    END_DATE = "2022-02-16"
    
    # Extract required files (no DataModule instantiation needed!)
    required_files = extract_files_for_dates(
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir=None,  # Set to a path to copy files
        copy_files=False   # Set to True to actually copy files
    )
    
    # Save list to file
    save_file_list(required_files, f"required_files_{START_DATE}_{END_DATE}.txt")
    
    # Create extraction directory and copy files
    output_dir = f"/dmidata/users/maxb/extract_inference_{START_DATE}_{END_DATE}"
    output_path = Path(output_dir)
    
    print(f"\n{'='*70}")
    print(f"CREATING EXTRACTION DIRECTORY")
    print("="*70)
    print(f"Target directory: {output_dir}")
    
    # Create main directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created main directory")
    
    # Create subdirectories for each source
    subdirs = {}
    for source in required_files.keys():
        source_dir = output_path / source
        source_dir.mkdir(exist_ok=True)
        subdirs[source] = source_dir
        print(f" Created subdirectory: {source}/")
    
    # Copy files
    print(f"\n{'='*70}")
    print(f"COPYING FILES")
    print("="*70)
    
    total_copied = 0
    total_skipped = 0
    total_failed = 0
    
    for source, files in required_files.items():
        if not files:
            print(f"\n{source.upper()}: No files to copy")
            continue
        
        print(f"\n{source.upper()}: Processing {len(files)} files...")
        
        for i, src_file in enumerate(files, 1):
            src_path = Path(src_file)
            dst_file = subdirs[source] / src_path.name
            
            if not dst_file.exists():
                try:
                    shutil.copy2(src_file, dst_file)
                    if i <= 3 or i == len(files):  # Show first 3 and last
                        print(f"  ✓ [{i:3d}/{len(files)}] Copied: {src_path.name}")
                    elif i == 4:
                        print(f"  ... copying remaining files ...")
                    total_copied += 1
                except Exception as e:
                    print(f"  ✗ [{i:3d}/{len(files)}] Failed: {src_path.name}")
                    print(f"      Error: {e}")
                    total_failed += 1
            else:
                if i <= 3:  # Show first 3 skipped
                    print(f"  - [{i:3d}/{len(files)}] Skipped: {src_path.name} (exists)")
                total_skipped += 1
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"\nStatistics:")
    print(f"  ✓ Copied:  {total_copied} files")
    print(f"  - Skipped: {total_skipped} files (already existed)")
    print(f"  ✗ Failed:  {total_failed} files")
    print(f"  → Total:   {total_copied + total_skipped + total_failed} files")
    
    # Show directory structure
    print(f"\nDirectory structure:")
    for source, files in required_files.items():
        print(f"  {output_dir}/{source}/")
        print(f"    └─ {len(files)} files")
    
    print(f"\n{'='*70}")
    print(f" All done! Files extracted to:")
    print(f"   {output_dir}")
    print("="*70)
