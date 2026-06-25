"""
Extract and list all files required by load_mfdata for a specific date range.
"""
import sys
from pathlib import Path
# Repo root (4dvarnet-starter) is 3 levels up: scripts -> CROSCIM -> contrib -> root
sys.path.append(str(Path(__file__).resolve().parents[3]))

from contrib.CROSCIM.dataloaders.load_data import get_paths_for_source, DEFAULT_VAR_GROUPS, DEFAULT_COVARIATES
from glob import glob
import datetime
import numpy as np
import shutil
from pathlib import Path


# ── Source configuration (shared by extraction and completeness checks) ───────
SOURCE_GLOBS = {
    "asip":       '/dmidata/users/maxb/ASIP_OSISAF_dataset/ASIP_L3/*nc',
    "cimr":       '/dmidata/users/maxb/CROSCIM_dataset/out_CIMR/CIMR5km_*nc',
    "cristal":    '/dmidata/users/maxb/CROSCIM_dataset/out_CRISTAL/CRISTAL5km_*nc',
    "models":     '/dmidata/users/maxb/CROSCIM_dataset/out_MOD/MOD5km_*nc',
    "covariates": '/dmidata/users/maxb/CROSCIM_dataset/atm_data/atm5km_*.nc',
}
SOURCE_DATE_FORMATS = {
    "asip": "%Y%m%d",
    "cimr": "%Y-%m-%d",
    "cristal": "%Y-%m-%d",
    "models": "%Y-%m-%d",
    "covariates": "%Y-%m-%d",
}


def required_sources():
    """Sources that must each provide one file per day (defines the 'full' count)."""
    sources = [s for s, vars_list in DEFAULT_VAR_GROUPS.items() if vars_list]
    if DEFAULT_COVARIATES:
        sources.append("covariates")
    sources.append("models")
    return sources


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
    
    # Path loaders / date formats for each source (module-level config)
    path_loaders = {s: (lambda s=s: glob(SOURCE_GLOBS[s])) for s in SOURCE_GLOBS}
    date_formats = SOURCE_DATE_FORMATS

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
        
        covariates_paths = glob(SOURCE_GLOBS["covariates"])
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
    models_paths = glob(SOURCE_GLOBS["models"])
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


def count_window_files(start_date, seq_length_days, sources, files_cache):
    """
    Count how many files each source provides for a window starting at start_date.

    Uses the same date-matching logic as the extraction (select_paths_from_dates),
    so a "full" window has seq_length_days files per source.
    """
    end_date = (datetime.datetime.strptime(start_date, "%Y-%m-%d")
                + datetime.timedelta(days=seq_length_days)).strftime("%Y-%m-%d")
    times = slice(start_date, end_date)
    return {
        s: len(select_paths_from_dates(files_cache[s], times, fmt=SOURCE_DATE_FORMATS[s]))
        for s in sources
    }


def find_complete_window(start_date, seq_length_days, sources, files_cache,
                         taken, max_shift_days=180):
    """
    From start_date, walk forward day by day to the first window that is both
    complete (every source has its full seq_length_days files) and not already
    assigned to another sequence (its start is not in `taken`).

    Returns (new_start, new_end, shift_days, counts) or None if none found within
    max_shift_days.
    """
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    for shift in range(max_shift_days + 1):
        cand = (start + datetime.timedelta(days=shift)).strftime("%Y-%m-%d")
        if cand in taken:
            continue
        counts = count_window_files(cand, seq_length_days, sources, files_cache)
        if all(counts[s] >= seq_length_days for s in sources):
            end = (datetime.datetime.strptime(cand, "%Y-%m-%d")
                   + datetime.timedelta(days=seq_length_days)).strftime("%Y-%m-%d")
            return cand, end, shift, counts
    return None


def validate_and_adjust_sequences(sequences, seq_length_days, max_shift_days=180):
    """
    For each sequence, check the window is complete (seq_length_days files per source).
    If not, shift its start forward to the nearest complete window that is not
    already used by another sequence, so the 25 windows stay distinct.

    Returns the adjusted list of (start_date, end_date) tuples.
    """
    sources = required_sources()
    files_cache = {s: glob(SOURCE_GLOBS[s]) for s in sources}
    expected_total = len(sources) * seq_length_days

    print(f"\n{'='*70}")
    print(f"VALIDATING SEQUENCES (expect {len(sources)} sources × "
          f"{seq_length_days} days = {expected_total} files each)")
    print(f"Sources: {sources}")
    print("="*70)

    adjusted = []
    taken = set()
    for idx, (start_date, end_date) in enumerate(sequences, 1):
        counts = count_window_files(start_date, seq_length_days, sources, files_cache)
        total = sum(counts.values())
        missing = {s: seq_length_days - n for s, n in counts.items() if n < seq_length_days}

        # Find nearest forward window that is complete AND not already taken
        # (shift=0 keeps the original window when it is already complete & free).
        res = find_complete_window(start_date, seq_length_days, sources,
                                   files_cache, taken, max_shift_days)
        if res is None:
            print(f"  {idx:2d}. {start_date} → {end_date}  ⚠️ incomplete ({total}), "
                  f"missing {missing} — NO free complete window within "
                  f"{max_shift_days} days, KEEPING original")
            adjusted.append((start_date, end_date))
            taken.add(start_date)
            continue

        ns, ne, shift, _ = res
        taken.add(ns)
        adjusted.append((ns, ne))
        if shift == 0:
            print(f"  {idx:2d}. {start_date} → {end_date}  ✓ complete ({total})")
        else:
            reason = f"incomplete ({total}), missing {missing}" if missing \
                     else f"complete but window already taken"
            print(f"  {idx:2d}. {start_date} → {end_date}  ⚠️ {reason}")
            print(f"      → shifted +{shift}d to {ns} → {ne} (complete, distinct)")

    return adjusted


def generate_sequences(year, n_sequences, seq_length_days):
    """
    Build n_sequences (start_date, end_date) pairs spread homogeneously over `year`.

    Each sequence spans `seq_length_days` days, with `end_date` exclusive — the same
    convention as select_paths_from_dates (which iterates over range((end-start).days)).
    The start dates are evenly spaced so the last window still ends within the year.

    Returns:
        list of (start_date, end_date) string tuples, e.g. [("2023-01-01", "2023-01-16"), ...]
    """
    year_start = datetime.date(year, 1, 1)
    year_end = datetime.date(year, 12, 31)

    # Latest start so the full window stays within the year (end is exclusive).
    last_start = year_end - datetime.timedelta(days=seq_length_days - 1)
    span_days = (last_start - year_start).days
    if span_days < 0:
        raise ValueError(
            f"seq_length_days={seq_length_days} is too long to fit in year {year}"
        )

    # Evenly spaced start offsets over the available window (homogeneous coverage).
    offsets = np.linspace(0, span_days, n_sequences).round().astype(int)

    sequences = []
    for off in offsets:
        start = year_start + datetime.timedelta(days=int(off))
        end = start + datetime.timedelta(days=seq_length_days)
        sequences.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return sequences


def process_sequences(sequences, base_output_dir, copy_files=True):
    """
    Extract (and optionally copy) the required files for all sequences, pooled by
    source.

    All sequences write into the SAME per-source subdirectories under
    base_output_dir (asip/ cimr/ covariates/ cristal/ models/). Files shared by
    overlapping sequences are copied once (existing files are skipped), so each
    source directory ends up with the de-duplicated union of every sequence's files.
    """
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    merged = {}   # source -> set of file paths (de-duplicated across sequences)
    for idx, (start_date, end_date) in enumerate(sequences, 1):
        print(f"\n{'#'*70}")
        print(f"# SEQUENCE {idx}/{len(sequences)}: {start_date} → {end_date}")
        print(f"# Output (pooled): {base_path}")
        print("#"*70)

        required_files = extract_files_for_dates(
            start_date=start_date,
            end_date=end_date,
            output_dir=str(base_path),     # pool everything into <base>/<source>/
            copy_files=copy_files,
        )
        for source, files in required_files.items():
            merged.setdefault(source, set()).update(files)

    # Single de-duplicated file list at the base directory.
    merged_lists = {source: sorted(files) for source, files in merged.items()}
    save_file_list(merged_lists, base_path / "required_files.txt")

    # ── Pooled summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"ALL {len(sequences)} SEQUENCES PROCESSED — pooled by source")
    print("="*70)
    grand_total = 0
    for source, files in merged_lists.items():
        print(f"  {source}: {len(files)} unique files")
        grand_total += len(files)
    print(f"\n{'='*70}")
    print(f"TOTAL unique files: {grand_total}")
    print(f"Base directory: {base_output_dir}")
    print("="*70)

    return merged_lists


if __name__ == "__main__":
    # ✅ Configure the multi-sequence extraction here
    YEAR = 2022
    N_SEQUENCES = 25
    SEQUENCE_LENGTH_DAYS = 15   # matches the original 2022-02-01 → 2022-02-16 window

    BASE_OUTPUT_DIR = f"/dmidata/users/maxb/extract_inference_{YEAR}_{N_SEQUENCES}seq"

    # Build 25 date windows spread homogeneously over the year.
    sequences = generate_sequences(YEAR, N_SEQUENCES, SEQUENCE_LENGTH_DAYS)

    print(f"\n{'='*70}")
    print(f"PLANNED SEQUENCES ({N_SEQUENCES} × {SEQUENCE_LENGTH_DAYS} days over {YEAR})")
    print("="*70)
    for idx, (start_date, end_date) in enumerate(sequences, 1):
        print(f"  {idx:2d}. {start_date} → {end_date}")

    # Validate completeness and shift incomplete windows forward to the nearest
    # window where every source has its full set of files.
    sequences = validate_and_adjust_sequences(sequences, SEQUENCE_LENGTH_DAYS)

    # Extract and copy files for every sequence.
    process_sequences(sequences, BASE_OUTPUT_DIR, copy_files=True)
