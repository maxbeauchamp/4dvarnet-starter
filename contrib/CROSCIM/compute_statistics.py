import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import random
import xarray as xr
import numpy as np
from glob import glob
from collections import defaultdict

# Nombre de fichiers à échantillonner
N_SAMPLES = 5

# Spécifie les types de normalisation attendus
norm_types = {
    "zscore": lambda arr: {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "type": "zscore"},
    "minmax": lambda arr: {"min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)), "type": "minmax"},
}

# Configuration : dictionnaire des variables à extraire avec leur type de normalisation
VAR_GROUPS = {
    "asip": {
        "sic": "minmax",
        "standard_deviation_sic": "zscore",
        "status_flag": "minmax"
    },
    "cimr": {
        "SIC": "minmax",
        "SIT": "zscore",
        "Tsurf": "zscore",
        "SICnoise": "zscore",
        "SITnoise": "zscore",
        "Tsurfnoise": "zscore"
    },
    "cristal": {
        "HS": "zscore",
        "SIT": "zscore",
        "SSH": "zscore",
        "HSnoise": "zscore",
        "SITnoise": "zscore",
        "SSHnoise": "zscore"
    }
}

COVARIATES = {
    "msl": "zscore",
    "t2m": "zscore",
    "u10": "zscore",
    "v10": "zscore",
    "tcc": "minmax",
    "d2m": "zscore",
    "ssrd": "zscore",
    "strd": "zscore",
    "tp": "zscore"
}


def compute_stats_from_files(file_list, variables, norm_types):
    # Initialisation accumulée pour chaque variable
    accum = {
        var: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": np.inf,
            "max": -np.inf
        } 
        for var in variables
    }

    for f in file_list:
        print(f)
        try:
            ds = xr.open_dataset(f, chunks={"time": 1})
            for var in variables:
                if var in ds:
                    data = ds[var].values
                    # Conversion en 1D + suppression des NaN
                    arr = data.flatten()
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        accum[var]["count"] += arr.size
                        accum[var]["sum"] += arr.sum()
                        accum[var]["sum_sq"] += (arr ** 2).sum()
                        accum[var]["min"] = min(accum[var]["min"], arr.min())
                        accum[var]["max"] = max(accum[var]["max"], arr.max())
            ds.close()
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
            continue

    # Calcul des stats finales
    stats = {}
    for var, agg in accum.items():
        if agg["count"] == 0:
            continue
        mean = agg["sum"] / agg["count"]
        var_val = (agg["sum_sq"] / agg["count"]) - mean**2
        std = np.sqrt(max(var_val, 0.0))  # éviter std négatif à cause des arrondis

        if variables[var] == "zscore":
            stats[var] = {"mean": float(mean), "std": float(std), "type": "zscore"}
        elif variables[var] == "minmax":
            stats[var] = {"min": float(agg["min"]), "max": float(agg["max"]), "type": "minmax"}

    return stats


def normalize_group(path, variables, norm_types, N_SAMPLES=50):
    files = sorted(path)
    files = random.sample(files, min(N_SAMPLES, len(files)))
    return compute_stats_from_files(files, variables, norm_types)

def build_all_normalization_dicts(asip_dir, cimr_dir, cristal_dir, era5_dir):
    print("🧊 Calcul des stats ASIP...")
    asip_stats = normalize_group(asip_dir, VAR_GROUPS["asip"], norm_types)

    print("🛰️  Calcul des stats CIMR...")
    cimr_stats = normalize_group(cimr_dir, VAR_GROUPS["cimr"], norm_types)

    print("🛰️  Calcul des stats CRISTAL...")
    cristal_stats = normalize_group(cristal_dir, VAR_GROUPS["cristal"], norm_types)

    print("🌦️  Calcul des stats COVARIATES...")
    covs_stats = normalize_group(era5_dir, COVARIATES, norm_types)

    norm_stats = {
        "asip": asip_stats,
        "cimr": cimr_stats,
        "cristal": cristal_stats
    }

    return norm_stats, covs_stats


asip_path = glob("/dmidata/users/maxb/ASIP_OSISAF_dataset/ASIP_L3/*nc")
cimr_path = glob("/dmidata/users/maxb/CROSCIM_dataset/out_CIMR/CIMR5km_*nc")
cristal_path = glob("/dmidata/users/maxb/CROSCIM_dataset/out_CRISTAL/CRISTAL5km_*nc")
era5_path = glob("/dmidata/users/maxb/CROSCIM_dataset/atm_data/atm5km_*.nc")

norm_stats, norm_stats_covs = build_all_normalization_dicts(
    asip_path, cimr_path, cristal_path, era5_path
)

# Enregistrement .txt (format Python)
with open("norm_stats.txt", "w") as f:
    f.write("norm_stats = ")
    f.write(repr(norm_stats))
    f.write("\n\n")
    f.write("norm_stats_covs = ")
    f.write(repr(norm_stats_covs))

import yaml

# Enregistrement .yaml (config)
with open("norm_stats.yaml", "w") as f:
    yaml.dump({"norm_stats": norm_stats, "norm_stats_covs": norm_stats_covs}, 
              f, sort_keys=False, default_flow_style=False)
