#!/bin/bash
# =============================================================================
# Submit the CROSCIM SIT forecast benchmark (25 sequences x N experiments) on
# Odyssey: contrib/CROSCIM/scripts/run_benchmark_sequences.py
#
# Usage:
#   sbatch submit_benchmark_odyssey.sbatch [run_benchmark_sequences.py args...]
#
# Examples:
#   sbatch submit_benchmark_odyssey.sbatch
#   sbatch submit_benchmark_odyssey.sbatch --experiments UNet_UOAI --seqs 1 2 3
#   sbatch submit_benchmark_odyssey.sbatch --dry-run
#
# The 4 benchmark xps (see XP_YAMLS below, mirrors EXPERIMENTS in
# run_benchmark_sequences.py) all read the same 5 raw dataset directories
# (ASIP_L3, data_noise, out_CRISTAL, atm_data, out_MOD) from
# /Odyssey/public/CROSCIM_dataset, spanning the full year exercised by the 25
# sequences. Those directories are staged once to SCRATCH and paths.* is
# overridden (via run_benchmark_sequences.py's Hydra override passthrough) to
# read from there.
# =============================================================================

#SBATCH --job-name=croscim_benchmark
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=96:00:00
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out
# BASH_ENV vide : évite que bash ne source ~/.bashrc (et son Setup.sh cassé) au
# démarrage du script batch non-interactif.
#SBATCH --export=ALL,BASH_ENV=

export HOME=/Odyssey/private/$USER
source $HOME/.bashrc
conda activate /Odyssey/private/m19beauc/conda_envs/4dvarnet-starter
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: conda activation failed (CONDA_PREFIX is empty)" >&2
    exit 1
fi
echo "[$(date)] conda env: ${CONDA_PREFIX}"
echo "[$(date)] python: $(which python) ($(python --version 2>&1))"

# pas de -u : les scripts d'activation conda référencent des variables non définies
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

XP_YAMLS=(
    "config/xp/CROSCIM/UNet_solvers/base_arctic_croscim_test_sit_UOAI_supervised_forecast.yaml"
    "config/xp/CROSCIM/UNet_solvers/base_arctic_croscim_test_sit_UOAI_supervised_forecast_res10.yaml"
    "config/xp/CROSCIM/UNet_unrolling_solvers/base_arctic_croscim_test_sit_supervised_forecast.yaml"
    "config/xp/CROSCIM/UNet_unrolling_solvers/base_arctic_croscim_test_sit_supervised_forecast_res10.yaml"
)

declare -A RAW_PATHS
for y in "${XP_YAMLS[@]}"; do
    while IFS=: read -r key value; do
        RAW_PATHS["$key"]="$value"
    done < <(grep -E '^\s+(asip|cimr|cristal|covariates|models):\s*/Odyssey/public' "$y" \
                | sed -E 's/^\s*([a-zA-Z_]+):\s*(\S+)\s*$/\1:\2/')
done

if [[ ${#RAW_PATHS[@]} -eq 0 ]]; then
    echo "no asip/cimr/cristal/covariates/models path found under /Odyssey/public in ${XP_YAMLS[*]}" >&2
    exit 1
fi

# create a temporary SCRATCH sub-directory to store the data we need
scratch_dir=/SCRATCH/$USER/$SLURM_JOB_ID
mkdir -p "$scratch_dir"

cleanup() {
    echo "[$(date)] Freeing SCRATCH dir: $scratch_dir"
    rm -rf "$scratch_dir"
}
trap cleanup EXIT

declare -a HYDRA_PATH_OVERRIDES=()
for key in "${!RAW_PATHS[@]}"; do
    src="${RAW_PATHS[$key]}"
    dest="${scratch_dir}/$(basename "$src")"
    echo "[$(date)] Staging paths.${key}: ${src} -> ${dest}"
    mkdir -p "$dest"
    rsync -a --ignore-existing "${src}/" "${dest}/"
    HYDRA_PATH_OVERRIDES+=("paths.${key}=${dest}")
done

echo "[$(date)] Starting benchmark"
HYDRA_FULL_ERROR=1 python contrib/CROSCIM/scripts/run_benchmark_sequences.py \
    "$@" \
    "${HYDRA_PATH_OVERRIDES[@]}"

echo "[$(date)] Done."
