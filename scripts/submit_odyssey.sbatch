#!/bin/bash
# =============================================================================
# Run a CROSCIM python entry point on one Odyssey H100.
#
# Usage (from the repository root):
#   sbatch scripts/submit_odyssey.sbatch <script.py> [args / hydra overrides...]
#
# Examples:
#   sbatch scripts/submit_odyssey.sbatch scripts/inspect_data.py \
#       --start 2022-02-01 --end 2022-02-28 --out /Odyssey/private/$USER/results/inspect
#   sbatch scripts/submit_odyssey.sbatch scripts/overfit.py --date 2022-02-15 --levels C --no-pyramid
#   sbatch scripts/submit_odyssey.sbatch main.py xp=N0 model.levels=[C] model.pyramid=false \
#       'datamodule.train_dates=[2021-01-15,2021-12-31]' ...
#
# The Zarr stores (*_zarr in /Odyssey/public/CROSCIM_dataset) are first
# copied to the node-local /SCRATCH/$USER/$SLURM_JOB_ID (removed at the end of
# the job) and the data root is redirected there: paths.data=... for main.py,
# --root ... for the scripts. STAGE=0 sbatch ... reads /Odyssey/public
# directly. No staging for scripts/convert_sources.py, which writes the
# stores; it needs no GPU either (override the request if a CPU partition is
# available).
# =============================================================================

#SBATCH --job-name=croscim
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

set -eo pipefail

SCRIPT="${1:?Usage: sbatch scripts/submit_odyssey.sbatch <script.py> [args...]}"
shift

cd "$SLURM_SUBMIT_DIR"

DATA_ROOT=/Odyssey/public/CROSCIM_dataset
ZARR_STORES=(ASIP_L3_zarr data_noise_zarr out_CRISTAL_zarr atm_data_zarr)
ROOT_OVERRIDE=()
if [[ "${STAGE:-1}" == "1" && "$(basename "$SCRIPT")" != "convert_sources.py" ]]; then
    scratch_dir=/SCRATCH/$USER/$SLURM_JOB_ID
    mkdir -p "$scratch_dir"
    cleanup() {
        echo "[$(date)] Freeing SCRATCH dir: $scratch_dir"
        rm -rf "$scratch_dir"
    }
    trap cleanup EXIT
    t0=$SECONDS
    for store in "${ZARR_STORES[@]}"; do
        echo "[$(date)] Staging ${DATA_ROOT}/${store} -> ${scratch_dir}/"
        rsync -a "${DATA_ROOT}/${store}" "$scratch_dir/"
    done
    echo "[$(date)] Staged $(du -sh "$scratch_dir" | cut -f1) in $((SECONDS - t0)) s"
    if [[ "$(basename "$SCRIPT")" == "main.py" ]]; then
        ROOT_OVERRIDE=("paths.data=${scratch_dir}")
    else
        ROOT_OVERRIDE=(--root "$scratch_dir")
    fi
fi

echo "[$(date)] Running: python ${SCRIPT} $* ${ROOT_OVERRIDE[*]}"
HYDRA_FULL_ERROR=1 srun python "$SCRIPT" "$@" "${ROOT_OVERRIDE[@]}"
echo "[$(date)] Done."
