#!/bin/bash
# =============================================================================
# Submit a CROSCIM training job on Odyssey.
#
# Usage:
#   sbatch submit_train_odyssey.sbatch <xp> [extra hydra overrides...]
#
# Example:
#   sbatch submit_train_odyssey.sbatch \
#       CROSCIM/UNet_solvers/base_arctic_croscim_wpreproc_sit_UOAI_supervised_forecast
#
# Only supports "wpreproc" xps, i.e. configs whose `paths.preproc` points to
# the shared preprocessed dataset (${paths.preproc}/preproc_CROSCIM_x*.nc).
# The needed .nc files are copied to $SCRATCH_DIR and paths.preproc is
# overridden to read from there instead of /Odyssey/public.
# =============================================================================

#SBATCH --job-name=train_croscim
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=96:00:00
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out

source /Odyssey/private/m19beauc/miniconda3/etc/profile.d/conda.sh
conda activate /Odyssey/private/m19beauc/conda_envs/4dvarnet-starter

set -eo pipefail

XP="${1:?Usage: sbatch submit_train_odyssey.sbatch <xp> [hydra overrides...]}"
shift
EXTRA_OVERRIDES=("$@")

cd "$SLURM_SUBMIT_DIR"

XP_YAML="config/xp/${XP}.yaml"
if [[ ! -f "$XP_YAML" ]]; then
    echo "xp config not found: $XP_YAML" >&2
    exit 1
fi

PREPROC_ROOT=$(grep -E '^\s+preproc:\s*/Odyssey/public' "$XP_YAML" | awk '{print $2}')
if [[ -z "$PREPROC_ROOT" ]]; then
    echo "xp '${XP}' has no paths.preproc pointing to /Odyssey/public — this script only supports wpreproc xps." >&2
    exit 1
fi

mapfile -t PREPROC_FILES < <(grep -oE '\$\{paths\.preproc\}/[A-Za-z0-9_.-]+' "$XP_YAML" | sed -E 's#.*/##' | sort -u)
if [[ ${#PREPROC_FILES[@]} -eq 0 ]]; then
    echo "no \${paths.preproc}/<file> reference found in ${XP_YAML}" >&2
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

for f in "${PREPROC_FILES[@]}"; do
    echo "[$(date)] Staging ${PREPROC_ROOT}/${f} -> ${scratch_dir}/${f}"
    rsync -a --ignore-existing "${PREPROC_ROOT}/${f}" "$scratch_dir/"
done

echo "[$(date)] Starting training for xp=${XP}"
HYDRA_FULL_ERROR=1 srun python main.py \
    "xp=${XP}" \
    "paths.preproc=${scratch_dir}" \
    "${EXTRA_OVERRIDES[@]}"

echo "[$(date)] Done."
