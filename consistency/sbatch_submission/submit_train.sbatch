#!/bin/bash
# =============================================================================
# Sequentially train the 8 methods (UNet, 4dvarnet_lstm, CM, VarCM, DynCM,
# VarDynCM, FM, DynFM) for a given xp (GP, SIC or SSH_GF), 2000 epochs each.
#
# Odyssey convention (cf. 4dvarnet-starter/submit_train_odyssey.sbatch).
#
# Usage:
#   sbatch submit_train.sbatch <GP|SIC|SSH_GF> [--reset]
# =============================================================================

#SBATCH --job-name=train_consistency
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=96:00:00
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out
# Empty BASH_ENV: prevents bash from sourcing ~/.bashrc (and its broken Setup.sh)
# at non-interactive batch script startup.
#SBATCH --export=ALL,BASH_ENV=

export HOME=/Odyssey/private/$USER
source $HOME/.bashrc
conda activate /Odyssey/private/$USER/conda_envs/4dvarnet-starter
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: conda activation failed (CONDA_PREFIX is empty)" >&2
    exit 1
fi
echo "[$(date)] conda env: ${CONDA_PREFIX}"
echo "[$(date)] python: $(which python) ($(python --version 2>&1))"

set -eo pipefail

XP="${1:?Usage: sbatch submit_train.sbatch <GP|SIC|SSH_GF> [--reset]}"
shift || true
EXTRA_ARGS=("$@")

cd "$SLURM_SUBMIT_DIR"

METHODS=(UNet 4dvarnet_lstm CM VarCM DynCM VarDynCM FM DynFM)

for method in "${METHODS[@]}"; do
    echo "[$(date)] ===== Training ${method} / ${XP} ====="
    python run_training.py --xp "$XP" --method "$method" "${EXTRA_ARGS[@]}"
done

echo "[$(date)] Done — 8 methods trained for xp=${XP}."
