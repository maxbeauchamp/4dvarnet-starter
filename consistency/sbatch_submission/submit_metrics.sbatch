#!/bin/bash
# =============================================================================
# Sequentially compute metrics for the 8 methods for a given xp (reloads the
# checkpoints produced by submit_train.sbatch): 50 members for the 6
# generative methods (CM, VarCM, DynCM, VarDynCM, FM, DynFM), 1 for the 2
# deterministic ones (UNet, 4dvarnet_lstm). Finishes by building the LaTeX
# summary table.
#
# Odyssey convention (cf. 4dvarnet-starter/submit_train_odyssey.sbatch).
#
# Usage:
#   sbatch submit_metrics.sbatch <GP|SIC|SSH_GF>
# =============================================================================

#SBATCH --job-name=metrics_consistency
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
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

XP="${1:?Usage: sbatch submit_metrics.sbatch <GP|SIC|SSH_GF>}"

cd "$SLURM_SUBMIT_DIR"

METHODS=(UNet 4dvarnet_lstm CM VarCM DynCM VarDynCM FM DynFM)

for method in "${METHODS[@]}"; do
    echo "[$(date)] ===== Metrics ${method} / ${XP} ====="
    python run_metrics.py --xp "$XP" --method "$method"
done

echo "[$(date)] ===== LaTeX table ====="
python make_latex_table.py --xp "$XP"

echo "[$(date)] Done — metrics + LaTeX table generated for xp=${XP}."
