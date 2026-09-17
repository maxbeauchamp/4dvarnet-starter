#!/bin/bash
# =============================================================================
# NFE (few-step inference) efficiency sweep, GP only -- substantiates the
# "efficient few-step inference" claim with an NFE-vs-RMSE/CRPS curve and
# NFE-vs-wall-clock timings, rather than relying only on the qualitative
# convergence picture. Reloads the checkpoints produced by
# submit_train.sbatch (SKIP_TRAINING=True, same mechanism as
# submit_metrics.sbatch) -- no retraining involved.
#
# Runs the 6 generative methods (CM, VarCM, DynCM, VarDynCM, FM, DynFM) over
# their NFE grid (see NFE_GRIDS in run_nfe_sweep.py), plus a single
# reference point for the 4dvarnet_lstm baseline (its solver iteration count
# is fixed by training, not swept). Finishes by building the final
# NFE-vs-RMSE / NFE-vs-wall-clock figure.
#
# Odyssey convention (cf. submit_metrics.sbatch).
#
# Usage:
#   sbatch submit_nfe_sweep.sbatch
# =============================================================================

#SBATCH --job-name=nfe_sweep_GP
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
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

cd "$SLURM_SUBMIT_DIR"

METHODS=(CM VarCM DynCM VarDynCM FM DynFM 4dvarnet_lstm)

for method in "${METHODS[@]}"; do
    echo "[$(date)] ===== NFE sweep ${method} / GP ====="
    python run_nfe_sweep.py --method "$method"
done

echo "[$(date)] ===== Efficiency figure ====="
python plot_nfe_efficiency.py --metric rmse

echo "[$(date)] Done — NFE sweep + figure generated: figures/GP/nfe_efficiency.{pdf,png}"
