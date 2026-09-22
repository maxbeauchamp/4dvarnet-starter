#!/bin/bash
# =============================================================================
# NFE (few-step inference) efficiency sweep -- substantiates the "efficient
# few-step inference" claim with an NFE-vs-RMSE and NFE-vs-CRPS curve,
# rather than relying only on the qualitative convergence picture. Reloads
# the checkpoints produced by submit_train.sbatch (SKIP_TRAINING=True, same
# mechanism as submit_metrics.sbatch) -- no retraining involved.
#
# GP: sweeps the 6 generative methods (CM, VarCM, DynCM, VarDynCM, FM,
# DynFM) over their NFE grid (see NFE_GRIDS in run_nfe_sweep.py), plus a
# single reference point for 4dvarnet_lstm (its solver iteration count is
# fixed by training, not swept).
# SSH_GF: only CM, FM, and 4dvarnet_lstm have the sweep cell instrumented
# (see INSTRUMENTED_XP_METHODS in run_nfe_sweep.py) -- note that SSH_GF's
# 4dvarnet_lstm notebook has no SOLVER_TYPE switch (always the UNet
# gradient model), see the caveat printed by that method's sweep cell.
#
# Either way, only plot_nfe_efficiency.py's flagship pair (CM, FM) plus the
# 4dvarnet_lstm reference line end up in the final figure -- the Var*/Dyn*
# CSVs (GP only) are written for inspection but not plotted by default
# (they showed sweep-instability artifacts at intermediate NFE).
#
# Odyssey convention (cf. submit_metrics.sbatch).
#
# Usage:
#   sbatch submit_nfe_sweep.sbatch <GP|SSH_GF>
# =============================================================================

#SBATCH --job-name=nfe_sweep
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

XP="${1:?Usage: sbatch submit_nfe_sweep.sbatch <GP|SSH_GF>}"

cd "$SLURM_SUBMIT_DIR"

if [[ "$XP" == "GP" ]]; then
    METHODS=(CM VarCM DynCM VarDynCM FM DynFM 4dvarnet_lstm)
elif [[ "$XP" == "SSH_GF" ]]; then
    METHODS=(CM FM 4dvarnet_lstm)
else
    echo "ERROR: no NFE sweep cells instrumented for xp=${XP}" >&2
    exit 1
fi

for method in "${METHODS[@]}"; do
    echo "[$(date)] ===== NFE sweep ${method} / ${XP} ====="
    python run_nfe_sweep.py --xp "$XP" --method "$method"
done

echo "[$(date)] ===== Efficiency figure ====="
python plot_nfe_efficiency.py --results-dir "results/${XP}" --out "figures/${XP}/nfe_efficiency"

echo "[$(date)] Done — NFE sweep + figure generated: figures/${XP}/nfe_efficiency.{pdf,png}"
