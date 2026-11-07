#!/bin/bash
# Execute the realism-sweep notebook headless on one GPU (submitted by submit_sweep.sh).
# Odyssey settings; other clusters: override with GRES / PARTITION / TIME / MEM / SBATCH_EXTRA in submit_sweep.sh.
#SBATCH --job-name=qg2l-sweep
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# BASH_ENV empty: bash does not source ~/.bashrc (and its broken Setup.sh) when the batch script starts
#SBATCH --export=ALL,BASH_ENV=

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CONDA_ENV=${CONDA_ENV:-/Odyssey/private/m19beauc/conda_envs/4dvarnet-starter}
NOTEBOOK=${NOTEBOOK:-contrib/qg2l/qg2l_realism_sweep.ipynb}
CHAIN_FILE=${CHAIN_FILE:-}

export HOME=/Odyssey/private/$USER
source "$HOME/.bashrc"
conda activate "$CONDA_ENV"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: conda activation failed (CONDA_PREFIX is empty)" >&2
    exit 1
fi
echo "[$(date)] conda env: ${CONDA_PREFIX}"
echo "[$(date)] python: $(which python) ($(python --version 2>&1))"
set -uo pipefail

cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1
echo "[sweep] job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) | repo $REPO | python $(which python)"
nvidia-smi -L || true

# execute a copy of the notebook and write the executed version (with all outputs) back in place
tmp=$(mktemp --suffix=.ipynb)
cp "$NOTEBOOK" "$tmp"
if python -c "import papermill" 2>/dev/null; then
    # papermill streams the cell outputs to this log and saves the notebook after every cell
    python -m papermill "$tmp" "$NOTEBOOK" --log-output --request-save-on-cell-execute -k python3
    status=$?
else
    python -m jupyter nbconvert --to notebook --execute "$tmp" --output "$(basename "$NOTEBOOK")" \
        --output-dir "$(dirname "$NOTEBOOK")" --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=python3
    status=$?
fi
rm -f "$tmp"

echo "[sweep] notebook execution finished with status $status at $(date)"
python -m contrib.qg2l.pipeline data/qg2l_scenarios_small || true
python -m contrib.qg2l.pipeline data/qg2l_scenarios_full || true

# everything done: cancel the remaining jobs of the chain
if [ "$status" -eq 0 ] && [ -n "$CHAIN_FILE" ] && [ -f "$CHAIN_FILE" ]; then
    for j in $(cat "$CHAIN_FILE"); do
        [ "$j" != "${SLURM_JOB_ID:-}" ] && scancel "$j" 2>/dev/null
    done
    echo "[sweep] sweep complete: remaining chained jobs cancelled"
fi
exit $status
