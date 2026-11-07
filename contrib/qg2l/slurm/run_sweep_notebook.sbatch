#!/bin/bash
# Execute the realism-sweep notebook headless on one GPU (submitted by submit_sweep.sh).
# Adapt the #SBATCH lines to your cluster (partition / account / qos / time limit).
#SBATCH --job-name=qg2l-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/qg2l-sweep-%j.out
##SBATCH --partition=gpu
##SBATCH --account=xxx

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
CONDA_ENV=${CONDA_ENV:-/Odyssey/private/m19beauc/conda_envs/4dvarnet-starter}
NOTEBOOK=${NOTEBOOK:-contrib/qg2l/qg2l_realism_sweep.ipynb}
CHAIN_FILE=${CHAIN_FILE:-}

cd "$REPO" || exit 1
export PATH="$CONDA_ENV/bin:$PATH"     # no 'conda activate' needed on the compute node
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
