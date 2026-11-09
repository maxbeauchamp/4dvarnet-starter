#!/bin/bash
# One step of the rebuttal plan, on one H100. Submitted by pilot.py, never chained:
#   sbatch --job-name icl-<id> --time ... --export=ALL,BASH_ENV=,... job.sbatch <run_dir> --cmd "<command>"
#SBATCH --mem=64G
#SBATCH --output=runs/slurm-%x-%j.out
set -uo pipefail
RUN_DIR="$1"; shift
export HOME=/Odyssey/private/$USER
export PATH="$ODY_CONDA_ENV/bin:$PATH" CONDA_PREFIX="$ODY_CONDA_ENV"   # no `conda activate`: ~/.bashrc is broken
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd "$ODY_REPO"
mkdir -p "$RUN_DIR"
echo "[job] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-} run_dir=$RUN_DIR args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -u odyssey/run_step.py --run-dir "$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/log.txt"
rc=${PIPESTATUS[0]}
echo "[job] $(date -Is) exit=$rc"
exit $rc
