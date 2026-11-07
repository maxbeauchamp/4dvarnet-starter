#!/bin/bash
# Submit the realism-sweep notebook as a chain of Slurm jobs (one GPU each, run one after the other).
#
#   bash contrib/qg2l/slurm/submit_sweep.sh            # 4 chained jobs (default)
#   bash contrib/qg2l/slurm/submit_sweep.sh 6          # 6 chained jobs
#   CONDA_ENV=/path/to/env bash contrib/qg2l/slurm/submit_sweep.sh
#
# Every job re-executes the whole notebook: finished steps are read from the caches, unfinished ones resume
# from their checkpoints, so a job killed by the time limit loses at most a few minutes of work. The first job
# that completes the notebook cancels the rest of the chain. Run from the repository root (after git pull).
set -euo pipefail
N=${1:-4}
cd "$(git rev-parse --show-toplevel)"
mkdir -p logs/slurm
export REPO=$(pwd)
export CONDA_ENV=${CONDA_ENV:-/Odyssey/private/m19beauc/conda_envs/4dvarnet-starter}
export CHAIN_FILE="$REPO/logs/slurm/qg2l-sweep-chain-$(date +%Y%m%d-%H%M%S).txt"
SCRIPT=contrib/qg2l/slurm/run_sweep_notebook.sbatch

prev=$(sbatch --parsable --export=ALL "$SCRIPT")
echo "$prev" > "$CHAIN_FILE"
echo "job 1/$N: $prev"
for i in $(seq 2 "$N"); do
    prev=$(sbatch --parsable --export=ALL --dependency=afterany:"$prev" "$SCRIPT")
    echo "$prev" >> "$CHAIN_FILE"
    echo "job $i/$N: $prev (after the previous one)"
done
echo
echo "logs:     tail -f logs/slurm/qg2l-sweep-<jobid>.out"
echo "status:   python -m contrib.qg2l.pipeline data/qg2l_scenarios_small   (and ..._full)"
echo "cancel:   scancel \$(cat $CHAIN_FILE)"
