#!/bin/bash
# Submit the realism-sweep notebook as a chain of Slurm jobs (one GPU each, run one after the other).
#
#   bash contrib/qg2l/slurm/submit_sweep.sh            # 2 chained jobs of 96 h (default, Odyssey settings)
#   bash contrib/qg2l/slurm/submit_sweep.sh 3          # 3 chained jobs
#   CONDA_ENV=/path/to/env bash contrib/qg2l/slurm/submit_sweep.sh
#
# Cluster-specific resources (override the #SBATCH lines of run_sweep_notebook.sbatch), e.g.
#   GRES=gpu:a100:1 PARTITION=gpu TIME=48:00:00 bash contrib/qg2l/slurm/submit_sweep.sh   (defaults: Odyssey, see the .sbatch)
#   SBATCH_EXTRA="--qos=normal --account=myproj" bash contrib/qg2l/slurm/submit_sweep.sh
# Available GPU types / partitions:  sinfo --Format=partition,nodelist:50,gres
#
# Every job re-executes the whole notebook: finished steps are read from the caches, unfinished ones resume
# from their checkpoints, so a job killed by the time limit loses at most a few minutes of work. The first job
# that completes the notebook cancels the rest of the chain. Run from the repository root (after git pull).
set -euo pipefail
N=${1:-2}   # 2 x 96 h by default
cd "$(git rev-parse --show-toplevel)"
mkdir -p logs/slurm
export REPO=$(pwd)
export CONDA_ENV=${CONDA_ENV:-/Odyssey/private/m19beauc/conda_envs/4dvarnet-starter}
export CHAIN_FILE="$REPO/logs/slurm/qg2l-sweep-chain-$(date +%Y%m%d-%H%M%S).txt"
SCRIPT=contrib/qg2l/slurm/run_sweep_notebook.sbatch
OPTS=(--parsable --export=ALL,BASH_ENV=)
[ -n "${GRES:-}" ] && OPTS+=(--gres="$GRES")
[ -n "${PARTITION:-}" ] && OPTS+=(--partition="$PARTITION")
[ -n "${TIME:-}" ] && OPTS+=(--time="$TIME")
[ -n "${MEM:-}" ] && OPTS+=(--mem="$MEM")
# shellcheck disable=SC2206
[ -n "${SBATCH_EXTRA:-}" ] && OPTS+=($SBATCH_EXTRA)
echo "sbatch options: ${OPTS[*]}"

prev=$(sbatch "${OPTS[@]}" "$SCRIPT")
echo "$prev" > "$CHAIN_FILE"
echo "job 1/$N: $prev"
for i in $(seq 2 "$N"); do
    prev=$(sbatch "${OPTS[@]}" --dependency=afterany:"$prev" "$SCRIPT")
    echo "$prev" >> "$CHAIN_FILE"
    echo "job $i/$N: $prev (after the previous one)"
done
echo
echo "logs:     tail -f logs/slurm/qg2l-sweep_<jobid>.out   (errors: .err)"
echo "status:   python -m contrib.qg2l.pipeline data/qg2l_scenarios_small   (and ..._full)"
echo "cancel:   scancel \$(cat $CHAIN_FILE)"
