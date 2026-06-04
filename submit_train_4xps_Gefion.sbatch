#!/bin/bash
# =============================================================================
# Launcher: submit 4 independent CROSCIM training jobs on Gefion
# Run as a plain bash script: bash submit_train_4xps_Gefion.sbatch
# Each experiment appears as a separate job in squeue.
# A 5th job copies checkpoints after all 4 jobs finish.
# =============================================================================

set -euo pipefail

SRC_DIR=/dcai/users/beamax
RESULTS_DIR=${SRC_DIR}/results
CKPT_DIR=${SRC_DIR}/4dvarnet-starter/ckpt/CROSCIM
WORK_DIR=${SRC_DIR}/4dvarnet-starter
ACCOUNT=cu_0026
NODES=1
NTASKS_PER_NODE=8
GPUS_PER_NODE=8
CPUS_PER_TASK=6
TIME=72:00:00

mkdir -p "${CKPT_DIR}"
mkdir -p "${WORK_DIR}/logs"

declare -a XP_CONFIGS=(
    "CROSCIM/UNet_unrolling_solvers/base_arctic_croscim_wpreproc_sit_supervised_forecast_Gefion"
    "CROSCIM/UNet_unrolling_solvers/base_arctic_croscim_wpreproc_sit_supervised_forecast_res10_Gefion"
    "CROSCIM/UNet_solvers/base_arctic_croscim_wpreproc_sit_UOAI_supervised_forecast_Gefion"
    "CROSCIM/UNet_solvers/base_arctic_croscim_wpreproc_sit_UOAI_supervised_forecast_res10_Gefion"
)

declare -a JOB_NAMES=(
    "croscim_UNet_unroll_fc"
    "croscim_UNet_unroll_fc_r10"
    "croscim_UOAI_fc"
    "croscim_UOAI_fc_r10"
)

declare -a TARGET_CKPTS=(
    "base_croscim_UNet_unrolling_sit_supervised_forecast.ckpt"
    "base_croscim_UNet_unrolling_sit_supervised_forecast_res10.ckpt"
    "base_croscim_UNet_sit_UOAI_supervised_forecast.ckpt"
    "base_croscim_UNet_sit_UOAI_supervised_forecast_res10.ckpt"
)

NUM_XPS=${#XP_CONFIGS[@]}
declare -a JOB_IDS=()

for (( i=0; i<NUM_XPS; i++ )); do
    XP_CONFIG="${XP_CONFIGS[$i]}"
    JOB_NAME="${JOB_NAMES[$i]}"
    XP_STEM=$(basename "${XP_CONFIG}")
    XP_OUTDIR="${RESULTS_DIR}/${XP_STEM}"

    JID=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --account="${ACCOUNT}" \
        --nodes="${NODES}" \
        --ntasks-per-node="${NTASKS_PER_NODE}" \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --time="${TIME}" \
        --output="${WORK_DIR}/logs/${XP_STEM}_%j.out" \
        --error="${WORK_DIR}/logs/${XP_STEM}_%j.err" \
        --wrap="

module load GCC/12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
module load Boost/1.82.0-GCC-12.3.0
module load snappy/1.1.10-GCCcore-12.3.0
module load GSL/2.7-GCC-12.3.0
module load Eigen/3.4.0-GCCcore-12.3.0
module load CUDA/12.8.0
source \$HOME/croscim/bin/activate
cd ${WORK_DIR}

echo '['\$(date)'] Starting ${XP_STEM}'
echo 'Hydra output dir: ${XP_OUTDIR}'
echo 'Lightning logger save_dir: ${XP_OUTDIR}'

srun python main.py \\
    'xp=${XP_CONFIG}' \\
    'hydra.run.dir=${XP_OUTDIR}' \\
    'logger.save_dir=${XP_OUTDIR}' \\
    'logger.name=.' \\
    'logger.version='

echo '['\$(date)'] Finished ${XP_STEM}'
")

    JOB_IDS+=("$JID")
    echo "Submitted ${JOB_NAME}  ->  job ${JID}  (xp: ${XP_CONFIG})"
done

DEP=$(IFS=:; echo "afterany:${JOB_IDS[*]}")

COPY_CMDS=""
for (( i=0; i<NUM_XPS; i++ )); do
    XP_STEM=$(basename "${XP_CONFIGS[$i]}")
    TARGET="${TARGET_CKPTS[$i]}"
    XP_OUTDIR="${RESULTS_DIR}/${XP_STEM}"

    COPY_CMDS+="
echo ''
echo '--- ${XP_STEM} ---'
echo 'Searching checkpoints under: ${XP_OUTDIR}'

find '${XP_OUTDIR}' -type f -name '*.ckpt' 2>/dev/null || true

BEST=\$(find '${XP_OUTDIR}' -type f -name '*.ckpt' ! -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)

if [[ -z \"\$BEST\" ]]; then
    echo 'No non-last checkpoint found, trying last.ckpt...'
    BEST=\$(find '${XP_OUTDIR}' -type f -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)
fi

if [[ -n \"\$BEST\" ]]; then
    cp -v \"\$BEST\" '${CKPT_DIR}/${TARGET}'
    echo \"Copied: \$BEST -> ${CKPT_DIR}/${TARGET}\"
else
    echo 'WARNING: no checkpoint found for ${XP_STEM}' >&2
fi
"
done

COPY_JID=$(sbatch --parsable \
    --job-name="croscim_copy_ckpts" \
    --account="${ACCOUNT}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --time=00:10:00 \
    --dependency="${DEP}" \
    --output="${WORK_DIR}/logs/copy_ckpts_%j.out" \
    --error="${WORK_DIR}/logs/copy_ckpts_%j.err" \
    --wrap="
set -euo pipefail

echo '['\$(date)'] Copying checkpoints...'
mkdir -p '${CKPT_DIR}'

${COPY_CMDS}

echo ''
echo '['\$(date)'] Done.'
ls -lh '${CKPT_DIR}'
")

echo ""
echo "Submitted checkpoint-copy job  ->  job ${COPY_JID}  (dependency: ${DEP})"
echo ""
echo "Track with:  squeue -u \$USER"