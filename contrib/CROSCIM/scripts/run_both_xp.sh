#!/bin/bash
# Lance les 2 expériences CROSCIM en arrière-plan, détachées (survivent à la
# fermeture de VS Code). Elles tournent en parallèle : l'unrolling sur les GPU
# [0,1], l'UOAI sur les GPU [2,3] (déjà fixé dans les configs).
#
# Usage :  bash run_both_xp.sh
# Les logs vont dans ~ (hors du repo, pour ne rien committer par erreur).
# (pas de `set -u` : les scripts d'activation conda référencent des variables
#  non définies et planteraient.)

PROJ=/Odyssey/private/m19beauc/4dvarnet-starter
cd "$PROJ" || exit 1

# Active l'environnement conda
source /Odyssey/private/m19beauc/conda/etc/profile.d/conda.sh
conda activate 4dvarnet-starter

export HYDRA_FULL_ERROR=1
TS=$(date +%Y%m%d_%H%M)

LOG_UNROLL=~/croscim_xp_unrolling_${TS}.log
LOG_UOAI=~/croscim_xp_UOAI_${TS}.log

# setsid -> nouvelle session sans terminal de contrôle => survit à VS Code fermé
setsid python main.py xp=CROSCIM/UNet_unrolling_solvers/base_arctic_croscim_wpreproc_sit_supervised_forecast.yaml \
    > "$LOG_UNROLL" 2>&1 < /dev/null &
echo "unrolling (GPU 0,1) -> PID $!  log: $LOG_UNROLL"

setsid python main.py xp=CROSCIM/UNet_solvers/base_arctic_croscim_wpreproc_sit_UOAI_supervised_forecast.yaml \
    > "$LOG_UOAI" 2>&1 < /dev/null &
echo "UOAI      (GPU 2,3) -> PID $!  log: $LOG_UOAI"

echo ""
echo "Suivi :  tail -f $LOG_UNROLL"
echo "         tail -f $LOG_UOAI"
