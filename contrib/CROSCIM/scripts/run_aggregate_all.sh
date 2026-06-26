#!/bin/bash
# Agrège les 3 résolutions via xarray (remplace ncecat qui segfaulte à grande échelle).
# Les 3 résolutions tournent EN PARALLÈLE (indépendantes ; compression mono-thread,
# donc 3 cœurs seulement) -> ~80 min au lieu de ~4 h en séquentiel.
#
# Lancement détaché conseillé (survit à la fermeture de VS Code) :
#   setsid bash run_aggregate_all.sh > ~/croscim_agg_$(date +%Y%m%d_%H%M).log 2>&1 < /dev/null &
set -u

export HDF5_USE_FILE_LOCKING=FALSE
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

pids=()
for res in 2 10 50; do
    echo "==================== lancement x${res} : $(date) ===================="
    python3 "${SCRIPT_DIR}/aggregate_xarray.py" "${res}" &
    pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "❌ une résolution a échoué (pid ${pids[$i]})"; rc=1; }
done

echo "==================== TOUT TERMINÉ (rc=$rc) : $(date) ===================="
exit $rc
