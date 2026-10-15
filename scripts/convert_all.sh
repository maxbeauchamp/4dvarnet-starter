#!/bin/bash
# =============================================================================
# Convert every source over a period: one sbatch job per source (they write
# different stores and run in parallel). Missing days of a source (e.g. CIMR
# in September 2022) are simply absent from its store: such days are not
# used as analysis days, and count as unobserved inside the windows.
#
# Usage (from the repository root):
#   scripts/convert_all.sh [start] [end] [workers]     # default: 2022-01-01 2022-12-31 8
#
# Progress bars in croscim_<jobid>.err; the stores of a source are replaced
# only when its job succeeds. Do not train while these jobs run.
# =============================================================================
set -euo pipefail
START="${1:-2022-01-01}"
END="${2:-2022-12-31}"
WORKERS="${3:-8}"
for src in asip cimr cristal era5; do
    sbatch -J "conv_${src}" scripts/submit_odyssey.sbatch scripts/convert_sources.py \
        --start "$START" --end "$END" --sources "$src" --workers "$WORKERS"
done
