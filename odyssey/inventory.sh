#!/bin/bash
# Lists the checkpoints of the submitted paper on Odyssey (read only), to choose ODY_OLD and the FM/DynFM teachers.
# Runs on a GPU node through the pilot (step p0_inventory), never on the login node.
ROOT=/Odyssey/private/$USER
echo "== clones of 4dvarnet-starter under $ROOT"
find "$ROOT" -maxdepth 4 -type d -name consistency -path '*4dvarnet*' -not -path '*/conda_envs/*' 2>/dev/null
echo
echo "== consistency checkpoints (dir, number of .ckpt, newest .ckpt date)"
for d in $(find "$ROOT" -maxdepth 7 -type d -name 'logs*' -path '*consistency/Notebooks/Notebooks_*' -not -path '*/conda_envs/*' 2>/dev/null | sort); do
  n=$(find "$d" -name '*.ckpt' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || continue
  newest=$(find "$d" -name '*.ckpt' -printf '%TY-%Tm-%Td %p\n' 2>/dev/null | sort | tail -1)
  echo "$d  ckpt=$n  newest: $newest"
done
echo
echo "== existing metrics CSVs"
find "$ROOT" -maxdepth 7 -path '*consistency/sbatch_submission/results/*' -name '*.csv' -not -path '*/conda_envs/*' 2>/dev/null | sort
