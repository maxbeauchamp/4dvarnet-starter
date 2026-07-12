#!/bin/bash
# =============================================================================
# Submit the CROSCIM SIT forecast benchmark (25 sequences x N experiments) on
# Odyssey: contrib/CROSCIM/scripts/run_benchmark_sequences.py
#
# Usage:
#   sbatch submit_benchmark_odyssey.sbatch [run_benchmark_sequences.py args...]
#
# Examples:
#   sbatch submit_benchmark_odyssey.sbatch
#   sbatch submit_benchmark_odyssey.sbatch --experiments UNet_UOAI --seqs 1 2 3
#   sbatch submit_benchmark_odyssey.sbatch --dry-run
#
# Unlike submit_train_odyssey.sbatch, this does not stage data to SCRATCH: the
# benchmark uses the "test_*" xps, which read many raw dataset directories
# (ASIP_L3, out_CRISTAL, atm_data, out_MOD/out_CIMR) directly from
# /Odyssey/public/CROSCIM_dataset.
# =============================================================================

#SBATCH --job-name=croscim_benchmark
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=96:00:00
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out

source /Odyssey/private/m19beauc/miniconda3/etc/profile.d/conda.sh
conda activate /Odyssey/private/m19beauc/conda_envs/4dvarnet-starter

# pas de -u : les scripts d'activation conda référencent des variables non définies
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

HYDRA_FULL_ERROR=1 python contrib/CROSCIM/scripts/run_benchmark_sequences.py "$@"
