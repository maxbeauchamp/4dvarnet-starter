#!/bin/bash

#SBATCH --account=cu_0026
#SBATCH --job-name=train_croscim
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --mem=100GB
#SBATCH --time=1:00:00
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out

cd $SLURM_SUBMIT_DIR/4dvarnet-starter

module load GCC/12.3.0 SciPy-bundle/2023.07
module load Boost/1.82.0
module load snappy/1.1.10
module load GSL/2.7
module load Eigen/3.4.0
module load CUDA/12.8.0
source $HOME/croscim/bin/activate

HYDRA_FULL_ERROR=1 python main.py xp=CROSCIM/4DVarNet_solvers/base_arctic_croscim_wpreproc_sic_res50
