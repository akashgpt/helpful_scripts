#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=pt1g16c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=0:10:00

export DP_INFER_BATCH_SIZE=32768
module purge
module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate ALCHEMY_env__PT
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=16

dp --pt train myinput.json
