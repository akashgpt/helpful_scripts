#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=pt1_10x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=0:30:00

# Network size scaling benchmark: PT backend, 1 GPU(s), 10x network
# Descriptor neurons: [75, 150, 300]
# Fitting neurons: [720, 720, 720]

export DP_INFER_BATCH_SIZE=32768

module purge
module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate ALCHEMY_env__PT
export PYTHONNOUSERSITE=1

dp --pt train myinput_10x.json
