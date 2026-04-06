#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=pt4_100x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=320G
#SBATCH --time=1:00:00

# Network size scaling benchmark: PT backend, 4 GPU(s), 100x network
# Descriptor neurons: [250, 500, 1000]
# Fitting neurons: [2400, 2400, 2400]

export DP_INFER_BATCH_SIZE=32768

module purge
module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate ALCHEMY_env__PT
export PYTHONNOUSERSITE=1

# Use random port to avoid collision
MASTER_PORT=$((29500 + RANDOM % 1000))

python -m torch.distributed.run \
    --master_port=$MASTER_PORT \
    --nproc_per_node=4 \
    --no-python \
    dp --pt train myinput_100x.json
