#!/bin/bash
#SBATCH --account=jiedeng       # not valid for Stellar
#SBATCH --job-name=pt_2g4c       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # 4 cpus per GPU x 2 GPUs
#SBATCH --gres=gpu:a100:2        # number of gpus per node
#SBATCH --mem=80G                # memory
#SBATCH --time=0:10:00           # total run time limit (HH:MM:SS)

# =============================================
# PyTorch backend, 2 GPUs, 4 CPUs per GPU — CPU scaling test
# =============================================

export DP_INFER_BATCH_SIZE=32768

module purge
module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate ALCHEMY_env__PT
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=4
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --no-python \
    dp --pt train myinput.json
