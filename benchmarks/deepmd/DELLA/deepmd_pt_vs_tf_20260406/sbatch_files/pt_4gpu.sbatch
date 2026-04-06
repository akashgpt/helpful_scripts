#!/bin/bash
#SBATCH --account=jiedeng       # not valid for Stellar
#SBATCH --job-name=pt_4gpu       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task
#SBATCH --gres=gpu:a100:4        # number of gpus per node
#SBATCH --mem=160G               # memory
#SBATCH --time=0:10:00           # total run time limit (HH:MM:SS)

# =============================================
# PyTorch backend, 4 GPUs — parallel training via DDP
# Uses python -m torch.distributed.run (equivalent to torchrun)
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

python -m torch.distributed.run \
    --nproc_per_node=4 \
    --no-python \
    dp --pt train myinput.json
