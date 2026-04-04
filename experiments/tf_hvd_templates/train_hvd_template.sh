#!/bin/bash
#SBATCH --account=__ACCOUNT__
#SBATCH --job-name=__JOB_NAME__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__CPUS_PER_TASK__
#SBATCH --gres=gpu:a100:__NUM_GPUS__
#SBATCH --mem=__MEMORY__
#SBATCH --time=00:10:00

set -euo pipefail

export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export DP_INFER_BATCH_SIZE=32768
export OMP_NUM_THREADS=__OMP_THREADS__
export DP_INTRA_OP_PARALLELISM_THREADS=__DP_INTRA_THREADS__
export DP_INTER_OP_PARALLELISM_THREADS=__DP_INTER_THREADS__

module load gcc-toolset/14
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate __ENV_PATH__

python - <<'PY'
import deepmd
import horovod.tensorflow as hvd
import tensorflow as tf

print("deepmd", deepmd.__version__)
print("tensorflow", tf.__version__)
print("horovod_rank", hvd.rank())
print("horovod_size", hvd.size())
PY

python -m horovod.runner.launch \
	-np __NUM_GPUS__ \
	-H localhost:__NUM_GPUS__ \
	dp train --mpi-log=workers myinput.json
