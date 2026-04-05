#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=tfhvd1sn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --time=00:10:00

set -euo pipefail
export PS1="${PS1-}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export DP_TRAIN_ARGS="--skip-neighbor-stat"
export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export DP_INFER_BATCH_SIZE=32768
export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=2
export DP_INTER_OP_PARALLELISM_THREADS=1
export NCCL_DEBUG=WARN

module purge
module load gcc/11
module load openmpi/gcc/4.1.6
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate "${ENV_PATH_OVERRIDE:-/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/dp_tf15_src_hvd_fixed3_20260405}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH-}"

nvidia-smi -L || true
python -c "import os, deepmd, tensorflow as tf; print('deepmd', deepmd.__version__); print('tensorflow', tf.__version__); print('cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES')); print('tf_gpus', tf.config.list_physical_devices('GPU'))"

mpirun -np 1 -H localhost:1 --oversubscribe --bind-to none --map-by slot --tag-output \
	-x PATH \
	-x LD_LIBRARY_PATH \
	-x PYTHONNOUSERSITE \
	-x HDF5_USE_FILE_LOCKING \
	-x DP_INFER_BATCH_SIZE \
	-x OMP_NUM_THREADS \
	-x DP_INTRA_OP_PARALLELISM_THREADS \
	-x DP_INTER_OP_PARALLELISM_THREADS \
	-x NCCL_DEBUG \
	-x CUDA_VISIBLE_DEVICES \
	python -c "import os, horovod.tensorflow as hvd, tensorflow as tf; hvd.init(); print('hvd_rank', hvd.rank(), 'hvd_size', hvd.size(), 'cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES'), 'tf_gpus', tf.config.list_physical_devices('GPU'))"

mpirun -np 1 -H localhost:1 --oversubscribe --bind-to none --map-by slot --tag-output \
	-x PATH \
	-x LD_LIBRARY_PATH \
	-x PYTHONNOUSERSITE \
	-x HDF5_USE_FILE_LOCKING \
	-x DP_INFER_BATCH_SIZE \
	-x OMP_NUM_THREADS \
	-x DP_INTRA_OP_PARALLELISM_THREADS \
	-x DP_INTER_OP_PARALLELISM_THREADS \
	-x NCCL_DEBUG \
	-x CUDA_VISIBLE_DEVICES \
	-x DP_TRAIN_ARGS \
	dp train --mpi-log=workers ${DP_TRAIN_ARGS:-} myinput.json
