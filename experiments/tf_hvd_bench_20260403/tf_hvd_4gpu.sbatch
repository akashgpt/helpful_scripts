#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=tfhvd4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=120G
#SBATCH --time=00:10:00

set -euo pipefail
export PS1="${PS1-}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

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
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12
conda activate /scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/dp_tf15_src_hvd_20260403
NVIDIA_LIB_PATHS="$(python -c "import pathlib, site; print(':'.join(str(path) for site_dir in site.getsitepackages() for path in sorted(pathlib.Path(site_dir, 'nvidia').glob('*/lib')) if path.is_dir()))")"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"

python -c "import os, deepmd, tensorflow as tf; print('deepmd', deepmd.__version__); print('tensorflow', tf.__version__); print('cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES'))"

mpirun -np 4 -H localhost:4 --oversubscribe --bind-to none --map-by slot --tag-output \
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
	python -c "import os, horovod.tensorflow as hvd; hvd.init(); print('hvd_rank', hvd.rank(), 'hvd_size', hvd.size(), 'cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES'))"

mpirun -np 4 -H localhost:4 --oversubscribe --bind-to none --map-by slot --tag-output \
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
	dp train --mpi-log=workers ${DP_TRAIN_ARGS:-} myinput.json
