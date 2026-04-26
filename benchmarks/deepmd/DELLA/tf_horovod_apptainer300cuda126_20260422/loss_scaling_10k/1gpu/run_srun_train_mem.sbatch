#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=dploss10k1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --time=01:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

image="/scratch/gpfs/BURROWS/akashgpt/softwares/APPTAINER_REPO/deepmd-kit_3.0.0_cuda126.sif"
nproc=1

export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export DP_INFER_BATCH_SIZE=32768
export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=2
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/deepmd-kit
export NCCL_DEBUG=WARN

echo "JOB_START $(date --iso-8601=seconds)"
echo "HOST $(hostname)"
echo "IMAGE ${image}"
echo "NPROC ${nproc}"
nvidia-smi -L || true

monitor_log="gpu_mem_util_${SLURM_JOB_ID}.csv"
(
	echo "timestamp,index,uuid,name,memory.used.MiB,memory.total.MiB,utilization.gpu.percent,utilization.memory.percent,power.draw.W"
	while true; do
		nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits
		sleep 1
	done
) > "${monitor_log}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; wait "${monitor_pid}" 2>/dev/null || true' EXIT
echo "GPU_MONITOR_LOG ${monitor_log}"

apptainer exec --nv "${image}" env PYTHONNOUSERSITE=1 python -c "import os, deepmd, tensorflow as tf, horovod.tensorflow as hvd; print('deepmd', deepmd.__version__); print('tensorflow', tf.__version__); print('hvd_import_ok'); print('cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES')); print('tf_gpus', tf.config.list_physical_devices('GPU'))"

srun --mpi=pmix --ntasks="${nproc}" --cpu-bind=cores --kill-on-bad-exit=1 \
	apptainer exec --nv "${image}" env \
	PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
	HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING}" \
	DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE}" \
	OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
	DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS}" \
	DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS}" \
	TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH}" \
	XLA_FLAGS="${XLA_FLAGS}" \
	NCCL_DEBUG="${NCCL_DEBUG}" \
	python -c "import os, horovod.tensorflow as hvd, tensorflow as tf; hvd.init(); print('hvd_rank', hvd.rank(), 'hvd_size', hvd.size(), 'cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES'), 'tf_gpus', tf.config.list_physical_devices('GPU'))"

srun --mpi=pmix --ntasks="${nproc}" --cpu-bind=cores --kill-on-bad-exit=1 \
	apptainer exec --nv "${image}" env \
	PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
	HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING}" \
	DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE}" \
	OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
	DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS}" \
	DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS}" \
	TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH}" \
	XLA_FLAGS="${XLA_FLAGS}" \
	NCCL_DEBUG="${NCCL_DEBUG}" \
	dp train --mpi-log=workers --skip-neighbor-stat myinput.json

echo "JOB_END $(date --iso-8601=seconds)"
