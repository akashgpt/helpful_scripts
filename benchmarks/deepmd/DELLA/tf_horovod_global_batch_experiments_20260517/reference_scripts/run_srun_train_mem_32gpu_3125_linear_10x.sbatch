#!/bin/bash
# 10x-step follow-up generated from an existing validated launcher.
# Source template: 16gpu_625_linear_10x
# New case: 32gpu_3125_linear_10x
# New steps: 3125; decay_steps: 3125
# Note: honors 10k/32*10x step scaling on 8 x 4 A100 nodes.
#SBATCH --account=jiedeng
#SBATCH --qos=gpu-test
#SBATCH --job-name=dpgb10x_32g_3125_lin
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=120G
#SBATCH --time=01:00:00

# DeePMD TF/Horovod multi-node global-batch experiment.
# Case: 32gpu_3125_linear_10x
# Nodes x GPUs/node: 8 x 4 = 32
# Steps: 3125, decay_steps: 3125, scale_by_worker: linear

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

image="/scratch/gpfs/BURROWS/akashgpt/softwares/APPTAINER_REPO/deepmd-kit_3.0.0_cuda126.sif"
nproc="${SLURM_NTASKS:-$((SLURM_NNODES * 4))}"

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
echo "CASE 32gpu_3125_linear_10x"
echo "STEP_SCALING 10k_div_32_times_10x physical_gpus_32 steps_3125"
echo "NODES ${SLURM_NNODES}  NTASKS ${nproc}  NODELIST ${SLURM_JOB_NODELIST}"
echo "IMAGE ${image}"
srun --ntasks-per-node=1 hostname || true
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

srun --mpi=pmix --ntasks="${nproc}" --cpu-bind=cores --kill-on-bad-exit=1 \
	apptainer exec --nv "${image}" env \
	PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
	NCCL_DEBUG="${NCCL_DEBUG}" \
	python -c "import os, socket, horovod.tensorflow as hvd, tensorflow as tf; hvd.init(); print('host', socket.gethostname(), 'hvd_rank', hvd.rank(), 'hvd_size', hvd.size(), 'local_rank', hvd.local_rank(), 'cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES'), 'n_tf_gpus', len(tf.config.list_physical_devices('GPU')))"

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
