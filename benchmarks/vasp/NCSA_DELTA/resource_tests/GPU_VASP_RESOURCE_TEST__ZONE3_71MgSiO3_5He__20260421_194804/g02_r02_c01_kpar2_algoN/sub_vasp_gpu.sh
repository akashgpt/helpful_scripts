#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=g02_r02_c01_kpar2_algoN
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export VASP_BIN=/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin/vasp_std__NCSA_DELTA_GPU

module reset
module load nvhpc-hpcx-cuda12/25.3 intel-oneapi-mkl/2024.2.2
ulimit -s unlimited

touch running_RUN_VASP
rm -f done_RUN_VASP failed_RUN_VASP

cleanup() {
	status=$?
	if [[ -n "${gpu_monitor_pid:-}" ]]; then
		kill "$gpu_monitor_pid" 2>/dev/null || true
		wait "$gpu_monitor_pid" 2>/dev/null || true
	fi
	rm -f running_RUN_VASP
	if [[ "$status" -eq 0 ]]; then
		touch done_RUN_VASP
	else
		touch failed_RUN_VASP
	fi
	exit "$status"
}
trap cleanup EXIT INT TERM

{
	echo "Job ID: ${SLURM_JOB_ID:-NA}"
	echo "Started at $(date)"
	echo "Node: $(hostname)"
	echo "SLURM_NTASKS=${SLURM_NTASKS:-NA}"
	echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-NA}"
	echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"
	echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-NA}"
	echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
	echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
	echo "# ========================================="
	echo "[nvidia-smi before]"
	nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv || true
	echo "# ========================================="
	echo
} > log.run_sim

(
	while true; do
		printf '%s\n' "timestamp,$(date --iso-8601=seconds)"
		nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
		nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits || true
		sleep 10
	done
) > gpu_memory_trace.csv 2>&1 &
gpu_monitor_pid=$!

set +e
mpirun --bind-to none -np "${SLURM_NTASKS:-2}" \
	-x OMP_NUM_THREADS \
	-x CUDA_VISIBLE_DEVICES \
	"$VASP_BIN" >> log.run_sim 2>&1
vasp_status=$?
set -e

{
	echo
	echo "# ========================================="
	echo "[nvidia-smi after]"
	nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv || true
	echo "# ========================================="
	if [[ "$vasp_status" -eq 0 ]]; then
		echo "Completed at $(date)"
	else
		echo "Failed with exit status $vasp_status at $(date)"
	fi
} >> log.run_sim

exit "$vasp_status"
