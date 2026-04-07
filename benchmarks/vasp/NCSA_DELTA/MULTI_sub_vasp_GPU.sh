#!/bin/bash
#SBATCH --account=bguf-delta-gpu    # *** burrows / jiedeng for DELLA and TIGER; bguf-delta-gpu for NCSA DELTA; not applicable for STELLAR ***
#SBATCH --job-name=multi_gpu        # create a short name for your job
#SBATCH --partition=gpuA100x4       # *** gpuA100x4 for NCSA DELTA; gpu for STELLAR ***
#SBATCH --output=multi_run_%j.out   # output file name with job ID
#SBATCH --error=multi_run_%j.err    # error file name with job ID
#SBATCH --nodes=1                   # node count (1 or more)
#SBATCH --gpus-per-node=4           # GPUs per node (DELTA: up to 4; STELLAR: up to 2)
#SBATCH --ntasks-per-node=4         # 1 MPI task per GPU
#SBATCH --cpus-per-task=16          # cpu-cores per GPU (feeds OMP threads)
#SBATCH --mem-per-cpu=1G            # memory per cpu-core
#SBATCH --time=1:00:00              # total run time limit (HH:MM:SS)
#SBATCH --mail-user=akashgpt@princeton.edu


# ======================================================================
# MULTI_sub_vasp_GPU.sh
# ======================================================================
# Runs many independent VASP calculations, one per GPU, across one or
# more nodes. Each VASP instance is 1 MPI rank + OMP threads + 1 GPU.
#
# Usage:
#   - Place VASP input files (INCAR, POSCAR, POTCAR, KPOINTS) in
#     subdirectories, each containing a file with "to_RUN" in its name.
#   - INCAR must have NPAR=1, KPAR=1 (single-GPU settings).
#   - Adjust --nodes, --gpus-per-node, --ntasks-per-node above.
#     Keep ntasks-per-node == gpus-per-node (1 task per GPU).
# ======================================================================


GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_NTASKS_PER_NODE%%(*}}
TOTAL_GPUS=$(( SLURM_NNODES * GPUS_PER_NODE ))


##################  Environment  ########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [[ $(hostname) == *"della"* ]] || [[ $(hostname) == *"stellar"* ]] || [[ $(hostname) == *"tiger"* ]]; then
	VASP_BIN=/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin/vasp_std
	module purge
	module load nvhpc/25.5
	module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
	module load intel-mkl/2024.2
	module load anaconda3/2025.12
	conda activate ALCHEMY_env
elif [[ $(hostname) == *"delta"* ]]; then
	VASP_BIN=/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin/vasp_std__NCSA_DELTA_GPU
	module reset
	module load nvhpc-hpcx-cuda12/25.3 intel-oneapi-mkl/2024.2.2 miniforge3-python
	eval "$(conda shell.bash hook)"
	conda activate ALCHEMY_env
	ulimit -s unlimited
fi


echo "=========================================="

# Build ordered list of (node:gpu_id) slots
NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
SLOTS=()
for node in "${NODES[@]}"; do
	for ((g = 0; g < GPUS_PER_NODE; g++)); do
		SLOTS+=("${node}:${g}")
	done
done


##################  Job list  ###########################################
to_RUN_keyword="to_RUN"
RUN_DIRS=()
for dir in */; do
	RUN_DIR_i="$dir"
	if [ -f "$RUN_DIR_i/INCAR" ]; then
		total_keyword_files=$(find "$RUN_DIR_i" -maxdepth 1 -type f -name "*$to_RUN_keyword*" | wc -l)
		if [ "$total_keyword_files" -gt 0 ]; then
			echo "Found $to_RUN_keyword in $RUN_DIR_i"
			RUN_DIRS+=("$RUN_DIR_i")
		fi
	fi
done

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
	echo "No run directories found (keyword: $to_RUN_keyword, INCAR required). Exiting."
	exit 0
fi

echo "Total run directories: ${#RUN_DIRS[@]}"
echo "Total GPUs: $TOTAL_GPUS (${SLURM_NNODES} nodes x ${GPUS_PER_NODE} GPUs/node)"
echo "Max concurrent jobs: $TOTAL_GPUS"
echo "CPUs per GPU: $SLURM_CPUS_PER_TASK"
echo "Run directories: ${RUN_DIRS[@]}"
echo "=========================================="


run_one_vasp_gpu() {
	(
		set -euo pipefail

		dir="$1"
		slot_index=$2   # 0-based index into SLOTS array

		# Parse node and GPU ID from slot
		IFS=':' read -r target_node gpu_id <<< "$3"

		cd "$dir"

		touch running_RUN_VASP
		rm -f done_RUN_VASP to_RUN*

		{
			echo "Parent Job ID: ${SLURM_JOB_ID:-NA}"
			echo "Started at $(date)"
			echo "Node: $target_node  GPU: $gpu_id"
			echo "# ========================================="
			echo
		} > log.run_sim

		export CUDA_VISIBLE_DEVICES="$gpu_id"

		if [[ $(hostname) == *"della"* ]] || [[ $(hostname) == *"stellar"* ]] || [[ $(hostname) == *"tiger"* ]]; then
			srun --exact --nodes=1 --ntasks=1 \
				--gpus=1 \
				--cpus-per-task="$SLURM_CPUS_PER_TASK" \
				--nodelist="$target_node" \
				"$VASP_BIN" >> log.run_sim 2>&1
		elif [[ $(hostname) == *"delta"* ]]; then
			mpirun --bind-to none -np 1 \
				--host "$target_node" \
				-x CUDA_VISIBLE_DEVICES \
				-x OMP_NUM_THREADS \
				"$VASP_BIN" >> log.run_sim 2>&1
		fi

		{
			echo
			echo "# ========================================="
			echo "Completed at $(date)"
		} >> log.run_sim

		rm -f running_RUN_VASP
		touch done_RUN_VASP
	)
}
export -f run_one_vasp_gpu


##################  Launch with GNU Parallel  ###########################
# GNU parallel's {%} gives the job slot number (1-based, cycles 1..TOTAL_GPUS).
# We map each slot to a (node:gpu_id) pair from the SLOTS array.
# {#} is the global job sequence number (for logging).

# Export variables needed inside run_one_vasp_gpu
export VASP_BIN SLURM_CPUS_PER_TASK SLURM_JOB_ID

# Build a slot file for parallel to read alongside the directories
# Each job gets: dir, slot_index, node:gpu_id
SLOT_FILE=$(mktemp)
for ((i = 0; i < ${#RUN_DIRS[@]}; i++)); do
	slot_idx=$(( i % TOTAL_GPUS ))
	echo "${RUN_DIRS[$i]} $slot_idx ${SLOTS[$slot_idx]}"
done > "$SLOT_FILE"

parallel -j "$TOTAL_GPUS" --lb --tag --colsep ' ' \
	run_one_vasp_gpu {1} {2} {3} :::: "$SLOT_FILE"

rm -f "$SLOT_FILE"

#########################################################################
echo "All VASP GPU runs finished at $(date)"
echo "=========================================="
