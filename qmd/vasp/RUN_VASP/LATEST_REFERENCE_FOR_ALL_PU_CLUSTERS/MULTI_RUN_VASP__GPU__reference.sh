#!/bin/bash
#SBATCH --account=bguf-delta-gpu    # *** burrows / jiedeng for DELLA and TIGER; bguf-delta-gpu for NCSA DELTA; not applicable for STELLAR ***
#SBATCH --job-name=multi_gpu        # create a short name for your job
#SBATCH --partition=gpuA100x4       # *** gpuA100x4 for NCSA DELTA; gpu for STELLAR ***
#SBATCH --output=multi_run_%j.out   # output file name with job ID
#SBATCH --error=multi_run_%j.err    # error file name with job ID
#SBATCH --nodes=1                   # node count (1 or more)
#SBATCH --gpus-per-node=4           # GPUs per node (DELTA: up to 4; STELLAR: up to 2)
#SBATCH --ntasks-per-node=4         # 1 MPI task per GPU
#SBATCH --cpus-per-task=1            # cpu-cores per GPU (feeds OMP threads)
#SBATCH --mem=180G                  # total CPU memory per node
#SBATCH --time=24:00:00              # total run time limit (HH:MM:SS)
#SBATCH --mail-user=akashgpt@princeton.edu


# ======================================================================
# MULTI_sub_vasp_GPU.sh
# ======================================================================
# Runs many independent VASP calculations, with a configurable number
# of GPUs per VASP calculation, across one or more nodes.
#
# Usage:
#   - Place VASP input files (INCAR, POSCAR, POTCAR, KPOINTS) in
#     subdirectories, each containing a file with "to_RUN" in its name.
#   - KPAR should generally scale with the number of GPUs used by each 
#     VASP run, unless memory is a constraint. See below.
#   - INCAR should generally have NPAR=1, KPAR=1 for these GPU runs.
#     In the 360-atom He_MgSiO3 benchmark (when GPU doesn't have enough
#     memory), KPAR=1 worked for 2/4 GPUs while KPAR=2 or KPAR=4 caused
#     GPU memory failures.
#   - Adjust --nodes, --gpus-per-node, --ntasks-per-node above.
#     Keep ntasks-per-node == gpus-per-node, and set
#     GPUS_PER_INDIVIDUAL_RUN to 1, 2, or 4.
#	- gpus-per-node must be divisible by GPUS_PER_INDIVIDUAL_RUN
#	  ntasks-per-node should match gpus-per-node
# ======================================================================

# number of nodes each individual VASP run will use
export INDIVIDUAL_JOB_TIMEOUT=7200 # seconds; if a VASP run takes longer than this, it will be killed
export MAX_srun_COMMANDS=1000  # hard-limit by administrators on how many srun commands can be active at once; All PU Clusters: 1000
export GPUS_PER_INDIVIDUAL_RUN=2

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_NTASKS_PER_NODE%%(*}}
TOTAL_GPUS=$(( SLURM_NNODES * GPUS_PER_NODE ))
if (( GPUS_PER_INDIVIDUAL_RUN < 1 )); then
	echo "GPUS_PER_INDIVIDUAL_RUN must be >= 1"
	exit 1
fi
if (( GPUS_PER_INDIVIDUAL_RUN > GPUS_PER_NODE )); then
	echo "GPUS_PER_INDIVIDUAL_RUN=$GPUS_PER_INDIVIDUAL_RUN exceeds GPUS_PER_NODE=$GPUS_PER_NODE"
	exit 1
fi
if (( GPUS_PER_NODE % GPUS_PER_INDIVIDUAL_RUN != 0 )); then
	echo "GPUS_PER_NODE=$GPUS_PER_NODE must be divisible by GPUS_PER_INDIVIDUAL_RUN=$GPUS_PER_INDIVIDUAL_RUN"
	exit 1
fi


##################  Environment  ########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [[ $(hostname) == *"della"* ]] || [[ $(hostname) == *"stellar"* ]]; then
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

# Build ordered list of node-local GPU groups.
NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
RUN_SLOTS=()
for node in "${NODES[@]}"; do
	for ((g = 0; g < GPUS_PER_NODE; g += GPUS_PER_INDIVIDUAL_RUN)); do
		gpu_ids=()
		for ((offset = 0; offset < GPUS_PER_INDIVIDUAL_RUN; offset++)); do
			gpu_ids+=("$((g + offset))")
		done
		IFS=,
		RUN_SLOTS+=("${node}:${gpu_ids[*]}")
		unset IFS
	done
done
MAX_INDIVIDUAL_CONCURRENT_JOBS="${#RUN_SLOTS[@]}"


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

# Cap to SLURM's MaxStepCount to avoid wasted time on failed srun calls
if [[ "${#RUN_DIRS[@]}" -gt "$MAX_srun_COMMANDS" ]]; then
    echo "Capping run list from ${#RUN_DIRS[@]} to $MAX_srun_COMMANDS (SLURM MaxStepCount limit)"
    RUN_DIRS=("${RUN_DIRS[@]:0:$MAX_srun_COMMANDS}")
fi

echo "Total run directories: ${#RUN_DIRS[@]}"
echo "Total GPUs: $TOTAL_GPUS (${SLURM_NNODES} nodes x ${GPUS_PER_NODE} GPUs/node)"
echo "GPUs per individual VASP run: $GPUS_PER_INDIVIDUAL_RUN"
echo "Max concurrent jobs: $MAX_INDIVIDUAL_CONCURRENT_JOBS"
echo "CPUs per GPU: $SLURM_CPUS_PER_TASK"
echo "Run directories: ${RUN_DIRS[@]}"
echo "=========================================="


run_one_vasp_gpu() {
	(
		set -euo pipefail

		dir="$1"
		slot_number=$2  # 1-based job slot from GNU parallel {%}

		# Reconstruct RUN_SLOTS array from exported string
		local -a slots_arr=($RUN_SLOTS_STR)
		local slot_idx=$(( (slot_number - 1) % ${#slots_arr[@]} ))
		IFS=':' read -r target_node gpu_ids <<< "${slots_arr[$slot_idx]}"

		cd "$dir"

		cleanup_running_marker() {
			rm -f running_RUN_VASP
		}
		trap cleanup_running_marker EXIT INT TERM

		touch running_RUN_VASP
		rm -f done_RUN_VASP failed_RUN_VASP to_RUN*

		{
			echo "Parent Job ID: ${SLURM_JOB_ID:-NA}"
			echo "Started at $(date)"
			echo "Node: $target_node  GPUs: $gpu_ids  Slot: $slot_number"
			echo "MPI ranks: $GPUS_PER_INDIVIDUAL_RUN"
			echo "# ========================================="
			echo
		} > log.run_sim

		export CUDA_VISIBLE_DEVICES="$gpu_ids"

		set +e
		if [[ $(hostname) == *"della"* ]] || [[ $(hostname) == *"stellar"* ]] || [[ $(hostname) == *"tiger"* ]]; then
			srun --exact --nodes=1 --ntasks="$GPUS_PER_INDIVIDUAL_RUN" \
				--gpus="$GPUS_PER_INDIVIDUAL_RUN" \
				--cpus-per-task="$SLURM_CPUS_PER_TASK" \
				--nodelist="$target_node" \
				"$VASP_BIN" >> log.run_sim 2>&1
			vasp_status=$?
		elif [[ $(hostname) == *"delta"* ]]; then
			mpirun --bind-to none -np "$GPUS_PER_INDIVIDUAL_RUN" \
				--host "${target_node}:${GPUS_PER_INDIVIDUAL_RUN}" \
				-x CUDA_VISIBLE_DEVICES \
				-x OMP_NUM_THREADS \
				"$VASP_BIN" >> log.run_sim 2>&1
			vasp_status=$?
		else
			echo "Unsupported host: $(hostname)" >> log.run_sim
			vasp_status=1
		fi
		set -e

		{
			echo
			echo "# ========================================="
			if [[ "$vasp_status" -eq 0 ]]; then
				echo "Completed at $(date)"
			else
				echo "Failed with exit status $vasp_status at $(date)"
			fi
		} >> log.run_sim

		rm -f running_RUN_VASP
		trap - EXIT INT TERM

		if [[ "$vasp_status" -eq 0 ]]; then
			touch done_RUN_VASP
		else
			touch failed_RUN_VASP
		fi

		return "$vasp_status"
	)
}
export -f run_one_vasp_gpu


##################  Launch with GNU Parallel  ###########################
# {%} = job slot number (1-based, recycled as jobs complete).
# This dynamically assigns each run to whichever GPU slot frees up next,
# so unequal-length jobs don't leave GPUs idle.

# Export variables needed inside run_one_vasp_gpu
export VASP_BIN SLURM_CPUS_PER_TASK SLURM_JOB_ID GPUS_PER_INDIVIDUAL_RUN
# Export RUN_SLOTS array as a string (bash can't export arrays to subshells via GNU parallel)
export RUN_SLOTS_STR="${RUN_SLOTS[*]}"

set +e
parallel -j "$MAX_INDIVIDUAL_CONCURRENT_JOBS" --timeout "$INDIVIDUAL_JOB_TIMEOUT" --lb --tag \
	run_one_vasp_gpu {} {%} ::: "${RUN_DIRS[@]}"
parallel_status=$?
set -e

#########################################################################
if [[ "$parallel_status" -eq 0 ]]; then
	echo "All VASP GPU runs finished successfully at $(date)"
else
	echo "One or more VASP GPU runs failed with GNU parallel exit status $parallel_status at $(date)"
fi
echo "=========================================="
exit "$parallel_status"
