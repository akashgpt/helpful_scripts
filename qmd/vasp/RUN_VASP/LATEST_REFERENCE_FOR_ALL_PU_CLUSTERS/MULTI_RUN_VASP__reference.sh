#!/bin/bash
##SBATCH --account=burrows
#SBATCH --job-name=multi_run        # create a short name for your job
#SBATCH --output=multi_run_%j.out   # output file name with job ID
#SBATCH --error=multi_run_%j.err    # error file name with job ID
#SBATCH --nodes=10                  # node count
#SBATCH --ntasks-per-node=12        # total number of tasks per node
#SBATCH --cpus-per-task=8           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G            # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ag5805@princeton.edu
##SBATCH --constraint=intel         # *** for Della only ***

# number of nodes each individual VASP run will use
export INDIVIDUAL_JOB_TIMEOUT=86400 # seconds; if a VASP run takes longer than this, it will be killed
export NODES_PER_INDIVIDUAL_RUN=1
export MAX_srun_COMMANDS=1000  # hard-limit by administrators on how many srun commands can be active at once; All PU Clusters: 1000
MAX_INDIVIDUAL_CONCURRENT_JOBS=$(( SLURM_NNODES / NODES_PER_INDIVIDUAL_RUN )) # how many VASP runs to run simultaneously
export TASKS_PER_NODE=${SLURM_NTASKS_PER_NODE%%(*}



##################  Environment  ########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [[ $(hostname) == *"della"* ]] || [[ $(hostname) == *"stellar"* ]] || [[ $(hostname) == *"tiger"* ]]; then
    export VASP_BIN=/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std
    module purge
    module load intel-oneapi/2024.2
    module load intel-mpi/oneapi/2021.13
    module load intel-mkl/2024.2
    module load hdf5/oneapi-2024.2/1.14.4
    module load anaconda3/2025.12
    conda activate ALCHEMY_env
elif [[ $(hostname) == *"delta"* ]]; then
    export VASP_BIN=/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std
    module reset
    module load PrgEnv-gnu cray-hdf5-parallel miniforge3-python
    eval "$(conda shell.bash hook)"
    conda activate ALCHEMY_env
    ulimit -s unlimited
fi

echo "=========================================="


##################  Job list  ###########################################
# ===============
SEARCH_MODE="folders_at_depth_1"  # "folders_at_depth_1" or "folders_at_all_depths"
# ===============
to_RUN_keyword="to_RUN"  # the directories that have this keyword in a filename will go into RUN_DIRS
to_SKIP_keywords=("running_RUN_VASP" "done_RUN_VASP") # directories with any of these keywords in a filename will be skipped
# ===============
# list your run‐directories here (or build the list dynamically)
# e.g., RUN_DIRS=(T10400 T11700 T14300 T15600)
# or RUN_DIRS is all the directories that contain the VASP INCAR files
# or RUN_DIRS=(T10400/SCALEE_0 T11700/SCALEE_0 T14300/SCALEE_0 T15600/SCALEE_0)
# or find all folders with the keyword $to_RUN_keyword in any file's name and add them to the RUN_DIRS array

RUN_DIRS=()  # initialize an empty array

# Build candidate directory list based on SEARCH_MODE
CANDIDATE_DIRS=()
if [ "$SEARCH_MODE" = "folders_at_depth_1" ]; then
    # Only immediate subdirectories (depth 1)
    for dir in */; do
        CANDIDATE_DIRS+=("$dir")
    done
elif [ "$SEARCH_MODE" = "folders_at_all_depths" ]; then
    # All subdirectories at any depth (sorted for reproducibility)
    while IFS= read -r dir; do
        CANDIDATE_DIRS+=("$dir")
    done < <(find . -mindepth 1 -type d | sort | sed 's|^\./||; s|$|/|')
else
    echo "ERROR: Unknown SEARCH_MODE='$SEARCH_MODE'. Use 'folders_at_depth_1' or 'folders_at_all_depths'."
    exit 1
fi

for RUN_DIR_i in "${CANDIDATE_DIRS[@]}"; do
    if [ -f "$RUN_DIR_i/INCAR" ]; then
        # check if to_RUN keyword exists in any file's name
        total_keyword_files=$(find "$RUN_DIR_i" -maxdepth 1 -type f -name "*$to_RUN_keyword*" | wc -l)
        if [ "$total_keyword_files" -gt 0 ]; then
            # check if any skip keyword exists — if so, skip this directory
            skip=false
            for skip_kw in "${to_SKIP_keywords[@]}"; do
                skip_count=$(find "$RUN_DIR_i" -maxdepth 1 -type f -name "*${skip_kw}*" | wc -l)
                if [ "$skip_count" -gt 0 ]; then
                    echo "Skipping $RUN_DIR_i (found $skip_kw)"
                    skip=true
                    break
                fi
            done
            if [ "$skip" = false ]; then
                echo "Found $to_RUN_keyword in $RUN_DIR_i"
                RUN_DIRS+=("$RUN_DIR_i")
            fi
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

echo "Total number of run directories: ${#RUN_DIRS[@]}"
echo "Max individual concurrent jobs: $MAX_INDIVIDUAL_CONCURRENT_JOBS"
echo "Each VASP run will use $NODES_PER_INDIVIDUAL_RUN node(s) with $SLURM_NTASKS_PER_NODE tasks x $SLURM_CPUS_PER_TASK cpus-per-task."
echo "Run directories: ${RUN_DIRS[@]}"
echo "=========================================="




run_one_vasp() {
    (
        set -euo pipefail

        dir="$1"
        nodes_per_run=$2

        cd "$dir"

        cleanup_running_marker() {
            rm -f running_RUN_VASP
        }
        trap cleanup_running_marker EXIT INT TERM

        # ----------------------------
        # Flags: running / done markers
        # ----------------------------
        touch running_RUN_VASP # indicates that this run is currently running
        rm -f done_RUN_VASP failed_RUN_VASP

        # ----------------------------
        # Log: log.run_sim
        # ----------------------------
        rm -f log.run_sim
        {
            echo "Parent Job ID: ${SLURM_JOB_ID:-NA}"
            echo "Started at $(date)"
            echo "# ========================================="
            echo
        } > log.run_sim

        # ----------------------------
        # Run VASP (MPI + OpenMP)
        # ----------------------------
        set +e
        if [[ $(hostname) == *"della"* ]] || [[ $(hostname) == *"stellar"* ]] || [[ $(hostname) == *"tiger"* ]]; then
            srun --nodes="$nodes_per_run" \
                --ntasks-per-node="$TASKS_PER_NODE" \
                --cpus-per-task="$SLURM_CPUS_PER_TASK" \
                "$VASP_BIN" >> log.run_sim 2>&1
            vasp_status=$?
        elif [[ $(hostname) == *"delta"* ]]; then
            srun --exclusive \
                --nodes="$nodes_per_run" \
                --ntasks-per-node="$TASKS_PER_NODE" \
                --cpus-per-task="$SLURM_CPUS_PER_TASK" \
                --cpu-bind=cores \
                --distribution=block:block \
                "$VASP_BIN" >> log.run_sim 2>&1
            vasp_status=$?
        else
            echo "Unsupported host: $(hostname)" >> log.run_sim
            vasp_status=1
        fi
        set -e

        # ----------------------------
        # Finalize
        # ----------------------------
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
            rm -f to_RUN*
            touch done_RUN_VASP # indicates that this run is done
        else
            touch failed_RUN_VASP # indicates that this run failed
        fi

        return "$vasp_status"
    )
}
export -f run_one_vasp

##################  Launch with GNU Parallel  ###########################
# Explanation:
#   -j $MAX_INDIVIDUAL_CONCURRENT_JOBS → run that many jobs simultaneously
#   --lb             → line-buffered output (keeps log ordering sane)
#   Each parallel task calls run_one_vasp which does its own srun
#     grabbing $NODES_PER_INDIVIDUAL_RUN node(s) exclusively.
#
#   {1}              → placeholder for the directory name
#   ::: "${RUN_DIRS[@]}" feeds the list to parallel

parallel -j "$MAX_INDIVIDUAL_CONCURRENT_JOBS" --timeout "$INDIVIDUAL_JOB_TIMEOUT" --lb --tag \
    run_one_vasp {} "$NODES_PER_INDIVIDUAL_RUN" ::: "${RUN_DIRS[@]}"

#########################################################################
echo "All VASP runs finished at $(date)"
echo "=========================================="