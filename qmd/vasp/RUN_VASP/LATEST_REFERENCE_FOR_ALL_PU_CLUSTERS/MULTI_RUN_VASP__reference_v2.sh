#!/bin/bash
#SBATCH --account=burrows
#SBATCH --job-name=multi_run        # create a short name for your job
#SBATCH --output=multi_run_%j.out   # output file name with job ID
#SBATCH --error=multi_run_%j.err    # error file name with job ID
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks-per-node=112        # total number of tasks per node
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G            # memory per cpu-core (4G is default)
#SBATCH --time=1:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ag5805@princeton.edu
##SBATCH --constraint=intel         # *** for Della only ***




# number of tasks each VASP run will use -- IF less than one node/few CPUs only
TASKS_PER_INDIVIDUAL_RUN=8
MAX_INDIVIDUAL_CONCURRENT_JOBS=$(( (SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE) / TASKS_PER_INDIVIDUAL_RUN ))



##################  Environment  ########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export PATH=$PATH:/scratch/gpfs/BURROWS/akashgpt/softwares/vasp.6.4.3/bin

module purge

## Della | Stellar | Tiger3 (all same)
module load intel-oneapi/2024.2
module load intel-mpi/oneapi/2021.13
module load intel-mkl/2024.2
module load hdf5/oneapi-2024.2/1.14.4



##################  Job list  ###########################################
to_RUN_keyword="to_RUN"  # the directories that have this keyword in a filename will go into RUN_DIRS
# list your run‐directories here (or build the list dynamically)
# e.g., RUN_DIRS=(T10400 T11700 T14300 T15600)
# or RUN_DIRS is all the directories that contain the VASP INCAR files
# or RUN_DIRS=(T10400/SCALEE_0 T11700/SCALEE_0 T14300/SCALEE_0 T15600/SCALEE_0)
# or find all folders with the keyword $to_RUN_keyword in any file's name and add them to the RUN_DIRS array
RUN_DIRS=()  # initialize an empty array
for dir in */; do
    RUN_DIR_i="$dir"
    if [ -f "$RUN_DIR_i/INCAR" ]; then
        # check if to_RUN_file* keyword exists in any file's name in the RUN_DIR_i directory
        total_keyword_files=$(find "$RUN_DIR_i" -maxdepth 1 -type f -name "*$to_RUN_keyword*" | wc -l) 
        if [ "$total_keyword_files" -gt 0 ]; then
            echo "Found $to_RUN_keyword in $RUN_DIR_i"
            RUN_DIRS+=("$RUN_DIR_i") # Append the directory to the RUN_DIRS array
        fi
    fi
done

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
    echo "No run directories found (keyword: $to_RUN_keyword, INCAR required). Exiting."
    exit 0
fi

echo "Run directories: ${RUN_DIRS[@]}"




run_one_vasp() {
    (
        set -euo pipefail

        dir="$1"
        tasks_per_run=$2

        cd "$dir"

        # ----------------------------
        # Flags: running / done markers
        # ----------------------------
        touch running_RUN_VASP
        rm -f done_RUN_VASP to_RUN*

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
        # Run VASP (MPI)
        # ----------------------------
        export OMP_NUM_THREADS=1
        export SRUN_CPUS_PER_TASK=1

        srun --nodes=1 --ntasks-per-node="$tasks_per_run" --cpus-per-task=1 \
            vasp_std >> log.run_sim 2>&1

        # ----------------------------
        # Finalize
        # ----------------------------
        {
        echo
        echo "# ========================================="
        echo "Completed at $(date)"
        } >> log.run_sim

        rm -f running_RUN_VASP
        touch done_RUN_VASP
    )
}
export -f run_one_vasp

##################  Launch with GNU Parallel  ###########################
# Explanation:
#   -j4              → run 4 jobs simultaneously
#   --lb             → line-buffered output (keeps log ordering sane)
#   srun … vasp_std  → each parallel task calls its own srun that
#                      grabs 2 nodes exclusively.
#   --nodes=2 --ntasks-per-node=12 --cpus-per-task=8 must match the
#                      per-job resources you want.
#
#   {1}              → placeholder for the directory name
#   ::: "${RUN_DIRS[@]}" feeds the list to parallel
#  --exclusive : ensure that no other srun jobs share the same nodes - remove this option if you want to allow sharing

parallel -j "$MAX_INDIVIDUAL_CONCURRENT_JOBS" --lb --tag \
    run_one_vasp {} "$TASKS_PER_INDIVIDUAL_RUN" ::: "${RUN_DIRS[@]}"

#########################################################################
echo "All VASP runs finished at $(date)"