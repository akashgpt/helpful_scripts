#!/bin/bash
##SBATCH --account=burrows
#SBATCH --job-name=multi_run      # short name for this multi‐run job
#SBATCH --output=multi_run_%j.out # output file name with job ID
#SBATCH --error=multi_run_%j.err  # error file name with job ID
#SBATCH --nodes=64                  # total nodes (4 jobs × 4 nodes each)
#SBATCH --ntasks-per-node=8       # MPI ranks per node
#SBATCH --cpus-per-task=7          # threads per rank
#SBATCH --partition=normal          # partition to run on
#SBATCH --time=48:00:00
#SBATCH --mail-user=akashgpt@tacc.utexas.edu
#SBATCH --mail-type=END,FAIL            # send email on all events (start, end, fail)

##################  Environment  ########################################

# ensure OMP/MPI see the right thread count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK   # some MPI wrappers honour this

module purge
module load intel/19.1.1
module load impi/19.0.9
module load gnuparallel/git20190729
module load vasp/6.4.1

# directory + date
echo "Running VASP multi‐run script"
echo "Time: $(date)"
echo "Location: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

##################  Job list  ###########################################
to_RUN_file="to_RUN_lt_1000"  # the directories that have this file will go into RUN_DIRS
# list your run‐directories here (or build the list dynamically)
# e.g., RUN_DIRS=(T10400/SCALEE_0 T11700/SCALEE_0 T14300/SCALEE_0 T15600/SCALEE_0)
# find all folders with the file $to_RUN_file and add them to the RUN_DIRS array
RUN_DIRS=()  # initialize an empty array

while IFS= read -r -d '' dir; do
    RUN_DIRS+=("$dir")
done < <(
    find . -type d -name 'SCALEE_0' \
            -exec test -f "{}/$to_RUN_file" \; \
            -print0
)

# echo ""
# echo "Run directories: ${RUN_DIRS[@]}"
# echo ""
printf '\nRun directories (%d):\n' "${#RUN_DIRS[@]}"
printf '  %s\n' "${RUN_DIRS[@]}"

##################  Launch with GNU Parallel  ###########################
# Explanation:
#   -j4              → run 4 jobs simultaneously
#   --lb             → line-buffered output (keeps log ordering sane)
#   ibrun … vasp_std  → each parallel task calls its own srun that
#                      grabs 4 nodes exclusively.
#   --nodes=4 --ntasks-per-node=8 --cpus-per-task=7 must match the
#                      per-job resources you want.
#
#   {1}              → placeholder for the directory name
#   ::: "${RUN_DIRS[@]}" feeds the list to parallel

parallel -j4 --lb \
    srun --nodes=4 --ntasks-per-node=8 --cpus-per-task=7 --exclusive \
        --chdir {1} \
        vasp_std '>' {1}/log.run_sim ::: "${RUN_DIRS[@]}"

#########################################################################
echo "All VASP runs finished at $(date)"