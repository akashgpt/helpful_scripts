#!/bin/bash
##SBATCH --account=burrows
#SBATCH --job-name=multi_run      # short name for this multi‐run job
#SBATCH --output=multi_run_%j.out # output file name with job ID
#SBATCH --error=multi_run_%j.err  # error file name with job ID
#SBATCH --nodes=__NODES_CHOSEN__  # total nodes (4 jobs × 4 nodes each)
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
to_RUN_keyword="to_RUN__"  # the directories that have this keyword in a filename will go into RUN_DIRS
# list your run‐directories here (or build the list dynamically)
# e.g., RUN_DIRS=(T10400/SCALEE_0 T11700/SCALEE_0 T14300/SCALEE_0 T15600/SCALEE_0)
# find all folders with the keyword $to_RUN_keyword in any file's name and add them to the RUN_DIRS array
RUN_DIRS=()  # initialize an empty array

for dir in */; do
    RUN_DIR_i="$dir/SCALEE_0"

    if [ -d "$RUN_DIR_i" ] && [ -f "$RUN_DIR_i/INCAR" ]; then
        # Check if RUN_DIR_i directory exists and contains an INCAR file
        # check if to_RUN_file* keyword exists in any file's name in the RUN_DIR_i directory
        total_keyword_files=$(find . -type f -name "*$to_RUN_keyword*" | wc -l)
        if [ "$total_keyword_files" -gt 0 ]; then
            echo "Found $to_RUN_keyword in $RUN_DIR_i"

            # if present, cp CONTCAR to POSCAR in the RUN_DIR_i directory
            if [ -f "$RUN_DIR_i/CONTCAR" ]; then
                cp "$RUN_DIR_i/CONTCAR" "$RUN_DIR_i/POSCAR"
                echo "Copied CONTCAR to POSCAR in $RUN_DIR_i"
            fi

            # Append the directory to the RUN_DIRS array
            RUN_DIRS=("${RUN_DIRS[@]}" "$RUN_DIR_i")

            # in RUN_DIR_i, in INCAR, exactly replace "NPAR   = 14" to "NPAR   = 8", preserving spaces
            if [ -f "$RUN_DIR_i/INCAR" ]; then
                sed -i 's/NPAR   = 14/NPAR   = 8/' "$RUN_DIR_i/INCAR"
                echo "Updated NPAR in $RUN_DIR_i/INCAR"
            fi
        
        fi
    fi
done

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