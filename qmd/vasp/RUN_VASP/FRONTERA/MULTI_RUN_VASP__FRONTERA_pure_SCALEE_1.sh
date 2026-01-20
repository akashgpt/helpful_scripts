#!/bin/bash
##SBATCH --account=burrows
#SBATCH --job-name=multi_run      # short name for this multi‐run job
#SBATCH --output=multi_run_%j.out # output file name with job ID
#SBATCH --error=multi_run_%j.err  # error file name with job ID
#SBATCH --nodes=32                  # total nodes (4 jobs × 4 nodes each)
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
# list your run‐directories here (or build the list dynamically)
# e.g., RUN_DIRS=(T10400/SCALEE_0 T11700/SCALEE_0 T14300/SCALEE_0 T15600/SCALEE_0)
# RUN_DIRS are all the directories that contain the VASP INCAR files

# 1) build an array of each directory containing */*_0H*/SCALEE_1/INCAR
RUN_DIRS=()
while IFS= read -r -d '' incar; do
    RUN_DIRS+=( "$(dirname "$incar")" )
done < <(
    find . \
        -type f \
        -path "*/*/*/*/*_0H*/SCALEE_1/INCAR" \
        -print0
)

# 2) dedupe & sort (optional, but usually desirable)
IFS=$'\n' RUN_DIRS=( $(printf '%s\n' "${RUN_DIRS[@]}" | sort -u) )
unset IFS


# Update NPAR in each INCAR
for dir in "${RUN_DIRS[@]}"; do
    if [ -f "$dir/INCAR" ]; then
        sed -i 's/^\( *NPAR[[:space:]]*=[[:space:]]*\)14/\18/' "$dir/INCAR"
        sed -i 's/NPAR   = 14/NPAR   = 8/' "$dir/INCAR"
        echo "Updated NPAR in $dir/INCAR"
    fi
done

# echo ""
# printf 'Found %d run dirs:\n' "${#RUN_DIRS[@]}"
# echo "Run directories: ${RUN_DIRS[@]}"
# echo ""

# remove the "./" prefix from each directory in RUN_DIRS
RUN_DIRS=("${RUN_DIRS[@]#./}")

# DEBUGGING
echo ""
echo "Found ${#RUN_DIRS[@]} run directories:"
for d in "${RUN_DIRS[@]}"; do
    printf '  • %s\n' "$d"
done
if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "ERROR: no run dirs – check your find pattern!" >&2
    exit 1
fi
echo ""

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

# parallel -j4 --lb \
#     srun --nodes=4 --ntasks-per-node=8 --cpus-per-task=7 --exclusive \
#         --chdir {1} \
#         vasp_std '>' {1}/log.run_sim ::: "${RUN_DIRS[@]}"



N_processes=${#RUN_DIRS[@]}
echo "Number of processes to run in parallel: $N_processes"
echo ""

parallel -j"$N_processes" --verbose --lb \
    srun --nodes=4 --ntasks-per-node=8 --cpus-per-task=7 --exclusive \
        --chdir {1} \
        vasp_std '>' {1}/log.run_sim ::: "${RUN_DIRS[@]}"

#########################################################################
echo "All VASP runs finished at $(date)"