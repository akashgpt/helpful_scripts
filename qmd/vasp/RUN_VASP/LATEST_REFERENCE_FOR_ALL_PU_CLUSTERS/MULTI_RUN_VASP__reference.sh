#!/bin/bash
#SBATCH --account=burrows
#SBATCH --job-name=multi_run        # create a short name for your job
#SBATCH --output=multi_run_%j.out   # output file name with job ID
#SBATCH --error=multi_run_%j.err    # error file name with job ID
#SBATCH --nodes=2                   # node count
#SBATCH --ntasks-per-node=14        # total number of tasks per node
#SBATCH --cpus-per-task=8           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G            # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ag5805@princeton.edu
##SBATCH --constraint=intel         # *** for Della only ***


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
# list your run‐directories here (or build the list dynamically)
# RUN_DIRS=(T10400 T11700 T14300 T15600)
# RUN_DIRS is all the directories that contain the VASP INCAR files
for dir in */; do
    if [ -f "$dir/INCAR" ]; then
        RUN_DIRS+=("$dir")
    fi
done
if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
    echo "No run directories found (keyword: $to_RUN_keyword, INCAR required). Exiting."
    exit 0
fi
echo "Run directories: ${RUN_DIRS[@]}"

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

parallel -j4 --lb \
    srun --nodes=2 --ntasks-per-node=12 --cpus-per-task=8 --exclusive \
        --chdir {1} \
        vasp_std '>' {1}/log.run_sim ::: "${RUN_DIRS[@]}"

#########################################################################
echo "All VASP runs finished at $(date)"