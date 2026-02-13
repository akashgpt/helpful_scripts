#!/usr/bin/env bash
##SBATCH --account=burrows
#SBATCH --job-name=vasp_multi      # short name for this multi‐run job
#SBATCH --nodes=8                  # total nodes (4 jobs × 2 nodes each)
#SBATCH --ntasks-per-node=12       # MPI ranks per node
#SBATCH --cpus-per-task=8          # threads per rank
#SBATCH --mem-per-cpu=1G
#SBATCH --time=24:00:00
#SBATCH --mail-user=ag5805@princeton.edu


##################  Environment  ########################################

# ensure OMP/MPI see the right thread count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK   # some MPI wrappers honour this

export PATH=$PATH:$HOME/softwares/vasp.6.3.2/bin


module purge
module load intel/2021.1 intel-mkl/2021.1.1 intel-mpi/intel/2021.1.1
module load hdf5/intel-2021.1/intel-mpi/1.10.6

##################  Job list  ###########################################
# list your run‐directories here (or build the list dynamically)
# RUN_DIRS=(T10400 T11700 T14300 T15600)
# RUN_DIRS is all the directories that contain the VASP INCAR files
for dir in */; do
    if [ -f "$dir/INCAR" ]; then
        RUN_DIRS+=("$dir")
    fi
done
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

parallel -j4 --lb \
    srun --nodes=2 --ntasks-per-node=12 --cpus-per-task=8 --exclusive \
        --chdir {1} \
        vasp_std '>' {1}/log.runsim ::: "${RUN_DIRS[@]}"

#########################################################################
echo "All VASP runs finished at $(date)"