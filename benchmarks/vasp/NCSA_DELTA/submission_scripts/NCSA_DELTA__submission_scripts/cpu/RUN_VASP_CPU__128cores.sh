#!/bin/bash
## VASP 6.6.0 CPU -- NCSA DELTA -- 128 cores (16 MPI x 8 OMP), 1 node
## Status: WORKING (elapsed ~2256 s on benchmark system)
#SBATCH --account=bguf-delta-cpu
#SBATCH --job-name=qmd
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1:05:00


##################  Environment  ########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VASP_BIN=/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std
module reset
module load PrgEnv-gnu cray-hdf5-parallel
ulimit -s unlimited


##############################
###### Log: log.runsim #######
touch running_RUN_VASP
rm -f done_RUN_VASP

if [ -f log.run_sim ]; then
	cat log.run_sim >> old.log.run_sim
fi
echo "Job ID: $SLURM_JOB_ID" > log.run_sim
echo "Started at $(date)" >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "" >> log.run_sim

# ****************************
# ~~~~~~ CALLING VASP ~~~~~~~~
srun --cpu-bind=cores --distribution=block:block "$VASP_BIN" >> log.run_sim
# ****************************

echo "" >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "Job $SLURM_JOB_ID completed at" `date ` >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "" >> log.run_sim

rm running_RUN_VASP
touch done_RUN_VASP
###### Log: log.runsim #######
##############################
