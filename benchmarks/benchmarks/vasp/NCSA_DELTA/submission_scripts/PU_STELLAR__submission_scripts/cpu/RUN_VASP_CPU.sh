#!/bin/bash
## VASP 6.6.0 CPU -- PU Stellar -- 96 cores (12 MPI x 8 OMP)
## Status: WORKING (elapsed ~2130 s on benchmark system)
#SBATCH --job-name=qmd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1:05:00


##################  Environment  ########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VASP_BIN=/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std
module purge
module load intel-oneapi/2024.2
module load intel-mpi/oneapi/2021.13
module load intel-mkl/2024.2
module load hdf5/oneapi-2024.2/1.14.4


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
srun "$VASP_BIN" >> log.run_sim
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
