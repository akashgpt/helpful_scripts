#!/bin/bash
#SBATCH --account=burrows
#SBATCH --job-name=qmd              # create a short name for your job
#SBATCH --nodes=2                   # node count
#SBATCH --ntasks-per-node=14        # total number of tasks per node
#SBATCH --cpus-per-task=8           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G            # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ag5805@princeton.edu


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export PATH=$PATH:/scratch/gpfs/BURROWS/akashgpt/softwares/vasp.6.4.3/bin

module purge

## Della | Stellar | Tiger3 (all same)
module load intel-oneapi/2024.2
module load intel-mpi/oneapi/2021.13
module load intel-mkl/2024.2
module load hdf5/oneapi-2024.2/1.14.4




##############################
###### Log: log.runsim #######
# record job id in log.run_sim
touch running_RUN_VASP

if [ -f log.run_sim ]; then
    rm log.run_sim
fi
echo "Job ID: $SLURM_JOB_ID" > log.run_sim
echo "# =========================================" >> log.run_sim
echo "" >> log.run_sim

# ****************************
# ****************************
# ~~~~~~ CALLING VASP ~~~~~~~~
srun vasp_std >> log.run_sim
# ****************************
# ****************************

echo "" >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "Job $SLURM_JOB_ID completed at" `date ` >> log.run_sim

rm running_RUN_VASP
touch done_RUN_VASP
###### Log: log.runsim #######
##############################
