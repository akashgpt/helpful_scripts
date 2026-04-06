#!/bin/bash
#SBATCH --account=burrows
#SBATCH --job-name=qmd_gpu           # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks-per-node=1          # 1 MPI task per GPU
#SBATCH --cpus-per-task=16           # cpu-cores per task (for OpenMP threads)
#SBATCH --gres=gpu:a100:1                 # number of A100 GPUs per node
#SBATCH --partition=all              # Della GPU partition
#SBATCH --mem=64G                    # total memory per node
#SBATCH --time=1:00:00              # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ag5805@princeton.edu


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# GPU-compiled VASP 6.6.0
export PATH=$PATH:/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin

module purge

## Della GPU: same modules used to compile VASP
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
module load intel-mkl/2024.2




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
