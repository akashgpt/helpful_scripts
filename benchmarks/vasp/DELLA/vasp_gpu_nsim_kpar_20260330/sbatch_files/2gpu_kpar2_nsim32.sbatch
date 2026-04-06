#!/bin/bash
#SBATCH --account=burrows
#SBATCH --job-name=qmd_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=all
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --mail-user=ag5805@princeton.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export PATH=$PATH:/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin

module purge
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
module load intel-mkl/2024.2

touch running_RUN_VASP
echo "Job ID: $SLURM_JOB_ID" > log.run_sim
echo "# =========================================" >> log.run_sim
echo "" >> log.run_sim
srun vasp_std >> log.run_sim
echo "" >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "Job $SLURM_JOB_ID completed at $(date)" >> log.run_sim
rm running_RUN_VASP
touch done_RUN_VASP
