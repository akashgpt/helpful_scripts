#!/bin/bash
## VASP 6.6.0 GPU -- PU Stellar -- 2 GPUs (KPAR=2)
## Status: WORKING (elapsed ~763 s on benchmark system, ~2.0x speedup)
## Notes:  Requires KPAR=2 in INCAR.
#SBATCH --job-name=qmd_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --mem=128G
#SBATCH --time=1:00:00


##################  Environment  ########################################
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VASP_BIN=/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin/vasp_std
module purge
module load nvhpc/25.5
module load openmpi/cuda-12.9/nvhpc-25.5/4.1.8
module load intel-mkl/2024.2


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
