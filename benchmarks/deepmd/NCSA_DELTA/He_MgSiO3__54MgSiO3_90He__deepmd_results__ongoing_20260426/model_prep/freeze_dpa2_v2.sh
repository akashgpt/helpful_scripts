#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=freeze_dpa2_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0:30:00

# v2 freeze: run `dp --pt freeze` from inside the original training dir,
# where the `checkpoint` pointer file + model.ckpt-1000000.pt already
# exist. v1 staged the .pt to a workdir but didn't create the checkpoint
# pointer that dp expects.

set -e
module purge
module load PrgEnv-gnu
module load gcc-native/13.2
module load cray-mpich
module load cudatoolkit/25.3_12.8
module load fftw/3.3.10-gcc13.3.1
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate ALCHEMY_env__PT
export PYTHONNOUSERSITE=1

TBENCH=/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/training_bench
SHARED=/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/He_MgSiO3__54MgSiO3_90He/shared

cd "$TBENCH/variant_train_dpa2_PT"
echo "=== [$(date)] Freezing DPA-2 baseline (cwd: $PWD) ==="
dp --pt freeze -o "$SHARED/model_dpa2.pth" 2>&1 | tee "$SHARED/log.dpa2_freeze_v2"
ls -la "$SHARED/model_dpa2.pth"
echo "=== [$(date)] Done ==="
