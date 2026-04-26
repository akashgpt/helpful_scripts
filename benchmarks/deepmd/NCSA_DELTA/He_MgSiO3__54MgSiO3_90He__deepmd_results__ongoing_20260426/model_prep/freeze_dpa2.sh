#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=freeze_dpa2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0:30:00

# One-shot: stage DPA-2 checkpoint from the running 1M-step training,
# freeze it to model_dpa2.pth for LAMMPS inference benchmarking.
# DPA-2 does not support `dp compress` (attention can't be tabulated),
# so no compression step.

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

SHARED="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
WORK="$SHARED/dpa2_freeze_workdir"
mkdir -p "$WORK"
cd "$WORK"

# Stage checkpoint and config
cp -f "$SHARED/dpa2_ckpt-26000.pt" model.ckpt.pt
cp -f "$SHARED/../training_bench/shared/train_dpa2.json" input.json

echo "=== [$(date)] Freezing DPA-2 checkpoint ==="
dp --pt freeze -o model_dpa2.pth 2>&1 | tee "$SHARED/log.dpa2_freeze"

cp -f model_dpa2.pth "$SHARED/model_dpa2.pth"
ls -la "$SHARED/model_dpa2.pth"
echo "=== [$(date)] Done ==="
