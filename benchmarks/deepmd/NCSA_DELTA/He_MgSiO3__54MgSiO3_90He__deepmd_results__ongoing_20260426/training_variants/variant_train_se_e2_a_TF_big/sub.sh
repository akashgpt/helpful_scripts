#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=train_se_tf_big
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=64G
#SBATCH --time=48:00:00

set -e
module purge
module load PrgEnv-gnu
module load gcc-native/13.2
module load cray-mpich
module load cudatoolkit/25.3_12.8
module load fftw/3.3.10-gcc13.3.1
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate ALCHEMY_env
export PYTHONNOUSERSITE=1
export MPICH_GPU_SUPPORT_ENABLED=1

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARED="$(cd "$HERE/../shared" && pwd)"
cp -f "$SHARED/train_se_e2_a_big.json" "$HERE/input.json"
cd "$HERE"

echo "=== [$(date)] TF training (se_e2_a BIG ~10x params, 1,000,000 steps) ==="
dp --tf train input.json 2>&1 | tee log.train
echo "=== [$(date)] Done ==="
