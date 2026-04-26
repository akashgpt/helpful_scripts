#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=pt_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1:00:00

# ===================================================================
# One-shot prep job:
#   1. Run a short (100-step) PT training to produce a .pth model
#   2. Freeze and compress it.
# The resulting model_comp.pth is shared by both variant_PT and
# variant_PT_KK benchmark runs.
#
# Only the architecture (se_e2_a, neuron=[25,50,100]/[240,240,240])
# matters for LAMMPS inference timing; weights can be untrained.
# ===================================================================

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
export MPICH_GPU_SUPPORT_ENABLED=1

SHARED_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SHARED_DIR"
echo "Working in: $(pwd)"

mkdir -p model-compression

echo "=== [$(date)] Training PT model (100 steps) ==="
dp --pt train pt_train_input.json 2>&1 | tee log.pt_train

echo "=== [$(date)] Freezing PT model ==="
dp --pt freeze -o model.pth 2>&1 | tee log.pt_freeze

echo "=== [$(date)] Compressing PT model ==="
dp --pt compress -i model.pth -o model_comp.pth 2>&1 | tee log.pt_compress

ls -la model.pth model_comp.pth
echo "=== [$(date)] Done ==="
