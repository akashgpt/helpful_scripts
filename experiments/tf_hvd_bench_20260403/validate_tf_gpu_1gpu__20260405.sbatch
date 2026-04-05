#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=tfgpuok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=00:05:00

set -euo pipefail
export PS1="${PS1-}"

ENV_PATH="${ENV_PATH_OVERRIDE:-/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/dp_tf15_src_hvd_fixed_20260405}"

module purge
module load gcc/11
module load openmpi/gcc/4.1.6
module load anaconda3/2025.12
eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"

export PYTHONNOUSERSITE=1
export TF_CPP_MIN_LOG_LEVEL=0
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH-}"

python - <<'PY'
import tensorflow as tf
print("TF_VERSION", tf.__version__)
print("GPUS", tf.config.list_physical_devices("GPU"))
PY
