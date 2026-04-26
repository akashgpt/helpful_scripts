#!/bin/bash
# Shared module/env loader for all benchmark sub scripts on Delta.
# Sourced by each variant_*/sub.*.sh after it sets BACKEND=(TF|PT).
set -e
module purge
module load PrgEnv-gnu
module load gcc-native/13.2
module load cray-mpich
module load cudatoolkit/25.3_12.8
module load fftw/3.3.10-gcc13.3.1
module load miniforge3-python
eval "$(conda shell.bash hook)"
if [ "${BACKEND}" = "TF" ]; then
	conda activate ALCHEMY_env
else
	conda activate ALCHEMY_env__PT
fi
export PYTHONNOUSERSITE=1
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
