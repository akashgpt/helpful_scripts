#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=bench_pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0:30:00

# LAMMPS benchmark — PT backend, no KOKKOS (plain lmp).

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
export OMP_NUM_THREADS=1

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARED="$(cd "$HERE/../shared" && pwd)"
MODEL="$SHARED/model_comp.pth"

cp -f "$SHARED/conf.lmp" "$HERE/conf.lmp"
cp -f "$SHARED/in.lammps.bench" "$HERE/in.lammps.bench"

cd "$HERE"
echo "=== [$(date)] PT (lmp) benchmark ==="
echo "MODEL=$MODEL"

srun -n 1 lmp -var MODEL "$MODEL" -in in.lammps.bench
echo "=== [$(date)] Done ==="
