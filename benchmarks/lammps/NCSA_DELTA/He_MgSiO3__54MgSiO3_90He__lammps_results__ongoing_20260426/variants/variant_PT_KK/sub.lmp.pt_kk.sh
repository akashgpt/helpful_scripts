#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=bench_pt_kk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:30:00

# LAMMPS benchmark — PT backend + KOKKOS (lmp_kk).
# KOKKOS activates integrators / neighbor / comm on GPU;
# pair_style deepmd itself has no KOKKOS variant so it stays on
# DeepMD's native PT-GPU kernels.

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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARED="$(cd "$HERE/../shared" && pwd)"
MODEL="$SHARED/model_comp.pth"

cp -f "$SHARED/conf.lmp" "$HERE/conf.lmp"
cp -f "$SHARED/in.lammps.bench" "$HERE/in.lammps.bench"

cd "$HERE"
echo "=== [$(date)] PT + KOKKOS (lmp_kk) benchmark ==="
echo "MODEL=$MODEL"

# -k on g 1   : turn KOKKOS on, 1 GPU
# -sf kk      : suffix kokkos (apply kk variants where available)
# -pk kokkos ...: KOKKOS package options
srun -n 1 lmp_kk -k on g 1 -sf kk -pk kokkos newton on neigh half \
     -var MODEL "$MODEL" -in in.lammps.bench
echo "=== [$(date)] Done ==="
