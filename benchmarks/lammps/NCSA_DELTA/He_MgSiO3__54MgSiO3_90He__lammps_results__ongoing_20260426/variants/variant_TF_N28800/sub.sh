#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=bench_tf_28800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0:30:00

BACKEND=TF
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARED="$(cd "$HERE/../shared" && pwd)"
source "$SHARED/common_env.sh"

CONF="$SHARED/conf_28800.lmp"
MODEL="/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/v1_i1/train/model-compression/pv_comp.pb"
NSTEPS_RELAX=20
NSTEPS_PROD=100

cp -f "$CONF"                  "$HERE/conf.lmp"
cp -f "$SHARED/in.lammps.bench" "$HERE/in.lammps.bench"

cd "$HERE"
echo "=== [$(date)] TF (lmp_plmd_ncsa_delta) N=28800 ==="
echo "MODEL=$MODEL"
srun -n 1 lmp_plmd_ncsa_delta \
	-var MODEL "$MODEL" \
	-var NSTEPS_relax "$NSTEPS_RELAX" \
	-var NSTEPS_4training "$NSTEPS_PROD" \
	-in in.lammps.bench
echo "=== [$(date)] Done ==="
