#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=bntf_N360_G4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0:30:00

BACKEND=TF
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARED="$(cd "$HERE/../shared" && pwd)"
source "$SHARED/common_env.sh"

CONF="$SHARED/conf.lmp"
MODEL="/work/nvme/bguf/akashgpt/qmd_data/He_MgSiO3/sim_data_ML/v1_i1/train/model-compression/pv_comp.pb"
NSTEPS_RELAX=200
NSTEPS_PROD=5000

cp -f "$CONF"                   "$HERE/conf.lmp"
cp -f "$SHARED/in.lammps.bench" "$HERE/in.lammps.bench"

cd "$HERE"
echo "=== [$(date)] TF N=360 G=4 ==="
echo "MODEL=$MODEL"
srun -n 4 lmp_plmd_ncsa_delta \
	-var MODEL "$MODEL" \
	-var NSTEPS_relax "$NSTEPS_RELAX" \
	-var NSTEPS_4training "$NSTEPS_PROD" \
	-in in.lammps.bench
echo "=== [$(date)] Done ==="
