#!/bin/bash
# =====================================================================
# VALIDATED H200 single-GPU VASP submission template (NCSA Delta).
#
# Provenance: verbatim (job-name genericised) from the empirically
# successful run
#   He_MgSiO3/sim_data_ML/setup_MLMD/benchmarking_tests/ENCUT_test/
#     72_MgSiO3__360_atoms__and_He/71MgSiO3_5He/ENCUT_0800/sub_vasp_h200.sh
# That 360-atom job completed exit 0 on H200 node gpue06 (gpuH200x8).
#
# KEY POINT: the SAME binary vasp_std__NCSA_DELTA_GPU runs on BOTH
# gpuA100x4 (A100/sm_80, native) and gpuH200x8 (H200/sm_90, via the
# NVHPC OpenACC runtime's launch-time JIT). To target A100 instead,
# only change `#SBATCH --partition` to gpuA100x4 — nothing else.
# See ../../H200_VASP_GPU__binary_and_submission_reference__20260518.md
# =====================================================================
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name=vasp_h200
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=180G
#SBATCH --time=04:00:00

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VASP_BIN=/work/nvme/bguf/akashgpt/softwares/vasp/vasp.6.6.0.gpu/bin/vasp_std__NCSA_DELTA_GPU

module reset
module load nvhpc-hpcx-cuda12/25.3 intel-oneapi-mkl/2024.2.2
ulimit -s unlimited

touch running_RUN_VASP
rm -f done_RUN_VASP failed_RUN_VASP

if [ -f log.run_sim ]; then
	mv log.run_sim "old.log.run_sim.${SLURM_JOB_ID}"
fi

echo "Job ID: $SLURM_JOB_ID" > log.run_sim
echo "Host: $(hostname)" >> log.run_sim
echo "Started at $(date)" >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "" >> log.run_sim

set +e
mpirun --bind-to none -np 1 "$VASP_BIN" >> log.run_sim 2>&1
vasp_status=$?
set -e

echo "" >> log.run_sim
echo "# =========================================" >> log.run_sim
echo "VASP exit status: $vasp_status" >> log.run_sim
echo "Finished at $(date)" >> log.run_sim
echo "# =========================================" >> log.run_sim

rm -f running_RUN_VASP

if [ "$vasp_status" -eq 0 ]; then
	touch done_RUN_VASP
else
	touch failed_RUN_VASP
	exit "$vasp_status"
fi
