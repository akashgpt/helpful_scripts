#!/bin/bash
#SBATCH --account=astro
#SBATCH --partition=pu
#SBATCH --qos=pu-short-stellar
#SBATCH --job-name=mgsiofe_conv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --mail-user=ag5805@princeton.edu
#SBATCH --output=/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test/slurm_mgsiofe_conv_%A_%a.out
#SBATCH --error=/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test/slurm_mgsiofe_conv_%A_%a.err
#SBATCH --array=0-10%8

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
VASP_BIN=/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std

module purge
module load intel-oneapi/2024.2
module load intel-mpi/oneapi/2021.13
module load intel-mkl/2024.2
module load hdf5/oneapi-2024.2/1.14.4

MANIFEST=/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test/benchmark_manifest__MgSiOFe__KSPACING_ENCUT.tsv
ROWS=$(awk -F '\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR > 1 && ($1 % 11) == idx { print }' "$MANIFEST")
if [[ -z "$ROWS" ]]; then
	echo "No manifest rows for array index $SLURM_ARRAY_TASK_ID" >&2
	exit 1
fi

while IFS=$'\t' read -r RUN_ID FAMILY CASE_NAME LABEL VALUE RUN_DIR; do
	if [[ -z "${RUN_ID:-}" ]]; then
		continue
	fi

	cd "$RUN_DIR"
	touch running_RUN_VASP
	rm -f done_RUN_VASP
	if [[ -f log.run_sim ]]; then
		cat log.run_sim >> old.log.run_sim
	fi

	{
		echo "Job ID: $SLURM_JOB_ID"
		echo "Array task: $SLURM_ARRAY_TASK_ID"
		echo "Run ID: $RUN_ID"
		echo "Family: $FAMILY"
		echo "Case: $CASE_NAME"
		echo "Label: $LABEL"
		echo "Value: $VALUE"
		echo "Run dir: $RUN_DIR"
		echo "Started at $(date)"
		echo "# ========================================="
		echo ""
	} > log.run_sim

	srun "$VASP_BIN" >> log.run_sim

	{
		echo ""
		echo "# ========================================="
		echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID completed run $RUN_ID at $(date)"
		echo "# ========================================="
		echo ""
	} >> log.run_sim

	rm -f running_RUN_VASP
	touch done_RUN_VASP
done <<< "$ROWS"
