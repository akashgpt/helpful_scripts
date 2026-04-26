#!/bin/bash
#SBATCH --account=astro
#SBATCH --partition=pu
#SBATCH --qos=pu-short-stellar
#SBATCH --job-name=mgsiofe_p5
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --mail-user=ag5805@princeton.edu
#SBATCH --output=/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test/slurm_mgsiofe_parallel5_%j.out
#SBATCH --error=/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test/slurm_mgsiofe_parallel5_%j.err

set -euo pipefail

MANIFEST=/scratch/gpfs/BURROWS/akashgpt/qmd_data/MgSiOFe__R2SCAN/test/benchmark_manifest__MgSiOFe__KSPACING_ENCUT.tsv
VASP_BIN=/scratch/gpfs/BURROWS/akashgpt/softwares/vasp/vasp.6.6.0/bin/vasp_std
MAX_PARALLEL_RUNS=5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module purge
module load intel-oneapi/2024.2
module load intel-mpi/oneapi/2021.13
module load intel-mkl/2024.2
module load hdf5/oneapi-2024.2/1.14.4

run_one() {
	local row="$1"
	local run_id family case_name label value run_dir

	IFS=$'\t' read -r run_id family case_name label value run_dir <<< "$row"
	cd "$run_dir"

	rm -f running_RUN_VASP done_RUN_VASP
	touch running_RUN_VASP
	if [[ -f log.run_sim ]]; then
		cat log.run_sim >> old.log.run_sim
	fi

	{
		echo "Parent job ID: $SLURM_JOB_ID"
		echo "Run ID: $run_id"
		echo "Family: $family"
		echo "Case: $case_name"
		echo "Label: $label"
		echo "Value: $value"
		echo "Run dir: $run_dir"
		echo "Started at $(date)"
		echo "# ========================================="
		echo ""
	} > log.run_sim

	srun --exclusive --nodes=1 --ntasks=96 --cpus-per-task=1 "$VASP_BIN" >> log.run_sim

	{
		echo ""
		echo "# ========================================="
		echo "Parent job $SLURM_JOB_ID completed run $run_id at $(date)"
		echo "# ========================================="
		echo ""
	} >> log.run_sim

	rm -f running_RUN_VASP
	touch done_RUN_VASP
}

mapfile -t rows < <(awk -F '\t' 'NR > 1 { print }' "$MANIFEST")

active=0
for row in "${rows[@]}"; do
	run_one "$row" &
	active=$((active + 1))

	if (( active >= MAX_PARALLEL_RUNS )); then
		wait
		active=0
	fi
done

if (( active > 0 )); then
	wait
fi
