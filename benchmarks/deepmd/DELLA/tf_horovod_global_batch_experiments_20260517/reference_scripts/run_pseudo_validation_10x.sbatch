#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --qos=gpu-test
#SBATCH --job-name=dpgb10x_pseudoval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-14%4

set -euo pipefail

root="/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517"
matrix="${root}/EXPERIMENT_MATRIX_10x_COMPLETED_FOR_PSEUDOVAL.tsv"
validation_root="${root}/pseudo_validation_10x_20260517"
systems_file="${validation_root}/PSEUDO_VALIDATION_SYSTEMS.tsv"
results_root="${validation_root}/results"
image="/scratch/gpfs/BURROWS/akashgpt/softwares/APPTAINER_REPO/deepmd-kit_3.0.0_cuda126.sif"
cases_file="${validation_root}/CASES.tsv"
case_count=15

export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export DP_INFER_BATCH_SIZE=32768
export OMP_NUM_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=4
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/deepmd-kit

write_system_rows() {
	local dataset_id="$1"
	local md_root="$2"
	find "${md_root}" \( -path '*/pre/recal/set_train/deepmd' -o -path '*/pre/recal/set_test/deepmd' \) -type d | LC_ALL=C sort | while IFS= read -r system_dir; do
		if [[ ! -s "${system_dir}/set.000/energy.npy" ]]; then
			continue
		fi
		local rel="${system_dir#${md_root}/}"
		local zone
		local config
		local pre
		local recal
		local split_dir
		local deepmd_dir
		IFS=/ read -r zone config pre recal split_dir deepmd_dir <<< "${rel}"
		local split="${split_dir#set_}"
		local system_id="${dataset_id}_${zone}_${config}_${split}"
		printf '%s\t%s\t%s\t%s\n' "${dataset_id}" "${system_id}" "${split}" "${system_dir}"
	done
}

mkdir -p "${validation_root}" "${results_root}"

{
	printf 'dataset_id\tsystem_id\tsplit\tsystem_dir\n'
	write_system_rows "v7_i34" "/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/md"
	write_system_rows "v8_i16" "/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v4/v8_i16/md"
} > "${systems_file}"

{
	printf 'array_index\tcase_id\tgroup\tstatus\tgpus\tnodes\tsteps\tdecay_steps\tscale_by_worker\tsource_or_output\n'
	tail -n +2 "${matrix}" | awk -F '\t' '{print NR-1 "\t" $0}'
} > "${cases_file}"

echo "JOB_START $(date --iso-8601=seconds)"
echo "ROOT ${root}"
echo "IMAGE ${image}"
echo "ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"
echo "ALCHEMY_FREEZE_COMPRESS_PATTERN /projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev/reference_input_files/submission_scripts/DELLA__TIGER/train_10m.apptr.sh"
echo "ALCHEMY_DP_TEST_PATTERN /projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev/TRAIN_MLMD_scripts/TRAIN_MLMD_LEVEL_3.sh"
nvidia-smi -L || true

system_count="$(tail -n +2 "${systems_file}" | wc -l)"
case_index="${SLURM_ARRAY_TASK_ID}"
echo "CASE_INDEX ${case_index}"
echo "SYSTEM_COUNT ${system_count}"

case_row="$(awk -F '\t' -v idx="${case_index}" 'NR > 1 && $1 == idx {print}' "${cases_file}")"
if [[ -z "${case_row}" ]]; then
	echo "ERROR no case for case index ${case_index}" >&2
	exit 1
fi
IFS=$'\t' read -r array_index case_id group status gpus nodes steps decay_steps scale_by_worker source_or_output <<< "${case_row}"
source_or_output="${source_or_output%$'\r'}"

	if [[ "${source_or_output}" = /* ]]; then
		run_dir="${source_or_output}"
	else
		run_dir="${root}/${source_or_output}"
	fi
	model_dir="${run_dir}/model-compression"
	frozen_model_path="${model_dir}/pv.pb"
	model_path="${model_dir}/pv_comp.pb"
	case_root="${results_root}/${case_id}"
	mkdir -p "${case_root}"
	echo "CASE_START ${case_id} $(date --iso-8601=seconds)"
	if [[ ! -s "${frozen_model_path}" ]]; then
		echo "FREEZE_START ${case_id} ${model_dir}"
		(
			cd "${model_dir}"
			apptainer exec "${image}" env \
				PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
				HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING}" \
				OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
				DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS}" \
				DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS}" \
				dp freeze -o pv.pb
		) > "${case_root}/log.freeze" 2>&1
		echo "FREEZE_END ${case_id} $(date --iso-8601=seconds)"
	else
		echo "FREEZE_SKIP ${case_id} ${frozen_model_path}"
	fi
	if [[ ! -s "${model_path}" ]]; then
		echo "COMPRESS_START ${case_id} ${model_dir}"
		(
			cd "${model_dir}"
			apptainer exec --nv "${image}" env \
				PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
				HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING}" \
				OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
				DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS}" \
				DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS}" \
				dp compress -i pv.pb -o pv_comp.pb -t ../myinput.json
		) > "${case_root}/log.compress" 2>&1
		echo "COMPRESS_END ${case_id} $(date --iso-8601=seconds)"
	else
		echo "COMPRESS_SKIP ${case_id} ${model_path}"
	fi
	if [[ ! -s "${model_path}" ]]; then
		echo "ERROR missing compressed model for ${case_id}: ${model_path}" >&2
		exit 1
	fi
	tail -n +2 "${systems_file}" | while IFS=$'\t' read -r dataset_id system_id split system_dir; do
		out_dir="${case_root}/${system_id}"
		mkdir -p "${out_dir}"
		echo "TEST_START ${case_id} ${dataset_id} ${system_id} $(date --iso-8601=seconds)"
		(
			cd "${out_dir}"
			apptainer exec --nv "${image}" env \
				PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
				HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING}" \
				DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE}" \
				OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
				DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS}" \
				DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS}" \
				TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH}" \
				XLA_FLAGS="${XLA_FLAGS}" \
				dp test -m "${model_path}" -s "${system_dir}" -n 0 -d dp_test
		) > "${out_dir}/log.dp_test" 2>&1
		echo "TEST_END ${case_id} ${dataset_id} ${system_id} $(date --iso-8601=seconds)"
	done
	echo "CASE_END ${case_id} $(date --iso-8601=seconds)"

echo "JOB_END $(date --iso-8601=seconds)"
