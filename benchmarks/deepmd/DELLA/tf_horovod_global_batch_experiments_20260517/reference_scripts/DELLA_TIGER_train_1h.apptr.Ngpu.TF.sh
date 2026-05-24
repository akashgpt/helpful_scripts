#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name=dp_tf_ngpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=160G
#SBATCH --time=01:00:00

# For a 1-GPU TF run, optionally change this to: #SBATCH --ntasks=1
# For a 1-GPU run, change this to: #SBATCH --gres=gpu:a100:1
# The runtime rank count follows the detected GPU allocation.

# ALCHEMY Della/Tiger Slurm DeePMD multi-GPU training template.
# - Copy this file into a training directory and submit it from there with sbatch.
# - Runs one scheduler slice; TRAIN_MLMD_LEVEL_2 owns chained resubmission.
# - Checkpoint-continuation safety uses conservative checkpoint choice and rollback validation.
# - Keep learning_rate.scale_by_worker="none" unless explicitly testing alternatives.

set -euo pipefail

# Resolve the working directory from the copied script path so all slices run in the same training folder.
script_path="${ALCHEMY_RESTART_SCRIPT_PATH:-${PWD}/train_1h.apptr.Ngpu.TF.sh}"
script_dir="$(dirname "${script_path}")"
cd "${script_dir}"

case_name="${ALCHEMY_RESTART_CASE:-train_1h_apptr_Ngpu_TF}"
job_id="${SLURM_JOB_ID:-manual}"

# Runtime/threading knobs used by DeePMD, TensorFlow/PyTorch, HDF5, and NCCL.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE:-32768}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-2}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=/opt/deepmd-kit}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Infer distributed-training topology from scheduler variables and local GPU count.
NODE_COUNT="${SLURM_JOB_NUM_NODES:-1}"

# Keep this template usable for 1, 2, 4, ... GPUs by deriving ranks from the actual allocation.
detect_gpus_per_node() {
	local detected="${ALCHEMY_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"

	if [ -z "${detected}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]; then
		detected="$(printf "%s" "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')"
	fi
	if [ -z "${detected}" ] && [ -f "${script_path}" ]; then
		detected="$(awk -F: '/^#SBATCH --gres=.*gpu/ {print $NF; exit}' "${script_path}")"
	fi
	if [ -z "${detected}" ] && command -v nvidia-smi >/dev/null 2>&1; then
		detected="$(nvidia-smi -L 2>/dev/null | wc -l)"
	fi

	detected="$(printf "%s" "${detected}" | awk 'match($0, /[0-9]+/) {print substr($0, RSTART, RLENGTH); exit}')"
	if [ -z "${detected}" ] || [ "${detected}" -le 0 ]; then
		detected=1
	fi
	printf "%s\n" "${detected}"
}

GPUS_PER_NODE="$(detect_gpus_per_node)"
TRAIN_RANKS=$(( NODE_COUNT * GPUS_PER_NODE ))
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n 1)"
MASTER_PORT="${ALCHEMY_MASTER_PORT:-$((29500 + RANDOM % 1000))}"

# Load the Della/Tiger module and conda runtime needed before calling dp/srun.
load_runtime() {
	set +u
	if command -v module >/dev/null 2>&1; then
		module purge >/dev/null 2>&1 || true
		module load gcc-toolset/14 >/dev/null 2>&1 || true
		module load openmpi/gcc/4.1.6 >/dev/null 2>&1 || true
		module load cudatoolkit/12.8 >/dev/null 2>&1 || true
		module load fftw/gcc/openmpi-4.1.6/3.3.10 >/dev/null 2>&1 || true
		module load anaconda3/2025.12 >/dev/null 2>&1 || true
	fi
	# Activate the DeePMD environment after scheduler/site modules are available.
	if command -v conda >/dev/null 2>&1; then
		eval "$(conda shell.bash hook 2>/dev/null || true)"
		conda activate "${ALCHEMY_CONDA_ENV:-ALCHEMY_env}" >/dev/null 2>&1 || true
	fi
	set -u
}

# Read the last numeric training step from lcurve.out; missing files mean no progress yet.
latest_step() {
	if [ ! -s lcurve.out ]; then
		printf "0\n"
		return 0
	fi
	awk '($1 ~ /^[0-9]+$/) {step=$1} END {if (step=="") step=0; print step}' lcurve.out
}



# Check the stable per-slice training log instead of guessing completion from walltime.
training_finished_in_log() {
	local log_file="$1"
	[ -s "${log_file}" ] && grep -Eiq 'finished[[:space:]_-]+training|training[[:space:]_-]+finished' "${log_file}"
}

# Choose a restart checkpoint numerically, preferring the second-latest checkpoint when available.
latest_checkpoint_prefix() {
	local selected
	selected=$(python - <<'PY'
from pathlib import Path
import re

checkpoints = []
for path in Path("model-compression").glob("model.ckpt-*.index"):
	match = re.fullmatch(r"model\.ckpt-(\d+)\.index", path.name)
	if match is not None:
		checkpoints.append((int(match.group(1)), path.name[:-6]))

checkpoints.sort()
if len(checkpoints) >= 2:
	print("model-compression/" + checkpoints[-2][1])
elif len(checkpoints) == 1:
	print("model-compression/" + checkpoints[-1][1])
PY
)
	if [ -z "${selected}" ]; then
		return 1
	fi
	printf "%s\n" "${selected}"
}

# Extract the numeric step embedded in a DeePMD checkpoint name.
checkpoint_step_from_prefix() {
	local prefix="${1:-}"
	printf "%s\n" "${prefix}" | awk '
		match($0, /model\.ckpt-[0-9]+/) {
			value = substr($0, RSTART + 11, RLENGTH - 11)
			print value
			exit
		}
	'
}

# Refuse unexpectedly stale checkpoints so a restart cannot silently jump far backwards.
validate_restart_prefix() {
	local prefix="${1:-}"
	local current_step="${2:-0}"
	local checkpoint_step
	local max_rollback="${ALCHEMY_RESTART_MAX_ROLLBACK_STEPS:-2000}"
	checkpoint_step="$(checkpoint_step_from_prefix "${prefix}")"
	if [ -z "${checkpoint_step}" ]; then
		echo "Could not parse checkpoint step from ${prefix}" >&2
		return 1
	fi
	if [ "${current_step}" -gt 0 ] && [ "${checkpoint_step}" -lt $((current_step - max_rollback)) ]; then
		echo "Refusing restart from ${prefix}: checkpoint step ${checkpoint_step} is too far behind current lcurve step ${current_step}." >&2
		echo "Set ALCHEMY_RESTART_MAX_ROLLBACK_STEPS to override if this rollback is intentional." >&2
		return 1
	fi
	return 0
}


# Main workflow begins here: prepare runtime and checkpoint directory.
load_runtime
mkdir -p model-compression
train_log="train.${case_name}.${job_id}.out"
rm -f "${train_log}"

# Emit a compact run header so scheduler output can be audited later.
echo "=========================================="
echo "JOB_START $(date --iso-8601=seconds)"
echo "CASE ${case_name}"
echo "BACKEND TF"
echo "CHECKPOINT_STRATEGY second_latest_when_available"
echo "CHAIN_MODE single_slice_level2_managed"
echo "SCRIPT_PATH ${script_path}"
echo "TRAIN_LOG ${train_log}"
echo "SLURM_JOB_ID ${job_id}"
echo "Nodes: ${NODE_COUNT}; GPUs/node: ${GPUS_PER_NODE}; training ranks: ${TRAIN_RANKS}"
echo "Current step: $(latest_step)"
echo "=========================================="
nvidia-smi -L || true

# Build the DeePMD train command, restarting from the selected checkpoint if one exists.
current_step="$(latest_step)"
if restart_prefix=$(latest_checkpoint_prefix); then
	validate_restart_prefix "${restart_prefix}" "${current_step}"
	echo "TRAIN_MODE restart ${restart_prefix}"
	train_args=(dp train --mpi-log=workers myinput.json --restart "${restart_prefix}")
else
	echo "TRAIN_MODE fresh"
	train_args=(dp train --mpi-log=workers myinput.json)
fi

# Allow walltime/interruption to reach the completion-marker check below.
set +e
# Launch TensorFlow DeePMD/Horovod with one MPI rank per GPU.
srun --mpi=pmix --ntasks="${TRAIN_RANKS}" --cpu-bind=cores --kill-on-bad-exit=1 \
	apptainer exec --nv "${ALCHEMY_DEEPMD_IMAGE:-${APPTAINER_REPO}/deepmd-kit_3.0.0_cuda126.sif}" env \
	PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
	HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING}" \
	DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE}" \
	OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
	DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS}" \
	DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS}" \
	TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH}" \
	XLA_FLAGS="${XLA_FLAGS}" \
	NCCL_DEBUG="${NCCL_DEBUG}" \
	"${train_args[@]}" 2>&1 | tee -a "${train_log}"
train_status=${PIPESTATUS[0]}
set -e

echo "TRAIN_EXIT_STATUS ${train_status}"

# DeePMD prints "finished training" only after it exits the training loop cleanly.
# Timed-out slices should not freeze/compress; Level 2 can submit the next slice.
if ! training_finished_in_log "${train_log}"; then
	step_after_train="$(latest_step)"
	echo "TRAIN_SLICE_ENDED_BEFORE_DEEPMD_FINISHED step=${step_after_train} log=${train_log}"
	touch TRAINING_INCOMPLETE
	rm -f TRAINING_COMPLETE FREEZE_COMPRESS_DONE
	exit 0
fi
touch TRAINING_COMPLETE
rm -f TRAINING_INCOMPLETE

echo "=========================================="
echo "DeePMD reported finished training. Starting freeze + compress at $(date)"
echo "=========================================="

# Freeze/compress only after DeePMD reports clean training completion.
( cd model-compression && apptainer exec "${ALCHEMY_DEEPMD_IMAGE:-${APPTAINER_REPO}/deepmd-kit_3.0.0_cuda126.sif}" dp freeze -o pv.pb )
apptainer exec --nv "${ALCHEMY_DEEPMD_IMAGE:-${APPTAINER_REPO}/deepmd-kit_3.0.0_cuda126.sif}" dp compress -i model-compression/pv.pb -o model-compression/pv_comp.pb
echo "Output: model-compression/pv_comp.pb"
touch FREEZE_COMPRESS_DONE

echo "=========================================="
echo "JOB_END $(date --iso-8601=seconds) step=$(latest_step)"
echo "=========================================="
