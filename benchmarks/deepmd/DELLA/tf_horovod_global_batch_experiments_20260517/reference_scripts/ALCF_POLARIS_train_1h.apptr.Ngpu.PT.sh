#!/bin/bash -l
#PBS -A CoreCollapseModel
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -j oe

# ALCHEMY Polaris/PBS DeePMD multi-GPU training template.
# - Copy this file into a training directory and submit it from there with qsub.
# - select=N reserves N whole Polaris nodes; each Polaris node has 4 GPUs.
# - Runs one scheduler slice; TRAIN_MLMD_LEVEL_2 owns chained resubmission.
# - Checkpoint-continuation safety uses conservative checkpoint choice and rollback validation.
# - Keep learning_rate.scale_by_worker="none" unless explicitly testing alternatives.

set -euo pipefail

# Resolve the working directory from the copied script path so all slices run in the same training folder.
cd "${PBS_O_WORKDIR}"

case_name="${ALCHEMY_RESTART_CASE:-train_1h_apptr_Ngpu_PT}"
script_path="${ALCHEMY_RESTART_SCRIPT_PATH:-${PBS_O_WORKDIR}/train_1h.apptr.Ngpu.PT.sh}"
script_dir="$(dirname "${script_path}")"
job_id="${PBS_JOBID:-manual}"
job_tag="${job_id%%.*}"

# Load the minimal Polaris runtime before activating conda and Apptainer.
setup_polaris_training_runtime() {
	if ! command -v module >/dev/null 2>&1; then
		echo "Environment modules are not available on this shell." >&2
		return 1
	fi

	module reset >/dev/null 2>&1 || true
	module load conda >/dev/null 2>&1 || true
	module load apptainer >/dev/null 2>&1 || true
	# Runtime/threading knobs used by DeePMD, TensorFlow/PyTorch, HDF5, and NCCL.
	export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
}

# Main workflow begins here: prepare runtime, conda, and Apptainer.
setup_polaris_training_runtime

# Activate the DeePMD environment after scheduler/site modules are available.
if command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook 2>/dev/null || true)"
	conda activate "${ALCHEMY_CONDA_ENV:-ALCHEMY_env}" >/dev/null 2>&1 || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
	echo "Apptainer not found in PATH." >&2
	exit 1
fi

if [ -z "${APPTAINER_REPO:-}" ]; then
	echo "APPTAINER_REPO is not set." >&2
	exit 1
fi

# Resolve the DeePMD Apptainer image; callers can override this with ALCHEMY_DEEPMD_IMAGE.
image="${ALCHEMY_DEEPMD_IMAGE:-${APPTAINER_REPO}/deepmd-kit_3.0.0_cuda126.sif}"
if [ ! -f "${image}" ]; then
	echo "Apptainer image not found: ${image}" >&2
	exit 1
fi

# Build optional mpiexec hostfile arguments from PBS_NODEFILE.
MPIEXEC_HOSTFILE_ARGS=()
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
	MPIEXEC_HOSTFILE_ARGS=(--hostfile "${PBS_NODEFILE}")
fi

# GPU count per node can be overridden when scheduler metadata is incomplete.
GPUS_PER_NODE="${POLARIS_GPUS_PER_NODE:-4}"
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
	# Infer distributed-training topology from scheduler variables and local GPU count.
	NODE_COUNT="$(sort -u "${PBS_NODEFILE}" | wc -l)"
	MASTER_ADDR="$(sort -u "${PBS_NODEFILE}" | head -n 1)"
else
	NODE_COUNT=1
	MASTER_ADDR="$(hostname)"
fi
TRAIN_RANKS=$(( NODE_COUNT * GPUS_PER_NODE ))
MASTER_PORT="${ALCHEMY_MASTER_PORT:-29500}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE:-32768}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-2}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=/opt/deepmd-kit}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

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
for path in Path("model-compression").glob("model.ckpt-*.pt"):
	match = re.fullmatch(r"model\.ckpt-(\d+)\.pt", path.name)
	if match is not None:
		checkpoints.append((int(match.group(1)), path.name))

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


train_log="train.${case_name}.${job_tag}.out"
rm -f "${train_log}"

# Emit a compact run header so scheduler output can be audited later.
echo "=========================================="
echo "JOB_START $(date --iso-8601=seconds)"
echo "CASE ${case_name}"
echo "BACKEND PT"
echo "CHECKPOINT_STRATEGY second_latest_when_available"
echo "CHAIN_MODE single_slice_level2_managed"
echo "SCRIPT_PATH ${script_path}"
echo "TRAIN_LOG ${train_log}"
echo "PBS_JOBID ${job_id}"
echo "Working directory: ${PBS_O_WORKDIR:-$PWD}"
echo "Polaris nodes: ${NODE_COUNT}; GPUs/node: ${GPUS_PER_NODE}; training ranks: ${TRAIN_RANKS}"
echo "Current step: $(latest_step)"
echo "=========================================="
nvidia-smi -L || true

mkdir -p model-compression

# Build the DeePMD train command, restarting from the selected checkpoint if one exists.
current_step="$(latest_step)"
if restart_prefix=$(latest_checkpoint_prefix); then
	validate_restart_prefix "${restart_prefix}" "${current_step}"
	echo "TRAIN_MODE restart ${restart_prefix}"
	train_args=(dp --pt train myinput.json --restart "${restart_prefix}")
else
	echo "TRAIN_MODE fresh"
	train_args=(dp --pt train myinput.json)
fi

# Allow walltime/interruption to reach the completion-marker check below.
set +e
# Launch PyTorch DeePMD on Polaris with one torchrun launcher per node.
mpiexec "${MPIEXEC_HOSTFILE_ARGS[@]}" -n "${NODE_COUNT}" --ppn 1 \
	apptainer exec --nv "${image}" env \
	PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
	HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING}" \
	DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE}" \
	OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
	DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS}" \
	DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS}" \
	NCCL_DEBUG="${NCCL_DEBUG}" \
	python -m torch.distributed.run \
	--nnodes="${NODE_COUNT}" \
	--nproc_per_node="${GPUS_PER_NODE}" \
	--rdzv_id="${job_tag}" \
	--rdzv_backend=c10d \
	--rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
	--no-python \
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
final_ckpt="$(latest_checkpoint_prefix || true)"
if [ -z "${final_ckpt}" ]; then
	echo "No PT checkpoint found for freeze/compress." >&2
	exit 1
fi
final_ckpt_abs="${PWD}/${final_ckpt}"
mpiexec "${MPIEXEC_HOSTFILE_ARGS[@]}" -n 1 --ppn 1 \
	apptainer exec --nv "${image}" dp --pt freeze -c "${final_ckpt_abs}" -o model-compression/pv.pth
mpiexec "${MPIEXEC_HOSTFILE_ARGS[@]}" -n 1 --ppn 1 \
	apptainer exec --nv "${image}" dp --pt compress -i model-compression/pv.pth -o model-compression/pv_comp.pth
echo "Output: model-compression/pv_comp.pth"
touch FREEZE_COMPRESS_DONE

echo "=========================================="
echo "JOB_END $(date --iso-8601=seconds) step=$(latest_step)"
echo "=========================================="
