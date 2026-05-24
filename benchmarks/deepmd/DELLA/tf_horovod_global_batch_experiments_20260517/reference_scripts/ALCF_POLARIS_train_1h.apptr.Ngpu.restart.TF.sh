#!/bin/bash -l
#PBS -A CoreCollapseModel
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -j oe

# ALCHEMY Polaris/PBS DeePMD restart template.
# - Copy this file into a training directory and submit it from there with qsub.
# - select=N reserves N whole Polaris nodes; each Polaris node has 4 GPUs.
# - Self-resubmits this same script until training reaches the target step.
# - Restart safety uses conservative checkpoint choice and rollback validation.
# - Keep learning_rate.scale_by_worker="none" unless explicitly testing alternatives.

set -euo pipefail

# Resolve the working directory from the copied script path so all slices run in the same training folder.
cd "${PBS_O_WORKDIR}"

case_name="${ALCHEMY_RESTART_CASE:-train_1h_apptr_Ngpu_restart_TF}"
target_step="${ALCHEMY_TARGET_STEP:-}"
script_path="${ALCHEMY_RESTART_SCRIPT_PATH:-${PBS_O_WORKDIR}/train_1h.apptr.Ngpu.restart.TF.sh}"
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

# Self-restart guard: stop chaining if recent lcurve means indicate a blown-up model.
health_gate() {
	local step="${1:-0}"
	local gate_status
	local code=0

	if [ "${step}" -lt "${ALCHEMY_HEALTH_GATE_MIN_STEP:-50000}" ]; then
		return 0
	fi
	if [ ! -s lcurve.out ]; then
		return 0
	fi

	set +e
	gate_status=$(awk '
		($1 ~ /^[0-9]+$/) {
			n += 1
			total[n] = $2
			energy[n] = $3
			virial[n] = $5
		}
		END {
			if (n < 1) {
				print "no_numeric_rows"
				exit 0
			}
			start = n - 199
			if (start < 1) {
				start = 1
			}
			for (i = start; i <= n; i++) {
				count += 1
				sum_total += total[i]
				sum_energy += energy[i]
				sum_virial += virial[i]
			}
			mean_total = sum_total / count
			mean_energy = sum_energy / count
			mean_virial = sum_virial / count
			printf "step=%s window=%d mean_total=%.8g mean_energy=%.8g mean_virial=%.8g", last_step, count, mean_total, mean_energy, mean_virial
			if (mean_total > total_limit || mean_energy > energy_limit || mean_virial > virial_limit) {
				exit 2
			}
		}
	' last_step="${step}" total_limit="${ALCHEMY_HEALTH_GATE_TOTAL_LIMIT:-8.0}" energy_limit="${ALCHEMY_HEALTH_GATE_E_LIMIT:-0.25}" virial_limit="${ALCHEMY_HEALTH_GATE_V_LIMIT:-0.15}" lcurve.out)
	code=$?
	set -e

	printf "%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "${code}" "${gate_status}" >> HEALTH_GATE.tsv
	if [ "${code}" -eq 2 ]; then
		touch HEALTH_GATE_FAILED
		echo "CHAIN_STOPPED_BY_HEALTH_GATE step=${step} ${gate_status}"
		return 1
	fi
	return 0
}

# Self-restart controller: decide whether to submit the same script again.
resubmit_if_needed() {
	local reason="${1:-unknown}"
	local step
	local attempts
	local next_job
	local max_chain_jobs="${ALCHEMY_MAX_CHAIN_JOBS:-24}"
	local attempt_file="${ALCHEMY_CHAIN_ATTEMPT_FILE:-CHAIN_ATTEMPTS.txt}"
	local marker=".resubmitted_${job_tag}"

	if [ -z "${target_step:-}" ]; then
		target_step="$(target_training_step 2>/dev/null || printf "0")"
	fi
	if [ "${target_step}" -le 0 ]; then
		echo "CHAIN_NO_TARGET_STEP reason=${reason}; not resubmitting."
		return 0
	fi

	step="$(latest_step)"
	if [ "${step}" -ge "${target_step}" ]; then
		echo "CHAIN_DONE step=${step} target=${target_step} reason=${reason}"
		return 0
	fi
	if ! health_gate "${step}"; then
		echo "CHAIN_HEALTH_GATE_STOP job=${job_id} step=${step} reason=${reason}" | tee -a CHAIN_HISTORY.tsv
		return 0
	fi
	if [ -f "${marker}" ]; then
		echo "CHAIN_RESUBMIT_ALREADY_DONE job=${job_id} step=${step} reason=${reason}"
		return 0
	fi

	attempts="$(cat "${attempt_file}" 2>/dev/null || printf "0")"
	if [ "${attempts}" -ge "${max_chain_jobs}" ]; then
		echo "CHAIN_MAX_ATTEMPTS_REACHED attempts=${attempts} step=${step} target=${target_step} reason=${reason}" | tee -a CHAIN_HISTORY.tsv
		return 0
	fi

	touch "${marker}"
	next_job="$(qsub "${script_path}")"
	echo "CHAIN_RESUBMITTED reason=${reason} current_job=${job_id} next_job=${next_job} step=${step} target=${target_step} attempts=${attempts}" | tee -a CHAIN_HISTORY.tsv
}

# PBS/Polaris may terminate near walltime; try to queue the next slice on TERM.
on_term() {
	echo "CHAIN_SIGNAL_TERM $(date --iso-8601=seconds) job=${job_id} step=$(latest_step)"
	resubmit_if_needed "pre_walltime_or_term_signal"
}

# EXIT trap is the fallback resubmission path if a pre-walltime signal is missed.
on_exit() {
	local code=$?
	echo "CHAIN_EXIT $(date --iso-8601=seconds) job=${job_id} code=${code} step=$(latest_step)"
	resubmit_if_needed "exit_code_${code}"
}

trap on_term TERM
trap on_exit EXIT

# Count self-restart attempts so a broken chain cannot resubmit forever.
record_chain_attempt() {
	local max_chain_jobs="${ALCHEMY_MAX_CHAIN_JOBS:-24}"
	local attempt_file="${ALCHEMY_CHAIN_ATTEMPT_FILE:-CHAIN_ATTEMPTS.txt}"
	local attempts
	attempts="$(cat "${attempt_file}" 2>/dev/null || printf "0")"
	attempts=$((attempts + 1))
	printf "%s\n" "${attempts}" > "${attempt_file}"
	echo "CHAIN_ATTEMPT ${attempts}/${max_chain_jobs}"
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

# Use ALCHEMY_TARGET_STEP when supplied, otherwise read training.numb_steps from myinput.json.
target_training_step() {
	if [ -n "${target_step}" ]; then
		printf "%s\n" "${target_step}"
		return 0
	fi
	python - <<'PY'
import json

with open("myinput.json", encoding="utf-8") as handle:
	data = json.load(handle)
print(int(data["training"]["numb_steps"]))
PY
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



# Emit a compact run header so scheduler output can be audited later.
echo "=========================================="
echo "JOB_START $(date --iso-8601=seconds)"
echo "CASE ${case_name}"
echo "BACKEND TF"
echo "CHECKPOINT_STRATEGY second_latest_when_available"
echo "CHAIN_MODE self_resubmitting_reference"
echo "SCRIPT_PATH ${script_path}"
echo "PBS_JOBID ${job_id}"
echo "Working directory: ${PBS_O_WORKDIR:-$PWD}"
echo "Polaris nodes: ${NODE_COUNT}; GPUs/node: ${GPUS_PER_NODE}; training ranks: ${TRAIN_RANKS}"
echo "Current step: $(latest_step)"
target_step="$(target_training_step)"
echo "Target step: ${target_step}"
echo "=========================================="
nvidia-smi -L || true

mkdir -p model-compression
record_chain_attempt

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

# Launch TensorFlow DeePMD/Horovod on Polaris with one MPI rank per GPU.
mpiexec "${MPIEXEC_HOSTFILE_ARGS[@]}" -n "${TRAIN_RANKS}" --ppn "${GPUS_PER_NODE}" \
	apptainer exec --nv "${image}" env \
	PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
	HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING}" \
	DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE}" \
	OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
	DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS}" \
	DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS}" \
	TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH}" \
	XLA_FLAGS="${XLA_FLAGS}" \
	NCCL_DEBUG="${NCCL_DEBUG}" \
	"${train_args[@]}"

# Mark incomplete slices for Level 2 or self-restart reference scripts; finalization waits for target_step.
step_after_train="$(latest_step)"
if [ "${step_after_train}" -lt "${target_step}" ]; then
	echo "TRAIN_SLICE_ENDED_BEFORE_TARGET step=${step_after_train} target=${target_step}"
	touch TRAINING_INCOMPLETE
	rm -f TRAINING_COMPLETE FREEZE_COMPRESS_DONE
	exit 0
fi
touch TRAINING_COMPLETE
rm -f TRAINING_INCOMPLETE

echo "=========================================="
echo "Target reached. Starting freeze + compress at $(date)"
echo "=========================================="

# Freeze/compress only after training reaches target_step.
( cd model-compression && mpiexec "${MPIEXEC_HOSTFILE_ARGS[@]}" -n 1 --ppn 1 \
	apptainer exec "${image}" dp freeze -o pv.pb )
mpiexec "${MPIEXEC_HOSTFILE_ARGS[@]}" -n 1 --ppn 1 \
	apptainer exec --nv "${image}" dp compress -i model-compression/pv.pb -o model-compression/pv_comp.pb
echo "Output: model-compression/pv_comp.pb"
touch FREEZE_COMPRESS_DONE

echo "=========================================="
echo "JOB_END $(date --iso-8601=seconds) step=$(latest_step)"
echo "=========================================="
