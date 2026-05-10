#!/usr/bin/env bash
# Build VASP 6.6.0 GPU on ALCF Polaris with a reproducible logged setup.

set -uo pipefail

usage() {
	cat <<'EOF'
Usage:
	install_vasp_6.6.0_gpu__POLARIS.sh /path/to/vasp.6.6.0.gpu [--clean]

What it does:
	1. Loads the Polaris/NVHPC/AOCL module/runtime environment.
	2. Copies makefile.include__POLARIS_GPU into the VASP source directory.
	3. Runs make std -j1 and writes a timestamped build log.
	4. If the build fails, writes a compact error-context log for debugging.

Environment overrides:
	MAKEFILE_INCLUDE_TEMPLATE=/path/to/makefile.include__POLARIS_GPU
	BUILD_TARGET=std
	BUILD_JOBS=1

Example:
	./install_vasp_6.6.0_gpu__POLARIS.sh /lus/grand/projects/CoreCollapseModel/akashgpt/softwares/vasp/vasp.6.6.0.gpu --clean
EOF
}

die() {
	echo "ERROR: $*" >&2
	exit 1
}

prepend_library_path() {
	local lib_dir="$1"

	if [[ -d "${lib_dir}" ]]; then
		export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
	fi
}

load_polaris_modules() {
	# Mirrors the ALCF Polaris VASP guide section
	# "Setting up compiler and libraries with module".
	echo "== Loading Polaris VASP GPU build environment =="

	if ! command -v module >/dev/null 2>&1; then
		die "Environment modules are unavailable. Run this on Polaris login/compute nodes with bash -l."
	fi

	module restore
	unset LD_PRELOAD
	module rm xalt >/dev/null 2>&1 || true
	module load cray-libsci
	module load nvidia/25.5

	if [[ -n "${NVIDIA_PATH:-}" ]]; then
		export NVROOT="${NVROOT:-${NVIDIA_PATH}}"
	fi
	if [[ -z "${NVROOT:-}" ]]; then
		die "NVROOT is not set and NVIDIA_PATH was unavailable after module load."
	fi

	prepend_library_path "${NVROOT}/cuda/lib64"
	prepend_library_path "${NVROOT}/compilers/extras/qd/lib"
	prepend_library_path "/soft/applications/vasp/aol-libs/3.2/amd-blis/lib/LP64"
	prepend_library_path "/soft/applications/vasp/aol-libs/3.2/amd-libflame/lib/LP64"
	prepend_library_path "/soft/applications/vasp/aol-libs/3.2/amd-fftw/lib"

	export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
}

check_required_path() {
	local path="$1"
	local label="$2"

	if [[ ! -e "${path}" ]]; then
		die "Missing ${label}: ${path}"
	fi
	echo "Found ${label}: ${path}"
}

find_existing_directory() {
	local label="$1"
	shift
	local candidate=""

	for candidate in "$@"; do
		if [[ -d "${candidate}" ]]; then
			echo "${candidate}"
			return 0
		fi
	done

	echo "ERROR: Could not find ${label}. Tried:" >&2
	for candidate in "$@"; do
		echo "  ${candidate}" >&2
	done
	return 1
}

patch_makefile_include_for_detected_paths() {
	local makefile_path="$1"

	python3 - "$makefile_path" "$AOCL_FFTW_INCLUDE_DIR" <<'PY_PATCH'
from pathlib import Path
import sys

makefile_path = Path(sys.argv[1])
fftw_include_dir = sys.argv[2]
text = makefile_path.read_text()

if 'FFTW_INCDIR ?=' not in text:
	text = text.replace('FFTW        = $(AOCL_ROOT)/amd-fftw\n', 'FFTW        = $(AOCL_ROOT)/amd-fftw\nFFTW_INCDIR ?= $(FFTW)/include\n', 1)

text = text.replace('INCS       += -I$(FFTW)/include', 'INCS       += -I$(FFTW_INCDIR)')

lines = []
replaced = False
for line in text.splitlines():
	if line.startswith('FFTW_INCDIR ?='):
		lines.append(f'FFTW_INCDIR ?= {fftw_include_dir}')
		replaced = True
	else:
		lines.append(line)
if not replaced:
	raise SystemExit('FFTW_INCDIR line was not found or inserted')

makefile_path.write_text('\n'.join(lines) + '\n')
PY_PATCH
}

print_environment_summary() {
	echo
	echo "== Build environment summary =="
	echo "Host: $(hostname)"
	echo "Date: $(date)"
	echo "PWD: ${PWD}"
	echo "VASP source: ${VASP_SRC}"
	echo "Build target: ${BUILD_TARGET}"
	echo "Build jobs: ${BUILD_JOBS}"
	echo "Template: ${MAKEFILE_TEMPLATE}"
	echo "AOCL_FFTW_INCLUDE_DIR: ${AOCL_FFTW_INCLUDE_DIR:-unset}"
	echo "NVROOT: ${NVROOT:-unset}"
	echo "NVIDIA_PATH: ${NVIDIA_PATH:-unset}"
	echo "MPICH_GPU_SUPPORT_ENABLED: ${MPICH_GPU_SUPPORT_ENABLED:-unset}"
	echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"
	echo
	echo "Compiler commands:"
	command -v nvfortran || true
	command -v nvc || true
	command -v nvc++ || true
	command -v ftn || true
	command -v cc || true
	echo
	echo "Loaded modules:"
	module list 2>&1 || true
}

write_error_context() {
	local log_file="$1"
	local context_file="$2"

	{
		echo "== Last 220 lines of build log =="
		tail -220 "${log_file}" || true
		echo
		echo "== Error-focused context =="
		grep -n -B40 -A30 -E "Error|error|FATAL|Fatal|fftw|fftw\\.o|fftw\\.F|Cannot|undefined reference|No such file|No rule" "${log_file}" || true
	} > "${context_file}"
}

run_build() {
	local make_status=0

	cd "${VASP_SRC}" || die "Could not cd into ${VASP_SRC}"

	check_required_path "makefile" "VASP top-level makefile"
	check_required_path "src" "VASP src directory"
	check_required_path "${MAKEFILE_TEMPLATE}" "Polaris makefile.include template"
	AOCL_FFTW_INCLUDE_DIR="$(find_existing_directory "AOCL FFTW include directory" \
		"${AOCL_FFTW_INCLUDE_DIR:-}" \
		"/soft/applications/vasp/aol-libs/3.2/amd-fftw/include" \
		"/soft/applications/vasp/aol-libs/3.2/amd-fftw/include_LP64" \
		"/soft/libraries/aocl/3.2.0/include_LP64" \
		"/soft/libraries/aocl/3.2.0/include")" || die "Could not locate an AOCL FFTW include directory."
	check_required_path "/soft/applications/vasp/aol-libs/3.2/amd-fftw/lib" "AOCL FFTW library directory"
	check_required_path "/soft/applications/vasp/aol-libs/3.2/amd-blis/lib/LP64/libblis-mt.a" "AOCL BLIS LP64 library"
	check_required_path "/soft/applications/vasp/aol-libs/3.2/amd-libflame/lib/LP64/libflame.a" "AOCL libFLAME LP64 library"

	if [[ -f makefile.include ]]; then
		cp -p makefile.include "makefile.include.backup.${TIMESTAMP}"
		echo "Backed up existing makefile.include -> makefile.include.backup.${TIMESTAMP}"
	fi
	cp "${MAKEFILE_TEMPLATE}" makefile.include
	patch_makefile_include_for_detected_paths makefile.include
	echo "Installed Polaris makefile.include from ${MAKEFILE_TEMPLATE}"
	echo "Using AOCL FFTW include directory: ${AOCL_FFTW_INCLUDE_DIR}"

	print_environment_summary

	if [[ "${DO_CLEAN}" == "1" ]]; then
		echo
		echo "== Running make clean =="
		make clean
	fi

	echo
	echo "== Starting build: make ${BUILD_TARGET} -j${BUILD_JOBS} =="
	set +e
	make "${BUILD_TARGET}" "-j${BUILD_JOBS}"
	make_status=$?
	set -e

	echo
	echo "== Build finished with status ${make_status} at $(date) =="
	if [[ "${make_status}" -ne 0 ]]; then
		write_error_context "${LOG_FILE}" "${ERROR_CONTEXT_FILE}"
		echo "Build failed."
		echo "Full log: ${LOG_FILE}"
		echo "Error context: ${ERROR_CONTEXT_FILE}"
		return "${make_status}"
	fi

	if [[ -x "bin/vasp_${BUILD_TARGET}" ]]; then
		echo "Built executable: ${VASP_SRC}/bin/vasp_${BUILD_TARGET}"
	elif [[ -x "bin/vasp_std" ]]; then
		echo "Built executable: ${VASP_SRC}/bin/vasp_std"
	else
		echo "WARNING: make succeeded, but no expected VASP executable was found in ${VASP_SRC}/bin"
	fi
	echo "Full log: ${LOG_FILE}"
	return 0
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

[[ $# -ge 1 ]] || {
	usage
	exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VASP_SRC="$(cd "$1" 2>/dev/null && pwd)" || die "VASP source directory does not exist: $1"
shift

DO_CLEAN=0
while [[ $# -gt 0 ]]; do
	case "$1" in
		--clean)
			DO_CLEAN=1
			shift
			;;
		*)
			die "Unknown argument: $1"
			;;
	esac
done

MAKEFILE_TEMPLATE="${MAKEFILE_INCLUDE_TEMPLATE:-${SCRIPT_DIR}/makefile.include__POLARIS_GPU}"
BUILD_TARGET="${BUILD_TARGET:-std}"
BUILD_JOBS="${BUILD_JOBS:-1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${VASP_SRC}/build_logs"
LOG_FILE="${LOG_DIR}/vasp_${BUILD_TARGET}_polaris_gpu_build_${TIMESTAMP}.log"
ERROR_CONTEXT_FILE="${LOG_DIR}/vasp_${BUILD_TARGET}_polaris_gpu_build_${TIMESTAMP}.error_context.log"

mkdir -p "${LOG_DIR}" || die "Could not create log directory: ${LOG_DIR}"

exec > >(tee "${LOG_FILE}") 2>&1

load_polaris_modules
run_build
