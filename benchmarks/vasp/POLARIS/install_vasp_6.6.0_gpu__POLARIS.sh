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
	# The ALCF VASP page currently documents the AOCL 3.2 VASP paths.
	# Additional AOCL 4.2 paths are detected below if Polaris exposes those instead.
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
		if [[ -n "${candidate}" && -d "${candidate}" ]]; then
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

find_first_file() {
	local label="$1"
	local name_pattern="$2"
	shift 2
	local candidate=""
	local root=""
	local found_file=""

	for candidate in "$@"; do
		if [[ -n "${candidate}" && -f "${candidate}" ]]; then
			echo "${candidate}"
			return 0
		fi
	done

	for root in "$@"; do
		if [[ -n "${root}" && -d "${root}" ]]; then
			found_file="$(find "${root}" -maxdepth 8 -type f -name "${name_pattern}" -print -quit 2>/dev/null || true)"
			if [[ -n "${found_file}" ]]; then
				echo "${found_file}"
				return 0
			fi
		fi
	done

	echo "ERROR: Could not find ${label}. Pattern: ${name_pattern}" >&2
	echo "Checked/searched:" >&2
	for candidate in "$@"; do
		echo "  ${candidate}" >&2
	done
	return 1
}

find_fftw_include_directory() {
	local candidate=""
	local root=""
	local found_file=""
	local -a candidates=(
		"${AOCL_FFTW_INCLUDE_DIR:-}"
		"${FFTW_INCDIR:-}"
		"${FFTW_ROOT:-}/include"
		"${FFTW_ROOT:-}/include_LP64"
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/include_LP64"
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/include_ILP64"
		"/soft/applications/vasp/aol-libs/3.2/amd-fftw/include"
		"/soft/applications/vasp/aol-libs/3.2/amd-fftw/include_LP64"
		"/soft/applications/vasp/aol-libs/3.2/amd-fftw/include/LP64"
		"/soft/libraries/aocl/3.2.0/include_LP64"
		"/soft/libraries/aocl/3.2.0/include"
	)
	local -a search_roots=(
		"/soft/libraries/math_libs/aocl-4.2"
		"/soft/libraries/math_libs"
		"/soft/applications/vasp/aol-libs/3.2"
		"/soft/applications/vasp/aol-libs"
		"/soft/libraries/aocl"
		"/soft/libraries"
	)

	for candidate in "${candidates[@]}"; do
		if [[ -n "${candidate}" && -d "${candidate}" ]]; then
			if compgen -G "${candidate}/fftw3*" >/dev/null; then
				echo "${candidate}"
				return 0
			fi
		fi
	done

	for root in "${search_roots[@]}"; do
		if [[ -d "${root}" ]]; then
			found_file="$(find "${root}" -maxdepth 8 -type f \
				\( -name 'fftw3.f03' -o -name 'fftw3.f' -o -name 'fftw3.h' \) \
				-print -quit 2>/dev/null || true)"
			if [[ -n "${found_file}" ]]; then
				dirname "${found_file}"
				return 0
			fi
		fi
	done

	echo "ERROR: Could not find an FFTW include directory containing fftw3.f03, fftw3.f, or fftw3.h." >&2
	echo "Checked fixed candidates:" >&2
	for candidate in "${candidates[@]}"; do
		echo "  ${candidate}" >&2
	done
	echo "Searched roots:" >&2
	for root in "${search_roots[@]}"; do
		echo "  ${root}" >&2
	done
	echo "You can override with: AOCL_FFTW_INCLUDE_DIR=/path/to/include ${0} ..." >&2
	return 1
}

find_fftw_library_directory() {
	local fftw_lib=""
	local fftw_dir=""
	fftw_lib="$(find_first_file "AOCL FFTW library" 'libfftw3.*' \
		"${AOCL_FFTW_LIB_DIR:-}" \
		"${FFTW_LIBDIR:-}" \
		"${FFTW_ROOT:-}/lib" \
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_LP64/libfftw3.a" \
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_ILP64/libfftw3.a" \
		"/soft/libraries/math_libs/aocl-4.2" \
		"/soft/libraries/math_libs" \
		"/soft/applications/vasp/aol-libs/3.2/amd-fftw/lib" \
		"/soft/libraries/aocl")" || return 1
	fftw_dir="$(dirname "${fftw_lib}")"
	if ! compgen -G "${fftw_dir}/libfftw3_omp.*" >/dev/null; then
		echo "ERROR: Found ${fftw_lib}, but no libfftw3_omp.* in ${fftw_dir}" >&2
		return 1
	fi
	echo "${fftw_dir}"
}

find_blas_library() {
	find_first_file "AOCL BLIS library" 'libblis*.a' \
		"${AOCL_BLAS_LIB:-}" \
		"${BLAS_LIB:-}" \
		"/soft/applications/vasp/aol-libs/3.2/amd-blis/lib/LP64/libblis-mt.a" \
		"/soft/applications/vasp/aol-libs/3.2/amd-blis/lib/ILP64/libblis-mt.a" \
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_LP64/libblis-mt.a" \
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_ILP64/libblis-mt.a" \
		"/soft/libraries/math_libs/aocl-4.2" \
		"/soft/libraries/math_libs" \
		"/soft/libraries/aocl"
}

find_lapack_library() {
	find_first_file "AOCL libFLAME library" 'libflame*.a' \
		"${AOCL_LAPACK_LIB:-}" \
		"${LAPACK_LIB:-}" \
		"/soft/applications/vasp/aol-libs/3.2/amd-libflame/lib/LP64/libflame.a" \
		"/soft/applications/vasp/aol-libs/3.2/amd-libflame/lib/ILP64/libflame.a" \
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_LP64/libflame.a" \
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_ILP64/libflame.a" \
		"/soft/libraries/math_libs/aocl-4.2" \
		"/soft/libraries/math_libs" \
		"/soft/libraries/aocl"
}

find_library_defining_symbol() {
	local symbol="$1"
	shift
	local root=""
	local found=""

	if ! command -v nm >/dev/null 2>&1; then
		echo "ERROR: nm is not available; cannot search for ${symbol}." >&2
		return 1
	fi

	for root in "$@"; do
		if [[ -n "${root}" && -f "${root}" ]]; then
			if nm -A "${root}" 2>/dev/null | grep -q "[[:space:]]${symbol}$"; then
				echo "${root}"
				return 0
			fi
		elif [[ -n "${root}" && -d "${root}" ]]; then
			found="$(find "${root}" -maxdepth 8 -type f \( -name '*.a' -o -name '*.so' -o -name '*.so.*' \) -print0 2>/dev/null \
				| xargs -0 -r nm -A 2>/dev/null \
				| awk -v symbol="${symbol}" '$NF == symbol {sub(/:.*/, "", $1); print $1; exit}')"
			if [[ -n "${found}" ]]; then
				echo "${found}"
				return 0
			fi
		fi
	done

	return 1
}

find_aocl_extra_libs() {
	local alcpu_lib=""
	alcpu_lib="$(find_library_defining_symbol "alcpu_flag_is_available" \
		"${AOCL_ALCPU_LIB:-}" \
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_LP64" \
		"/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_ILP64" \
		"/soft/libraries/math_libs/aocl-4.2" \
		"/soft/libraries/math_libs" \
		"/soft/libraries/aocl" \
		"/soft/applications/vasp/aol-libs")" || true

	if [[ -n "${alcpu_lib}" ]]; then
		echo "${alcpu_lib}"
	fi
}

patch_makefile_include_for_detected_paths() {
	local makefile_path="$1"

	python3 - "$makefile_path" "$AOCL_FFTW_INCLUDE_DIR" "$AOCL_FFTW_LIB_DIR" "$AOCL_BLAS_LIB" "$AOCL_LAPACK_LIB" "$AOCL_EXTRA_LIBS" <<'PY_PATCH'
from pathlib import Path
import sys

makefile_path = Path(sys.argv[1])
fftw_include_dir = sys.argv[2]
fftw_lib_dir = sys.argv[3]
blas_lib = sys.argv[4]
lapack_lib = sys.argv[5]
aocl_extra_libs = sys.argv[6]
text = makefile_path.read_text()

if 'FFTW_INCDIR ?=' not in text:
	text = text.replace('FFTW        = $(AOCL_ROOT)/amd-fftw\n', 'FFTW        = $(AOCL_ROOT)/amd-fftw\nFFTW_INCDIR ?= $(FFTW)/include\n', 1)
if 'FFTW_LIBDIR ?=' not in text:
	text = text.replace('FFTW_INCDIR ?= $(FFTW)/include\n', 'FFTW_INCDIR ?= $(FFTW)/include\nFFTW_LIBDIR ?= $(FFTW)/lib\n', 1)
if 'AOCL_EXTRA_LIBS ?=' not in text:
	text = text.replace('FFTW_LIBDIR ?= $(FFTW)/lib\n', 'FFTW_LIBDIR ?= $(FFTW)/lib\nAOCL_EXTRA_LIBS ?=\n', 1)

text = text.replace('INCS       += -I$(FFTW)/include', 'INCS       += -I$(FFTW_INCDIR)')
text = text.replace('LLIBS      += -L$(FFTW)/lib -lfftw3 -lfftw3_omp -lomp', 'LLIBS      += -L$(FFTW_LIBDIR) -lfftw3 -lfftw3_omp -lomp')
text = text.replace('LLIBS       = $(SCALAPACK) $(LAPACK) $(BLAS) $(CUDA)', 'LLIBS       = $(SCALAPACK) $(LAPACK) $(BLAS) $(AOCL_EXTRA_LIBS) $(CUDA)')

lines = []
seen_include = False
seen_lib = False
for line in text.splitlines():
	if line.startswith('BLAS') and '=' in line and 'BLACS' not in line:
		lines.append(f'BLAS        = {blas_lib}')
	elif line.startswith('LAPACK') and '=' in line:
		lines.append(f'LAPACK      = {lapack_lib}')
	elif line.startswith('AOCL_EXTRA_LIBS ?='):
		lines.append(f'AOCL_EXTRA_LIBS ?= {aocl_extra_libs}')
	elif line.startswith('FFTW_INCDIR ?='):
		lines.append(f'FFTW_INCDIR ?= {fftw_include_dir}')
		seen_include = True
	elif line.startswith('FFTW_LIBDIR ?='):
		lines.append(f'FFTW_LIBDIR ?= {fftw_lib_dir}')
		seen_lib = True
	else:
		lines.append(line)
if not seen_include:
	raise SystemExit('FFTW_INCDIR line was not found or inserted')
if not seen_lib:
	raise SystemExit('FFTW_LIBDIR line was not found or inserted')

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
	echo "AOCL_FFTW_LIB_DIR: ${AOCL_FFTW_LIB_DIR:-unset}"
	echo "AOCL_BLAS_LIB: ${AOCL_BLAS_LIB:-unset}"
	echo "AOCL_LAPACK_LIB: ${AOCL_LAPACK_LIB:-unset}"
	echo "AOCL_EXTRA_LIBS: ${AOCL_EXTRA_LIBS:-unset}"
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
	AOCL_FFTW_INCLUDE_DIR="$(find_fftw_include_directory)" || die "Could not locate an AOCL FFTW include directory."
	AOCL_FFTW_LIB_DIR="$(find_fftw_library_directory)" || die "Could not locate an AOCL FFTW library directory."
	AOCL_BLAS_LIB="$(find_blas_library)" || die "Could not locate an AOCL BLIS library."
	AOCL_LAPACK_LIB="$(find_lapack_library)" || die "Could not locate an AOCL libFLAME library."
	AOCL_EXTRA_LIBS="${AOCL_EXTRA_LIBS:-$(find_aocl_extra_libs)}"
	prepend_library_path "${AOCL_FFTW_LIB_DIR}"
	prepend_library_path "$(dirname "${AOCL_BLAS_LIB}")"
	prepend_library_path "$(dirname "${AOCL_LAPACK_LIB}")"

	if [[ -f makefile.include ]]; then
		cp -p makefile.include "makefile.include.backup.${TIMESTAMP}"
		echo "Backed up existing makefile.include -> makefile.include.backup.${TIMESTAMP}"
	fi
	cp "${MAKEFILE_TEMPLATE}" makefile.include
	patch_makefile_include_for_detected_paths makefile.include
	echo "Installed Polaris makefile.include from ${MAKEFILE_TEMPLATE}"
	echo "Using AOCL FFTW include directory: ${AOCL_FFTW_INCLUDE_DIR}"
	echo "Using AOCL FFTW library directory: ${AOCL_FFTW_LIB_DIR}"
	echo "Using AOCL BLIS library: ${AOCL_BLAS_LIB}"
	echo "Using AOCL libFLAME library: ${AOCL_LAPACK_LIB}"
	if [[ -n "${AOCL_EXTRA_LIBS}" ]]; then
		echo "Using AOCL extra library/libs: ${AOCL_EXTRA_LIBS}"
	else
		echo "Using AOCL extra library/libs: <none detected>"
	fi

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
