#!/bin/bash

###############################
# Summary:
# This script installs DeePMD-kit, PLUMED, LAMMPS, and ASAP (asaplib) together
# with required PLUMED patches into a single conda environment (ALCHEMY_env).
# The script is intended to be run on a cluster with GPU support (used for
# DeePMD calculations) but can also run on a purely CPU system.
#
# Installation order:
#   1. Create conda env with Python 3.11 + conda packages (cudnn, gsl, fftw,
#      dpdata, ase, etc.)
#   2. Install TensorFlow via pip (needed for DeePMD build)
#   3. Build DeePMD-kit from source (pip install . + cmake/make)
#   4. Additional conda installs (dscribe>=2.0, scipy, scikit-learn, etc.)
#      -- these may clobber TF pip dependencies
#   5. Clone the patched ASAP (asaplib) from ${asap_repo_url}@${asap_branch}
#      and install via pip --no-deps (depends on dscribe>=2.0 from step 4;
#      same env as DeePMD/LAMMPS so the old separate `asap` env can be
#      retired)
#   6. Build PLUMED from source (with custom CV patches)
#   7. Build LAMMPS from source (with DEEPMD + PLUMED packages)
#   8. Reinstall TensorFlow via pip (--force-reinstall) to repair any
#      dependencies (wrapt, absl-py, etc.) clobbered by step 4
#
# NOTE:
# - If no need of PLUMED patches, go for the "easy install" option on the
#   deepmodelling website.
# - If no need of PLUMED, then best to go for the APPTAINER/DOCKER version!
# - Horovod (multi-GPU via TF backend) is NOT installed because horovod 0.28.1
#   is incompatible with TF 2.21 (as of 2026-04). Single-GPU dp train works
#   fine without it. For multi-GPU, use the PyTorch backend with native DDP:
#     dp train input.json --backend pt
#
# Usage: source <name of this script>
# Log file will be created in the same directory as log.deepmd-kit${ALCHEMY_env_suffix}.sh
#
# IMPORTANT: The conda activation script (env_vars.sh) automatically sets
# PYTHONNOUSERSITE=1 to prevent ~/.local/lib/pythonX.Y/site-packages from
# leaking stale packages into the conda environment (common source of
# NumPy/pyarrow/wrapt/absl-py errors). If you bypass the activation script,
# set this variable manually in your submit scripts.
#
# Example submit script for Della:
#   module purge
#   module load gcc-toolset/14
#   module load openmpi/gcc/4.1.6
#   module load cudatoolkit/12.8
#   module load fftw/gcc/openmpi-4.1.6/3.3.10
#   module load anaconda3/2025.12
#   conda activate ALCHEMY_env
#   export PYTHONNOUSERSITE=1
#
# Example submit script for Stellar:
#   module purge
#   module load gcc-toolset/10
#   module load openmpi/gcc/4.1.6
#   module load cudatoolkit/12.4
#   module load fftw/gcc/openmpi-4.1.6/3.3.10
#   module load anaconda3/2025.12
#   conda activate ALCHEMY_env
#   export PYTHONNOUSERSITE=1
#
# Example submit script for Delta RH9:
#   module reset
#   module load PrgEnv-gnu
#   module load gcc-native/13.2
#   module load cray-mpich
#   module load cudatoolkit/25.3_12.8
#   module load fftw/3.3.10-gcc13.3.1
#   module load miniforge3-python
#   eval "$(conda shell.bash hook)"
#   conda activate ALCHEMY_env
#   export MPICH_GPU_SUPPORT_ENABLED=1  # helpful for multi-rank GPU runs
#
# Example setup for ALCF Polaris:
#   module restore
#   module switch PrgEnv-nvidia PrgEnv-gnu
#   module use /soft/modulefiles
#   module load cudatoolkit-standalone
#   module load spack-pe-base cmake cray-fftw conda
#   eval "$(conda shell.bash hook)"
#   conda activate ALCHEMY_env
#   export MPICH_GPU_SUPPORT_ENABLED=1
#
# LAMMPS e.g.:
#   lmp -in <name of lammps input file>
#
# author: akashgpt and jinalee
###############################

# =============================
conda_env_name="ALCHEMY_env" # name of the conda environment to create and install everything in
del_existing_conda_env_and_dir=1 # 0/no: reuse an existing conda env; 1/yes: delete and recreate it
dir_w_plumed_patches="/projects/BURROWS/akashgpt/lammp*"
# dir_w_plumed_patches="/projects/bguf/akashgpt/lammp*"
# dir_w_plumed_patches="/lus/grand/projects/CoreCollapseModel/akashgpt/softwares/lammp*"
asap_repo_url="https://github.com/akashgpt/ASAP.git" # patched ASAP (asaplib) for dscribe>=2.0 / NumPy 2.x; cloned same way as deepmd-kit
asap_branch="ALCHEMY"
# =============================


# Force BLAS / OMP thread caps to 1 for the entire install. PBS interactive
# sessions on Polaris (and similar) export OMP_NUM_THREADS to the core count,
# which would (a) cause OpenBLAS thread-spawn storms during TF/NumPy import
# smoke tests on login nodes and (b) silently override our intended caps.
# Using bare "=1" (not "${VAR:-1}") makes the cap unambiguous in the log.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"


deepmd_plmd_lmp_misc__folder_name="deepmd-kit_and_others__${conda_env_name}"
conda_env="${conda_env_name}"
lmp_exec_name="lmp"
build_jobs=16
mpi_cxx_compiler="mpicxx"
deepmd_cmake_extra_args=()
lammps_cmake_extra_args=()


# send all output to log file
exec > >(tee -i log.${deepmd_plmd_lmp_misc__folder_name})
exec 2>&1

echo ""
echo "====================="
echo "Date|Time: $(date)"
echo "Hostname: $(hostname)"
echo "conda_env_name: ${conda_env_name}"
echo "deepmd_plmd_lmp_misc__folder_name: ${deepmd_plmd_lmp_misc__folder_name}"
echo "conda_env: ${conda_env}"
echo "del_existing_conda_env_and_dir: ${del_existing_conda_env_and_dir}"
echo "lmp_exec_name: ${lmp_exec_name}"
echo "OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "NUMEXPR_NUM_THREADS: ${NUMEXPR_NUM_THREADS}"
echo "TF_NUM_INTRAOP_THREADS: ${TF_NUM_INTRAOP_THREADS}"
echo "TF_NUM_INTEROP_THREADS: ${TF_NUM_INTEROP_THREADS}"
echo "PYTHONNOUSERSITE: ${PYTHONNOUSERSITE}"
echo "====================="


load_first_available_module() {
	local module_name=""

	for module_name in "$@"; do
		if module load "${module_name}" >/dev/null 2>&1; then
			echo "Loaded module: ${module_name}"
			return 0
		fi
	done

	echo "ERROR: Could not load any of these modules: $*" >&2
	return 1
}


set_cuda_toolkit_root_from_environment() {
	local cuda_root_candidate=""

	if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
		return 0
	fi

	for cuda_root_candidate in "${CUDA_HOME:-}" "${CUDATOOLKIT_HOME:-}" "${CUDA_PATH:-}" "${NVHPC_CUDA_HOME:-}"; do
		if [[ -n "${cuda_root_candidate}" && -d "${cuda_root_candidate}" ]]; then
			export CUDAToolkit_ROOT="${cuda_root_candidate}"
			return 0
		fi
	done

	return 0
}


resolve_cmake_command() {
	if command -v cmake3 >/dev/null 2>&1; then
		command -v cmake3
		return 0
	fi

	if command -v cmake >/dev/null 2>&1; then
		command -v cmake
		return 0
	fi

	echo "ERROR: Neither cmake3 nor cmake was found in PATH." >&2
	return 1
}


resolve_plumed_patch_glob() {
	local patch_glob_candidate=""

	if compgen -G "${dir_w_plumed_patches}" >/dev/null; then
		return 0
	fi

	for patch_glob_candidate in \
		"/lus/grand/projects/CoreCollapseModel/akashgpt/lammp*" \
		"/projects/BURROWS/akashgpt/lammp*" \
		"/projects/bguf/akashgpt/lammp*"; do
		if compgen -G "${patch_glob_candidate}" >/dev/null; then
			dir_w_plumed_patches="${patch_glob_candidate}"
			echo "Using PLUMED patch glob: ${dir_w_plumed_patches}"
			return 0
		fi
	done

	echo "ERROR: Could not find PLUMED patch files matching: ${dir_w_plumed_patches}" >&2
	return 1
}


debug_banner() {
	local message="$1"

	echo ""
	echo "===================== ALCHEMY_INSTALL_DEBUG ====================="
	echo "${message}"
	echo "Date|Time: $(date)"
	echo "PWD: $(pwd)"
	echo "================================================================="
}


fail_build() {
	# Surface a clear failure banner with diagnostic context when a build step
	# fails. Always returns the supplied non-zero exit code so callers can
	# pipe via:
	#     <build cmd> || { fail_build "<label>" $?; return 1; }
	local label="$1"
	local rc="${2:-1}"
	local build_log=""

	echo ""
	echo "######################################################################"
	echo "BUILD FAILURE: ${label} (rc=${rc})"
	echo "######################################################################"
	echo "Date|Time:         $(date)"
	echo "PWD:               $(pwd)"
	echo "Host identifiers:  ${host_id_bundle:-(not set)}"
	echo "CUDA toolkit:      ${CUDAToolkit_ROOT:-unset}"
	echo "Conda prefix:      ${CONDA_PREFIX:-unset}"
	echo "Full install log:  ${parent_dir:-(unknown)}/log.${deepmd_plmd_lmp_misc__folder_name:-(unknown)}"
	for build_log in \
		CMakeOutput.log \
		CMakeError.log \
		CMakeFiles/CMakeOutput.log \
		CMakeFiles/CMakeError.log \
		config.log; do
		if [[ -r "${build_log}" ]]; then
			echo "Build artefact:    $(pwd)/${build_log}"
		fi
	done
	echo "######################################################################"
	return "${rc}"
}


debug_command() {
	local description="$1"
	shift

	echo ""
	echo "ALCHEMY_INSTALL_DEBUG command: ${description}"
	echo "ALCHEMY_INSTALL_DEBUG running: $*"
	if "$@"; then
		echo "ALCHEMY_INSTALL_DEBUG ok: ${description}"
		return 0
	fi

	echo "ALCHEMY_INSTALL_DEBUG warning: ${description} failed with status $?. Continuing because this is diagnostic." >&2
	return 0
}


require_command_available() {
	local command_name="$1"

	if ! command -v "${command_name}" >/dev/null 2>&1; then
		echo "ERROR: Required command '${command_name}' not found." >&2
		echo "ALCHEMY_INSTALL_DEBUG PATH=${PATH}" >&2
		return 1
	fi

	echo "ALCHEMY_INSTALL_DEBUG command '${command_name}' -> $(command -v "${command_name}")"
}


require_path_exists() {
	local path_to_check="$1"
	local description="$2"

	if [[ ! -e "${path_to_check}" ]]; then
		echo "ERROR: Missing ${description}: ${path_to_check}" >&2
		return 1
	fi

	echo "ALCHEMY_INSTALL_DEBUG found ${description}: ${path_to_check}"
}


conda_environment_exists() {
	local environment_name="$1"

	conda env list | awk '{print $1}' | grep -Fxq "${environment_name}"
}


debug_python_import() {
	# Run from a neutral cwd so Python does not shadow the installed package
	# with a same-named source-tree subdirectory (e.g. an incomplete deepmd/
	# left behind by a failed `pip install .` would otherwise be imported
	# and produce misleading "No module named 'deepmd._version'" errors).
	local module_name="$1"

	require_command_available python || return 1
	echo "ALCHEMY_INSTALL_DEBUG python import check: ${module_name}"
	( cd /tmp && python -c "import importlib; module = importlib.import_module('${module_name}'); print('${module_name}', getattr(module, '__version__', 'version_unknown'))" )
}


debug_snapshot() {
	local label="$1"

	debug_banner "Snapshot: ${label}"
	echo "Hostname: $(hostname)"
	echo "Shell: ${SHELL:-unknown}"
	echo "User: ${USER:-unknown}"
	echo "PATH: ${PATH}"
	echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"
	echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}"
	echo "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT:-unset}"
	echo "CUDA_HOME: ${CUDA_HOME:-unset}"
	echo "MPI_CXX_COMPILER: ${MPI_CXX_COMPILER:-unset}"
	echo "MPICH_GPU_SUPPORT_ENABLED: ${MPICH_GPU_SUPPORT_ENABLED:-unset}"
	echo "CRAY_ACCEL_TARGET: ${CRAY_ACCEL_TARGET:-unset}"
	echo "http_proxy: ${http_proxy:-unset}"
	echo "https_proxy: ${https_proxy:-unset}"

	if type module >/dev/null 2>&1; then
		echo "Loaded modules:"
		module list 2>&1 || true
	fi

	debug_command "disk usage for current directory" df -h .
	debug_command "compiler version" "${MPI_CXX_COMPILER:-c++}" --version
	debug_command "cmake version" "${cmake_cmd:-cmake}" --version
	debug_command "conda info" conda info
	debug_command "pip version" pip --version
	debug_command "python version" python --version
	# CUDA toolkit + CCCL/Thrust headers. Mismatches here (e.g. CUDA 12.9 CCCL
	# 2.x macros incompatible with DeePMD-kit's nvcc invocations) are the most
	# common reason the deepmd-kit wheel build fails on Polaris-class nodes.
	debug_command "nvcc version" bash -c '
		nvcc_bin="${CUDAToolkit_ROOT:-${CUDA_HOME:-/usr}}/bin/nvcc"
		if [[ -x "${nvcc_bin}" ]]; then
			"${nvcc_bin}" --version
		elif command -v nvcc >/dev/null 2>&1; then
			nvcc --version
		else
			echo "nvcc not found"
		fi
	'
	debug_command "CCCL / Thrust header versions" bash -c '
		root="${CUDAToolkit_ROOT:-${CUDA_HOME:-/usr}}"
		printed=0
		for f in \
			"${root}/include/cub/version.cuh" \
			"${root}/include/thrust/version.h" \
			"${root}/include/cuda/std/__cccl/version.h"; do
			if [[ -r "$f" ]]; then
				echo "--- $f ---"
				grep -E "CCCL_VERSION|CUB_VERSION|THRUST_VERSION" "$f" || true
				printed=1
			fi
		done
		[[ "$printed" = 0 ]] && echo "No CCCL/Thrust headers found under ${root}/include/"
	'
	# Per-user task/process quota for this login session (cgroup v2 first, then v1).
	# Records the actual upper bound that triggered prior OpenBLAS pthread_create
	# EAGAIN failures on ALCF Polaris login nodes.
	debug_command "user cgroup pids.max" bash -c '
		for f in \
			/sys/fs/cgroup/user.slice/user-$(id -u).slice/pids.max \
			/sys/fs/cgroup/pids/user.slice/user-$(id -u).slice/pids.max; do
			if [[ -r "$f" ]]; then
				echo "$f: $(cat "$f")"
				exit 0
			fi
		done
		echo "user cgroup pids.max: not readable"
	'
	debug_command "ulimit -a" bash -c "ulimit -a"
}


debug_lammps_binary() {
	local lammps_binary="$1"

	require_path_exists "${lammps_binary}" "LAMMPS executable" || return 1
	echo "ALCHEMY_INSTALL_DEBUG LAMMPS help smoke check: ${lammps_binary}"
	"${lammps_binary}" -h | sed -n '1,80p'
}


# check if deepmd_plmd_lmp_misc__folder_name already exists and exit if so
if [ -d "$deepmd_plmd_lmp_misc__folder_name" ]; then
	if [[ "${del_existing_conda_env_and_dir}" -gt 0 ]]; then
		echo "Directory ${deepmd_plmd_lmp_misc__folder_name} already exists. Removing because del_existing_conda_env_and_dir=${del_existing_conda_env_and_dir}."
		rm -rf "${deepmd_plmd_lmp_misc__folder_name}" || return 1
	else
		echo "Directory ${deepmd_plmd_lmp_misc__folder_name} already exists. Exiting..."
		# end script with error without closing the respective terminal
		return 1
	fi
fi

parent_dir=`pwd`
cd ${parent_dir}


# Compute nodes on some clusters (e.g. ALCF Polaris) use Cray Shasta-style
# hostnames like "xNNNNcNsNNbNnN" that do not contain the cluster's marketing
# name. Build a bundle of host identifiers so login + compute nodes both match:
#   - hostname        : short name
#   - hostname -f     : FQDN, typically "<short>.<...>.polaris.alcf.anl.gov" on Polaris compute
#   - $PBS_O_HOST     : set inside PBS jobs to the submitting login node
host_short="$(hostname 2>/dev/null || echo '')"
host_fqdn="$(hostname -f 2>/dev/null || echo '')"
host_id_bundle="${host_short} ${host_fqdn} ${PBS_O_HOST:-}"
echo "Cluster detection host identifiers: ${host_id_bundle}"

# if cluster della9, then ... else if stellar, then ...
if [[ "${host_id_bundle}" == *"della"* ]]; then
    module purge
    echo "# ========================== #"
    echo "Loading modules for Della"
    echo "# ========================== #"
    module load gcc-toolset/14
    module load openmpi/gcc/4.1.6
    module load cudatoolkit/12.8
    module load fftw/gcc/openmpi-4.1.6/3.3.10
    module load anaconda3/2025.12
elif [[ "${host_id_bundle}" == *"stellar"* ]]; then
    module purge
    echo "# ========================== #"
    echo "Loading modules for Stellar"
    echo "# ========================== #"
    module load gcc-toolset/10
    module load openmpi/gcc/4.1.6
    module load cudatoolkit/12.4
    module load fftw/gcc/openmpi-4.1.6/3.3.10
    module load anaconda3/2025.12
elif [[ "${host_id_bundle}" == *"delta"* ]]; then
    module reset
    echo "# ========================== #"
    echo "Loading modules for Delta"
    echo "# ========================== #"
    module load PrgEnv-gnu
    module load gcc-native/13.2
    module load cray-mpich
    module load cudatoolkit/25.3_12.8
    module load fftw/3.3.10-gcc13.3.1
    module load miniforge3-python
elif [[ "${host_id_bundle}" == *"polaris"* ]]; then
    module restore
    echo "# ========================== #"
    echo "Loading modules for ALCF Polaris"
    echo "# ========================== #"

    module switch PrgEnv-nvidia PrgEnv-gnu 2>/dev/null || module swap PrgEnv-nvidia PrgEnv-gnu 2>/dev/null || true
    module use /soft/modulefiles
    # CUDA toolkit fallback order on Polaris.
    #
    # Background: DeePMD-kit 3.x's CUDA sources do not compile against CCCL 2.x
    # macros (the "_CCCL_PP_SPLICE_WITH_IMPL1 passed 3 arguments, but takes
    # just 2" build failure observed on this cluster with 12.9.1).
    #
    # Polaris toolkit inventory (probed 2026-05):
    #   - 11.8.0 .. 12.4.1  : NO cuda/std/__cccl/version.h header (pre-CCCL-2)
    #                         => safe for DeePMD-kit nvcc compiles.
    #   - 12.5.0 .. 12.6.3  : CCCL bundled, version uncertain; *may* work.
    #   - 12.8.x and newer  : CCCL 2.x macros confirmed-broken w/ DeePMD-kit.
    #
    # Fallback order: prefer the newest pre-CCCL-2 release first, fall through
    # to older pre-CCCL releases, then optimistically try the 12.5/12.6 line,
    # and only as a last resort accept a known-broken toolkit (so the operator
    # at least gets a clean build-failure banner pointing at the CUDA pin).
    load_first_available_module \
        cudatoolkit-standalone/12.4.1 \
        cudatoolkit-standalone/12.4.0 \
        cudatoolkit-standalone/12.3.2 \
        cudatoolkit-standalone/12.2.2 \
        cudatoolkit-standalone/11.8.0 \
        cudatoolkit-standalone/12.6.3 \
        cudatoolkit-standalone/12.6.2 \
        cudatoolkit-standalone/12.6.1 \
        cudatoolkit-standalone/12.6.0 \
        cudatoolkit-standalone/12.5.0 \
        cudatoolkit-standalone/12.8.1 \
        cudatoolkit-standalone/12.8.0 \
        cudatoolkit-standalone/12.9.1 \
        cudatoolkit-standalone || return 1
    load_first_available_module spack-pe-base || return 1
    load_first_available_module cmake || return 1
    load_first_available_module cray-fftw || return 1
    load_first_available_module conda miniforge3-python || return 1
    load_first_available_module apptainer || true

    # Polaris compute nodes need the ALCF proxy for outbound downloads.
    # These are harmless on login nodes and avoid git/pip/wget failures in
    # interactive compute-node build jobs.
    export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
    export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
    export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
    export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"

    export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
    export CRAY_ACCEL_TARGET="${CRAY_ACCEL_TARGET:-nvidia80}"
    mpi_cxx_compiler="${MPI_CXX_COMPILER:-CC}"
    build_jobs="${BUILD_JOBS:-8}"
    deepmd_cmake_extra_args+=("-DCMAKE_CUDA_ARCHITECTURES=80")
    lammps_cmake_extra_args+=("-DCMAKE_CUDA_ARCHITECTURES=80")
elif [[ "${host_id_bundle}" == *"tiger"* ]]; then
    echo "Run the following command for Tiger (no access to GPUs on Tiger):"
    echo "module purge && module load anaconda3/2025.12 && conda create -n ALCHEMY_env -c conda-forge -y deepmd-kit lammps horovod ase parallel dpdata"
    echo "Exiting."
    exit 0
else
    echo "Unknown cluster. Please load the required modules manually."
    exit 1
fi


# Ensure the conda shell function is available for 'conda activate'.
# Some module systems (e.g., anaconda3 on Della, miniforge on Delta) put conda
# on PATH but do not initialize shell activation. This makes it work everywhere.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi

cmake_cmd="$(resolve_cmake_command)" || return 1
set_cuda_toolkit_root_from_environment

cuda_cmake_args=()
if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
    cuda_cmake_args+=("-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
fi

export MPI_CXX_COMPILER="${mpi_cxx_compiler}"
export CXX="${CXX:-${MPI_CXX_COMPILER}}"
resolve_plumed_patch_glob || return 1
require_command_available git || return 1
require_command_available wget || return 1
require_command_available tar || return 1
require_command_available make || return 1
require_command_available "${cmake_cmd}" || return 1
require_command_available "${MPI_CXX_COMPILER}" || return 1



# echo required modules
echo "====================="
echo "Required modules"
module list
echo "====================="
echo "cmake command: ${cmake_cmd}"
echo "MPI C++ compiler wrapper: ${MPI_CXX_COMPILER}"
echo "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT:-not set}"
echo "build_jobs: ${build_jobs}"
echo "dir_w_plumed_patches: ${dir_w_plumed_patches}"
echo "====================="
debug_snapshot "after module and compiler setup"


## DeePMD-kit installation
echo "====================="
echo "Installing DeePMD-kit"
echo "====================="
git clone https://github.com/deepmodeling/deepmd-kit.git ${deepmd_plmd_lmp_misc__folder_name} # can choose folder name 
cd ${deepmd_plmd_lmp_misc__folder_name}
deepmd_source_dir=`pwd`
cd $parent_dir

if conda_environment_exists "${conda_env}"; then
	if [[ "${del_existing_conda_env_and_dir}" -gt 0 ]]; then
		echo "Existing conda environment '${conda_env}' found. Removing because del_existing_conda_env_and_dir=${del_existing_conda_env_and_dir}."
		conda env remove -y --name "${conda_env}" || return 1
		conda create -y --name "${conda_env}" python=3.11 || return 1
	else
		echo "Existing conda environment '${conda_env}' found. Reusing it because del_existing_conda_env_and_dir=${del_existing_conda_env_and_dir}."
	fi
else
	conda create -y --name "${conda_env}" python=3.11 || return 1
fi
conda activate $conda_env
conda config --env --add channels conda-forge 2>/dev/null || true
conda config --env --set channel_priority strict 2>/dev/null || true
conda config --env --set solver libmamba 2>/dev/null || true
debug_snapshot "after conda env creation and activation"
require_path_exists "${CONDA_PREFIX}" "active conda prefix" || return 1
require_command_available python || return 1
require_command_available pip || return 1
pip install --upgrade pip || { fail_build "pip self-upgrade" $?; return 1; }

# All installed libraries
# conda install -y -c conda-forge cuda-toolkit
conda install -y -c conda-forge cudnn
# conda install -y -c conda-forge cudatoolkit-dev
# conda install -y -c conda-forge cuda-cudart cuda-version=12 nccl
# # conda install -y -c conda-forge openmpi
conda install -y -c conda-forge gsl fftw
# conda install -y -c conda-forge clang-format

# # to fix the issue: cmake3: symbol lookup error: /lib64/libldap.so.2: undefined symbol: EVP_md2, version OPENSSL_3.0.0
conda install -y -c conda-forge openldap openssl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# only if cuda over conda
# mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# export CUDA_HOME=$CONDA_PREFIX


# for dpdata
echo "======================="
echo "Installing dpdata, ase, parallel, and other python packages"
echo "======================="
conda install -y conda-forge::dpdata
#################
# Install ase and parallel for post-processing and running multiple simulations in parallel
conda install -y -c conda-forge ase parallel
#################



# TensorFlow — installed here because DeePMD's build (pip install . and cmake)
# requires TF headers and libraries. It will be reinstalled at the END of the
# script to repair dependencies clobbered by subsequent conda installs.
echo "====================="
echo "Installing TensorFlow"
echo "====================="
pip install --upgrade tensorflow --no-cache-dir || return 1
debug_python_import tensorflow || return 1




cd $deepmd_source_dir/source

echo "====================="
echo "Downloading LAMMPS and PLUMED"
echo "====================="

# Save the current stdout (file descriptor 1) to FD 3
exec 3>&1
# Redirect stdout to /dev/null to stop output to the terminal
exec 1>/dev/null
# download + extract LAMMPS and PLUMED. Capture rcs so we can restore stdout
# (via FD 3) before reporting any failure -- otherwise the fail_build banner
# would itself be redirected to /dev/null.
wget https://github.com/lammps/lammps/archive/stable_2Aug2023_update3.tar.gz && \
    tar -zxvf stable_2Aug2023_update3.tar.gz
lammps_dl_rc=$?
wget https://github.com/plumed/plumed2/archive/refs/tags/v2.8.2.tar.gz && \
    tar -zxvf v2.8.2.tar.gz
plumed_dl_rc=$?
# Restore stdout from FD 3 to resume output to the terminal
exec 1>&3
if [[ "${lammps_dl_rc}" -ne 0 ]]; then
    fail_build "LAMMPS download/extract" "${lammps_dl_rc}"
    return 1
fi
if [[ "${plumed_dl_rc}" -ne 0 ]]; then
    fail_build "PLUMED download/extract" "${plumed_dl_rc}"
    return 1
fi


cd $deepmd_source_dir
export DP_VARIANT=cuda
# export CUDAToolkit_ROOT=$CUDA_HOME

# `pip install .` runs DeePMD-kit's scikit-build (or scikit-build-core) backend,
# which performs its OWN internal cmake configure step. That step does not see
# our deepmd_cmake_extra_args / cuda_cmake_args arrays, so pins like
# -DCMAKE_CUDA_ARCHITECTURES=80 and -DCUDAToolkit_ROOT=... would be silently
# dropped from the Python C-extension build (.so files under deepmd/lib/).
# CMAKE_ARGS is the env var honored by both scikit-build flavors, so we
# forward the same args via it. The explicit cmake invocation later (for the
# C++ libdeepmd_op_cuda + LAMMPS USER-DEEPMD plugin) keeps using the arrays
# directly; this just plugs the gap for the pip-install path.
_deepmd_pip_cmake_args=("${cuda_cmake_args[@]}" "${deepmd_cmake_extra_args[@]}")
if [[ ${#_deepmd_pip_cmake_args[@]} -gt 0 ]]; then
    export CMAKE_ARGS="${_deepmd_pip_cmake_args[*]}${CMAKE_ARGS:+ ${CMAKE_ARGS}}"
    echo "Forwarding to scikit-build via CMAKE_ARGS: ${CMAKE_ARGS}"
fi
unset _deepmd_pip_cmake_args

pip install . || { fail_build "DeePMD-kit pip install (1st)" $?; return 1; }
debug_python_import deepmd || return 1

# will possibly fail ^ in Della >> if so, do the following
# dscribe>=2.0 is the NumPy-2-compatible line; the ASAP_v2 source has been
# patched (np.complex_ -> np.complex128 and SOAP/ACSF kwarg renames) to match.
conda install -y -c conda-forge "dscribe>=2.0,<3"
conda install -y -c conda-forge "click>=7.0"
conda install -y -c conda-forge scipy scikit-learn ase umap-learn pyyaml tqdm pandas
pip install . || { fail_build "DeePMD-kit pip install (2nd, after conda installs)" $?; return 1; }

## ASAP (asaplib) installation
# Clone the patched ASAP (akashgpt/ASAP, branch ALCHEMY -- modernized for
# NumPy 2.x / Python 3.10+ / dscribe 2.x) and install via pip --no-deps.
# --no-deps because every runtime dep is already covered by the conda
# installs above (dscribe, scipy, scikit-learn, ase, umap-learn, pyyaml,
# tqdm, pandas, click). Putting asap here lets us retire the separate
# `asap` conda env and have one env for the full ALCHEMY workflow
# (dp train + LAMMPS+deepmd MD + asap gen_desc).
echo "====================="
echo "Installing ASAP (asaplib) from ${asap_repo_url} (branch ${asap_branch})"
echo "====================="
asap_source_dir="${parent_dir}/ASAP_${conda_env_name}"
if [ -d "${asap_source_dir}" ]; then
    echo "Directory ${asap_source_dir} already exists; reusing without re-cloning."
else
    git clone --branch "${asap_branch}" --single-branch "${asap_repo_url}" "${asap_source_dir}"
fi
require_path_exists "${asap_source_dir}/setup.py" "ASAP source tree" || return 1
python -m pip install --no-deps "${asap_source_dir}" \
    || { fail_build "ASAP (asaplib) pip install" $?; return 1; }
debug_python_import asaplib || return 1
require_command_available asap || return 1
debug_command "asap CLI smoke test" asap --help

cd $deepmd_source_dir/source
mkdir build
cd build

"${cmake_cmd}" \
    -DUSE_TF_PYTHON_LIBS=TRUE \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DUSE_CUDA_TOOLKIT=TRUE \
    "${cuda_cmake_args[@]}" \
    "${deepmd_cmake_extra_args[@]}" \
    -DLAMMPS_SOURCE_ROOT="${deepmd_source_dir}/source/lammps-stable_2Aug2023_update3" \
    -DDP_USING_C_API=OFF \
    .. || { fail_build "DeePMD-kit cmake configure" $?; return 1; }

# ^ cmake command might have issues finding python on Della; 
# if so, edit the shebang (first line) at /home/ag5805/.local/bin/cmake 
# From "#!/usr/local/bin/python3.11" to "#!/usr/bin/env python3" 
# (or the equivalent, depending on the python version)





# cmake3 \
#     -DUSE_TF_PYTHON_LIBS=TRUE \
#     -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
#     -DCUDAToolkit_ROOT=$CUDA_HOME \
#     -DUSE_CUDA_TOOLKIT=TRUE \
#     -DLAMMPS_SOURCE_ROOT="${deepmd_source_dir}/source/lammps-stable_2Aug2023_update3" \
#     -DDP_USING_C_API=OFF \
#     ..

make -j"${build_jobs}" install && make -j"${build_jobs}" lammps \
    || { fail_build "DeePMD-kit make install + make lammps" $?; return 1; }
require_path_exists "${deepmd_source_dir}/source/build/USER-DEEPMD" "DeePMD LAMMPS plugin directory" || return 1
require_command_available dp || return 1
debug_command "dp help after DeePMD build" dp -h

echo ""
## PLUMED installation
echo "====================="
echo "Installing PLUMED"
echo "====================="
cd $deepmd_source_dir/source
cd plumed2-2.8.2

# =============================
# IMPORTANT STEP IF YOU HAVE CUSTOM MADE CV: copy any .cpp files for collective variables directly into /src/colvar
# + Make sure to use cpp files that are compatible with the plumed2 version being used here
# e.g.:
# cp $AG_BURROWS/lammp*/* src/colvar/
resolve_plumed_patch_glob || return 1
cp $dir_w_plumed_patches/* src/colvar/
# ==============================

./configure --prefix=$CONDA_PREFIX --enable-modules=all CXX="${MPI_CXX_COMPILER}" CXXFLAGS="-Ofast" \
    || { fail_build "PLUMED ./configure" $?; return 1; }
make -j"${build_jobs}" install \
    || { fail_build "PLUMED make install" $?; return 1; }
require_command_available plumed || return 1
debug_command "plumed version after install" plumed --version


# activate Plumed2 relevant env variables when conda env gets activated
# echo "Add the text at the end of this script in ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"


# ==============================
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat << 'EOF' > "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
# Activate PLUMED environment variables and isolate from user site-packages

# Prevent ~/.local/lib/pythonX.Y/site-packages from shadowing conda packages
export PYTHONNOUSERSITE=1

# Polaris/Cray MPICH runtime defaults. These are harmless on non-Polaris
# hosts and useful for GPU-aware MPI runs launched through PBS/mpiexec.
# Polaris compute nodes use Cray Shasta hostnames (e.g. "xNNNNcNsNNbNnN")
# that lack the marketing name, so also check FQDN and PBS_O_HOST.
_polaris_host_bundle="$(hostname 2>/dev/null) $(hostname -f 2>/dev/null) ${PBS_O_HOST:-}"
if [[ "${_polaris_host_bundle}" == *"polaris"* ]]; then
    export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
    export CRAY_ACCEL_TARGET="${CRAY_ACCEL_TARGET:-nvidia80}"
fi
unset _polaris_host_bundle

# Set paths for PLUMED
export libdir="$CONDA_PREFIX/lib"
export bindir="$CONDA_PREFIX/bin"
export includedir="$CONDA_PREFIX/include"
export soext="so"                # Dynamic library extension (set to empty to disable PLUMED)
export progname="plumed"         # Name of the PLUMED program
export use_absolute_soname="no"  # "yes" if soname is absolute (no LD_LIBRARY_PATH needed)

# Prepend the binary directory to PATH
export PATH="$bindir:$PATH"

# Prepend the include directories
export CPATH="$includedir:$CPATH"
export INCLUDE="$includedir:$INCLUDE"

# Prepend the library directory
export LIBRARY_PATH="$libdir:$LIBRARY_PATH"

# Set the path for VIM syntax files (for PLUMED)
export PLUMED_VIMPATH="$libdir/$progname/vim"

# Prepend the pkg-config path
export PKG_CONFIG_PATH="$libdir/pkgconfig:$PKG_CONFIG_PATH"

# If the dynamic library extension is set, configure runtime paths
if [ -n "$soext" ]; then
    if [ -n "$PLUMED_KERNEL" ]; then
        echo "WARNING: PLUMED_KERNEL variable was already set, overriding it" >&2
    fi
    if [ "$use_absolute_soname" != "yes" ]; then
        if [ "$soext" = "dylib" ]; then
            export DYLD_LIBRARY_PATH="$libdir:$DYLD_LIBRARY_PATH"
        else
            export LD_LIBRARY_PATH="$libdir:$LD_LIBRARY_PATH"
        fi
    fi
    # Prepend the Python path for PLUMED
    export PYTHONPATH="$libdir/$progname/python:$PYTHONPATH"
    # Set the PLUMED_KERNEL variable
    export PLUMED_KERNEL="$libdir/lib${progname}Kernel.$soext"
fi
EOF
# ==============================



echo ""
## LAMMPS installation
echo "====================="
echo "Installing LAMMPS"
echo "====================="
conda deactivate
conda activate $conda_env # # to make all plumed environment variables are set before proceeding -- see end of this file

cd $deepmd_source_dir/source
cd lammps-stable_2Aug2023_update3/src/
cp -r $deepmd_source_dir/source/build/USER-DEEPMD DEEPMD
cd ..
mkdir build
cd build

echo "include(${deepmd_source_dir}/source/lmp/builtin.cmake)" >> ../cmake/CMakeLists.txt

# version 1 -- works but without multiple cores + uses gpu for deepmd
"${cmake_cmd}" \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
        -DCMAKE_INSTALL_PATH=${CONDA_PREFIX} \
        -DCMAKE_CXX_COMPILER="${MPI_CXX_COMPILER}" \
        "${cuda_cmake_args[@]}" \
        "${lammps_cmake_extra_args[@]}" \
        -DPKG_KSPACE=yes \
        -DPKG_RIGID=yes \
        -DPKG_MANYBODY=yes \
        -DPKG_MOLECULE=yes \
        -DPKG_EXTRA-FIX=yes \
        -DPKG_DEEPMD=yes \
        -DPKG_EXTRA=yes \
        -DPKG_REPLICA=yes \
        -DPKG_SHAKE=yes \
        -DPKG_PLUMED=yes \
        -DDOWNLOAD_PLUMED=no \
        -DPLUMED_MODE=static \
        -DENABLE_TESTING=no \
        -DLAMMPS_INSTALL_RPATH=yes \
        -DBUILD_SHARED_LIBS=yes \
        ../cmake || { fail_build "LAMMPS cmake configure" $?; return 1; }


# cmake3 \
#         -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
#         -DCMAKE_INSTALL_PATH=${CONDA_PREFIX} \
#         -DCMAKE_CXX_COMPILER=mpicxx \
#         -DPKG_KSPACE=yes \
#         -DPKG_RIGID=yes \
#         -DPKG_MANYBODY=yes \
#         -DPKG_MOLECULE=yes \
#         -DPKG_EXTRA-FIX=yes \
#         -DPKG_DEEPMD=yes \
#         -DPKG_EXTRA=yes \
#         -DPKG_REPLICA=yes \
#         -DPKG_SHAKE=yes \
#         -DPKG_PLUMED=yes \
#         -DDOWNLOAD_PLUMED=no \
#         -DPLUMED_MODE=shared \
#         -DPKG_PLUGIN=ON -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_FULL_LIBDIR=${CONDA_PREFIX}/lib \
#         -DENABLE_TESTING=no \
#         -DLAMMPS_INSTALL_RPATH=yes \
#         -DBUILD_SHARED_LIBS=yes \
#         ../cmake




# # version 2 -- testing for multiple cores
# cmake3 \
#         -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
#         -DCMAKE_INSTALL_PATH=${CONDA_PREFIX} \
#         -DCMAKE_CXX_FLAGS_RELEASE="-Ofast -fopenmp" \
#         -DBUILD_OMP=yes \
#         -DBUILD_MPI=yes \
#         -DCMAKE_CXX_COMPILER=mpicxx \
#         -DPKG_KSPACE=yes \
#         -DPKG_RIGID=yes \
#         -DPKG_MANYBODY=yes \
#         -DPKG_MOLECULE=yes \
#         -DPKG_EXTRA-FIX=yes \
#         -DPKG_DEEPMD=yes \
#         -DPKG_EXTRA=yes \
#         -DPKG_REPLICA=yes \
#         -DPKG_SHAKE=yes \
#         -DPKG_PLUMED=yes \
#         -DDOWNLOAD_PLUMED=no \
#         -DPLUMED_MODE=static \
#         -DENABLE_TESTING=no \
#         -DLAMMPS_INSTALL_RPATH=yes \
#         -DBUILD_SHARED_LIBS=yes \
#         ../cmake

make -j"${build_jobs}" install \
    || { fail_build "LAMMPS make install" $?; return 1; }
debug_lammps_binary "$PWD/lmp" || return 1




# Making a symbolic link to the lmp_mpi executable in the LAMMPS source directory. While this executable is technically
# accessible from any conda environment since it's in ~/.local/bin, it's not guaranteed that the auxiliary packages will work.
# ln -s $PWD/<name of the executable> ~/.local/bin/lmp_plmd # likely "lmp"
mkdir -p "$HOME/.local/bin"
rm -f ~/.local/bin/${lmp_exec_name}
ln -s $PWD/lmp ~/.local/bin/${lmp_exec_name}
require_path_exists "$HOME/.local/bin/${lmp_exec_name}" "LAMMPS symlink" || return 1


## Reinstall TensorFlow to fix clobbered dependencies
# The conda installs above (dpdata, ase, scipy, scikit-learn, etc.) clobber
# pip-managed TF dependencies (wrapt, absl-py, etc.). Reinstalling TF at the
# end restores all its dependencies to a consistent state.
echo "====================="
echo "Reinstalling TensorFlow and fixing Python dependencies"
echo "====================="
# mxnet requires numpy<2 but TF 2.21+ requires numpy>=2. mxnet is an unused
# horovod backend (we use the TensorFlow backend). Removing it before the TF
# install prevents pip from printing a spurious dependency conflict warning.
pip uninstall -y mxnet 2>/dev/null
echo "INFO: Uninstalled mxnet (unused horovod backend, incompatible with numpy>=2)."
pip install --upgrade --force-reinstall tensorflow --no-cache-dir || return 1
debug_python_import tensorflow || return 1
debug_python_import deepmd || return 1
debug_python_import dpdata || return 1
debug_python_import ase || return 1
debug_python_import dscribe || return 1
debug_python_import asaplib || return 1
require_command_available dp || return 1
require_command_available plumed || return 1
require_command_available asap || return 1
debug_lammps_binary "$HOME/.local/bin/${lmp_exec_name}" || return 1
debug_snapshot "final installation state"

## Horovod note
# As of 2026-04, horovod 0.28.1 (latest release) cannot compile against TF 2.21
# due to missing highwayhash headers and C++ incompatibilities with GCC>=14.
# Multi-GPU DeePMD training options:
#   - Use the PyTorch backend with native DDP: dp train input.json --backend pt
#   - Wait for a horovod release compatible with TF 2.21
# Single-GPU training with the TF backend works fine without horovod.
echo "WARNING: Horovod is NOT installed — incompatible with TF 2.21 (as of 2026-04)."
echo "         Single-GPU dp train works. For multi-GPU, use --backend pt (PyTorch+DDP)."

echo ""
echo "====================="
echo "Installation complete!"
echo "Date|Time: $(date)"
echo "====================="
echo "REMINDER: Always add 'export PYTHONNOUSERSITE=1' to your scheduler scripts"
echo "          before running dp or lmp commands."
