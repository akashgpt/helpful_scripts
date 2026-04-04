#!/bin/bash

set -euo pipefail
export PS1="${PS1-}"

# Future-install helper for the TensorFlow 2.15 + source-built DeepMD + Horovod
# stack that worked in the April 3, 2026 benchmarks.
#
# Default behavior is to clone the already-tested environment, which is the most
# reliable path on this cluster and avoids rebuilding large dependencies.
#
# Usage examples:
#	TARGET_ENV_PREFIX=/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/dp_tf15_src_hvd_clone_a /bin/bash install_dp_tf15_src_hvd__20260403.sh
#	INSTALL_MODE=rebuild TARGET_ENV_PREFIX=/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/dp_tf15_src_hvd_rebuild_a /bin/bash install_dp_tf15_src_hvd__20260403.sh
#
# Supported modes:
#	clone   : clone the proven working env
#	rebuild : rebuild from the local DeepMD source tree and rebuild Horovod

INSTALL_MODE="${INSTALL_MODE:-clone}"
TARGET_ENV_PREFIX="${TARGET_ENV_PREFIX:-/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/dp_tf15_src_hvd_clone_$(date +%Y%m%d_%H%M%S)}"
REFERENCE_ENV_PREFIX="${REFERENCE_ENV_PREFIX:-/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/dp_tf15_src_hvd_20260403}"
DEEPMD_SOURCE_DIR="${DEEPMD_SOURCE_DIR:-/scratch/gpfs/BURROWS/akashgpt/softwares/installing_MLMD_related_stuff/deepmd_src_tf15_hvd_20260403}"

if [[ -e "${TARGET_ENV_PREFIX}" ]]; then
	echo "Target env already exists: ${TARGET_ENV_PREFIX}" >&2
	exit 1
fi

module purge
module load gcc/11
module load openmpi/gcc/4.1.6
module load cudatoolkit/12.8
module load fftw/gcc/openmpi-4.1.6/3.3.10
module load anaconda3/2025.12

if [[ "${INSTALL_MODE}" == "clone" ]]; then
	if [[ ! -d "${REFERENCE_ENV_PREFIX}" ]]; then
		echo "Reference env not found for clone mode: ${REFERENCE_ENV_PREFIX}" >&2
		echo "Either set REFERENCE_ENV_PREFIX to a valid env or rerun with INSTALL_MODE=rebuild." >&2
		exit 1
	fi

	conda create -y -p "${TARGET_ENV_PREFIX}" --clone "${REFERENCE_ENV_PREFIX}"

	echo
	echo "Cloned working env to:"
	echo "  ${TARGET_ENV_PREFIX}"
	echo
	echo "Suggested activation:"
	echo "  module purge"
	echo "  module load gcc/11"
	echo "  module load openmpi/gcc/4.1.6"
	echo "  module load cudatoolkit/12.8"
	echo "  module load fftw/gcc/openmpi-4.1.6/3.3.10"
	echo "  module load anaconda3/2025.12"
	echo "  conda activate ${TARGET_ENV_PREFIX}"
	echo "  export PYTHONNOUSERSITE=1"
	echo '  NVIDIA_LIB_PATHS="$(python -c "import pathlib, site; print('\'':\''.join(str(path) for site_dir in site.getsitepackages() for path in sorted(pathlib.Path(site_dir, '\''nvidia'\'').glob('\''*/lib'\'')) if path.is_dir()))")"'
	echo '  export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"'
	exit 0
fi

if [[ "${INSTALL_MODE}" != "rebuild" ]]; then
	echo "Unsupported INSTALL_MODE=${INSTALL_MODE}. Use clone or rebuild." >&2
	exit 1
fi

if [[ ! -d "${DEEPMD_SOURCE_DIR}" ]]; then
	echo "DeepMD source tree not found: ${DEEPMD_SOURCE_DIR}" >&2
	exit 1
fi

conda create -y -p "${TARGET_ENV_PREFIX}" python=3.11 pip
conda activate "${TARGET_ENV_PREFIX}"
export PYTHONNOUSERSITE=1
export PATH="${CONDA_PREFIX}/bin:${PATH}"

pip install --upgrade pip setuptools wheel
pip install "tensorflow[and-cuda]==2.15.1" mpi4py
pip install "scikit-build-core>=0.5,!=0.6.0,<0.13" dependency_groups setuptools_scm hatch-fancy-pypi-readme

if ! command -v cmake >/dev/null 2>&1; then
	if command -v cmake3 >/dev/null 2>&1; then
		ln -sf "$(command -v cmake3)" "${CONDA_PREFIX}/bin/cmake"
	else
		echo "Neither cmake nor cmake3 is available in PATH." >&2
		exit 1
	fi
fi

pip install --no-build-isolation "${DEEPMD_SOURCE_DIR}"

export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_NCCL_HOME="${CONDA_PREFIX}"
export HOROVOD_NCCL_INCLUDE="${CONDA_PREFIX}/include"
export HOROVOD_NCCL_LIB="${CONDA_PREFIX}/lib"
export HOROVOD_CMAKE="${CONDA_PREFIX}/bin/cmake"
export CC=mpicc
export CXX=mpicxx
export CUDAHOSTCXX="$(command -v g++)"
export CMAKE_CUDA_HOST_COMPILER="$(command -v g++)"

pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall "horovod==0.28.1"

echo
echo "Rebuilt env at:"
echo "  ${TARGET_ENV_PREFIX}"
echo
echo "Quick verification:"
echo '  python -c "import tensorflow as tf, deepmd, horovod.tensorflow as hvd; print(tf.__version__, deepmd.__version__, hvd.__file__)"'
