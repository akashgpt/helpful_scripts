#!/bin/bash

set -euo pipefail
export PS1="${PS1-}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

ENV_PATH_DEFAULT="/scratch/gpfs/BURROWS/akashgpt/softwares/conda_envs_dir_secondary/envs/dp_tf15_src_hvd_fixed_20260405"
SOURCE_TREE_DEFAULT="/scratch/gpfs/BURROWS/akashgpt/softwares/installing_MLMD_related_stuff/deepmd_src_tf15_hvd_20260403"

ENV_PATH="${1:-$ENV_PATH_DEFAULT}"
SOURCE_TREE="${2:-$SOURCE_TREE_DEFAULT}"

function main() {
	module purge
	module load gcc/11
	module load openmpi/gcc/4.1.6
	module load cudatoolkit/12.8
	module load fftw/gcc/openmpi-4.1.6/3.3.10
	module load anaconda3/2025.12
	eval "$(conda shell.bash hook)"

	if [ -e "${ENV_PATH}" ]; then
		echo "Environment already exists: ${ENV_PATH}"
		return 1
	fi

	echo "Creating environment at ${ENV_PATH}"
	conda create -y -p "${ENV_PATH}" python=3.11 pip setuptools wheel cmake ninja
	conda activate "${ENV_PATH}"
	ensure_runtime_path_hook

	python -m pip install --upgrade pip

	echo "Installing TensorFlow 2.15.1 with the exact matching CUDA runtime packages"
	python -m pip install --no-cache-dir --ignore-installed \
		"tensorflow==2.15.1" \
		"nvidia-cublas-cu12==12.2.5.6" \
		"nvidia-cuda-cupti-cu12==12.2.142" \
		"nvidia-cuda-nvcc-cu12==12.2.140" \
		"nvidia-cuda-nvrtc-cu12==12.2.140" \
		"nvidia-cuda-runtime-cu12==12.2.140" \
		"nvidia-cudnn-cu12==8.9.4.25" \
		"nvidia-cufft-cu12==11.0.8.103" \
		"nvidia-curand-cu12==10.3.3.141" \
		"nvidia-cusolver-cu12==11.5.2.141" \
		"nvidia-cusparse-cu12==12.1.2.141" \
		"nvidia-nccl-cu12==2.16.5" \
		"nvidia-nvjitlink-cu12==12.2.140"

	echo "Installing build helpers"
	python -m pip install --no-cache-dir --ignore-installed \
		"mpi4py" \
		"scikit-build-core>=0.5,<0.13,!=0.6.0" \
		"dependency_groups" \
		"setuptools_scm" \
		"hatch-fancy-pypi-readme" \
		"packaging"

	echo "Installing NCCL headers for Horovod build"
	conda install -y -c conda-forge "nccl"

	configure_tensorflow_runtime
	build_horovod
	install_deepmd_support_deps
	build_deepmd_from_source
	run_import_sanity_check
	print_summary
}

function configure_tensorflow_runtime() {
	echo "Applying TensorFlow virtualenv GPU symlink fix"
	local tf_dir
	tf_dir="$("${ENV_PATH}/bin/python" - <<'PY'
import tensorflow as tf
import pathlib
print(pathlib.Path(tf.__file__).resolve().parent)
PY
)"

	(
		cd "${tf_dir}"
		ln -svf ../nvidia/*/lib/*.so* . >/dev/null
	)

	local ptxas_src
	ptxas_src="$("${ENV_PATH}/bin/python" - <<'PY'
import pathlib
import nvidia.cuda_nvcc
root = pathlib.Path(nvidia.cuda_nvcc.__file__).resolve().parent
match = next(root.rglob("ptxas"), None)
print("" if match is None else match)
PY
)"
	if [ -n "${ptxas_src}" ]; then
		ln -sf "${ptxas_src}" "${ENV_PATH}/bin/ptxas"
	fi
}

function ensure_runtime_path_hook() {
	mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
	printf '%s\n' 'export _OLD_DEEPMD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"' \
		'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH-}"' > "${CONDA_PREFIX}/etc/conda/activate.d/deepmd_tf15_runtime.sh"
	printf '%s\n' 'export LD_LIBRARY_PATH="${_OLD_DEEPMD_LD_LIBRARY_PATH-}"' \
		'unset _OLD_DEEPMD_LD_LIBRARY_PATH' > "${CONDA_PREFIX}/etc/conda/deactivate.d/deepmd_tf15_runtime.sh"
	export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH-}"
}

function build_deepmd_from_source() {
	echo "Building DeePMD-kit from source tree ${SOURCE_TREE}"
	if [ ! -d "${SOURCE_TREE}" ]; then
		echo "Missing source tree: ${SOURCE_TREE}"
		return 1
	fi

	export PYTHONNOUSERSITE=1
	export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH-}"
	export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
	export CUDA_HOME="${CONDA_PREFIX}"

	python -m pip install --no-build-isolation --force-reinstall --ignore-installed --no-deps "${SOURCE_TREE}"
}

function build_horovod() {
	echo "Building Horovod against the cleaned TensorFlow runtime"
	export PYTHONNOUSERSITE=1
	export CC=mpicc
	export CXX=mpicxx
	export CUDA_HOME="${CONDA_PREFIX}"
	export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
	export CUDAHOSTCXX="$(which g++)"
	export CMAKE_CUDA_HOST_COMPILER="$(which g++)"
	export HOROVOD_WITH_MPI=1
	export HOROVOD_WITHOUT_GLOO=1
	export HOROVOD_WITH_TENSORFLOW=1
	export HOROVOD_WITHOUT_PYTORCH=1
	export HOROVOD_WITHOUT_MXNET=1
	export HOROVOD_GPU_OPERATIONS=NCCL
	export HOROVOD_NCCL_HOME="${CONDA_PREFIX}"
	export HOROVOD_NCCL_INCLUDE="${CONDA_PREFIX}/include"
	export HOROVOD_NCCL_LIB="${CONDA_PREFIX}/lib"

	python -m pip install --no-cache-dir --force-reinstall --ignore-installed --no-build-isolation horovod
}

function install_deepmd_support_deps() {
	echo "Installing DeePMD runtime dependencies without disturbing TensorFlow pins"
	python -m pip install --no-cache-dir --ignore-installed \
		"scipy<1.12" \
		"pyyaml" \
		"dargs>=0.4.7" \
		"wcmatch" \
		"array-api-compat"
}

function run_import_sanity_check() {
	echo "Running import sanity check"
	python - <<'PY'
import deepmd
import horovod.tensorflow as hvd
import tensorflow as tf
print("DEEPMD_VERSION", deepmd.__version__)
print("TF_VERSION", tf.__version__)
print("HOROVOD_OK", hasattr(hvd, "init"))
PY
}

function print_summary() {
	echo
	echo "Installed corrected environment:"
	echo "  ${ENV_PATH}"
	echo
	echo "Validation command on a GPU node:"
	echo "  ${ENV_PATH}/bin/python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
}

main "$@"
