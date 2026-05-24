#!/bin/bash

###############################
# Summary:
# This script installs DeePMD-kit (PyTorch backend ONLY — no TensorFlow),
# PLUMED, LAMMPS, and ASAP/SOAP descriptor tooling together with required
# PLUMED patches into a single conda environment (ALCHEMY_env__PT_SOAP).
#
# The PyTorch backend enables MULTI-GPU PARALLEL TRAINING via native PyTorch DDP
# (no horovod needed). LAMMPS uses the DeePMD C API which supports the PT
# backend directly — no model conversion needed (.pt models work in LAMMPS).
#
# WHY NO TENSORFLOW:
# TensorFlow and conda-forge PyTorch cannot coexist for C++ compilation due to
# a protobuf ABI conflict (TF headers use FullTypeDef inheriting from
# google::protobuf::Message, but conda-forge protobuf has an incompatible
# GetClassData() pure virtual). Since we need conda-forge PyTorch (for CXX11
# ABI compatibility with GCC 14), TF must be excluded entirely.
#
# Installation order:
#   1. Create conda env with Python 3.11 + conda packages (cudnn, nccl, gsl,
#      fftw, dpdata, ase, etc.)
#   2. Install PyTorch via conda-forge (CUDA 12, CXX11 ABI compatible)
#   3. Build DeePMD-kit v3.1.3 from source with DP_ENABLE_PYTORCH=1,
#      DP_ENABLE_TENSORFLOW=0 (pip install . + cmake/make)
#   4. Additional conda installs (dscribe>=2.0, scipy, scikit-learn, etc.)
#   5. Clone patched ASAP (asaplib) from ${asap_repo_url}@${asap_branch}
#      and install via pip --no-deps for SOAP/ASAP descriptor support
#   6. Build PLUMED from source (with custom CV patches)
#   7. Build LAMMPS from source (with DEEPMD C API + PLUMED packages)
#
# NOTE:
# - If no need of PLUMED patches, go for the "easy install" option on the
#   deepmodelling website.
# - If no need of PLUMED, then best to go for the APPTAINER/DOCKER version!
# - Horovod is NOT needed — multi-GPU uses PyTorch DDP via torchrun.
#
# ==========================================
# MULTI-GPU TRAINING (after installation):
# ==========================================
#
#   Use "python -m torch.distributed.run" (equivalent to "torchrun" but avoids
#   shebang issues on compute nodes where the hardcoded Python path may differ).
#
#   Single node, 2 GPUs:
#     python -m torch.distributed.run --nproc_per_node=2 --no-python \
#         dp --pt train input.json
#
#   Single node, 4 GPUs:
#     python -m torch.distributed.run --nproc_per_node=4 --no-python \
#         dp --pt train input.json
#
#   Multi-node (e.g. 2 nodes x 4 GPUs each, inside a Slurm job):
#     python -m torch.distributed.run \
#              --nnodes=${SLURM_NNODES} \
#              --nproc_per_node=4 \
#              --rdzv_id=${SLURM_JOB_ID} \
#              --rdzv_backend=c10d \
#              --rdzv_endpoint=$(hostname):29500 \
#              --no-python \
#              dp --pt train input.json
#
#   IMPORTANT: Multi-GPU uses data parallelism, so effective batch_size =
#   batch_size * N_GPUs. The learning rate is auto-scaled, but you should
#   manually reduce decay_steps by ~N_GPUs in input.json for equivalent
#   convergence.
#
#   Verify distributed mode in the training log:
#     DEEPMD INFO distributed
#     DEEPMD INFO world size: 4    # should be > 1
#
# ==========================================
# LAMMPS WITH PT MODELS:
# ==========================================
#   LAMMPS is built with DeePMD C API (DeePMD::deepmd_c) which supports the
#   PyTorch backend directly. Use .pt models in LAMMPS pair_style as-is:
#     pair_style  deepmd model.pt
#   No model conversion is needed.
#
# Usage: source <name of this script>
# Log file will be created in the same directory as log.deepmd-kit_and_others__<env_name>
#
# IMPORTANT: The conda activation script (env_vars.sh) automatically sets
# PYTHONNOUSERSITE=1 to prevent ~/.local/lib/pythonX.Y/site-packages from
# leaking stale packages into the conda environment (common source of
# NumPy/pyarrow/wrapt/absl-py errors). If you bypass the activation script,
# set this variable manually in your submit scripts.
#
# Example submit script for Della (single-GPU):
#   module purge
#   module load gcc-toolset/14
#   module load openmpi/gcc/4.1.6
#   module load cudatoolkit/12.8
#   module load fftw/gcc/openmpi-4.1.6/3.3.10
#   module load anaconda3/2025.12
#   conda activate ALCHEMY_env__PT_SOAP
#   export PYTHONNOUSERSITE=1
#   dp --pt train input.json
#
# Example submit script for Della (multi-GPU, 4 GPUs):
#   module purge
#   module load gcc-toolset/14
#   module load openmpi/gcc/4.1.6
#   module load cudatoolkit/12.8
#   module load fftw/gcc/openmpi-4.1.6/3.3.10
#   module load anaconda3/2025.12
#   conda activate ALCHEMY_env__PT_SOAP
#   export PYTHONNOUSERSITE=1
#   python -m torch.distributed.run --nproc_per_node=4 --no-python \
#       dp --pt train input.json
#
# Example submit script for Stellar (single-GPU):
#   module purge
#   module load gcc-toolset/10
#   module load openmpi/gcc/4.1.6
#   module load cudatoolkit/12.4
#   module load fftw/gcc/openmpi-4.1.6/3.3.10
#   module load anaconda3/2025.12
#   conda activate ALCHEMY_env__PT_SOAP
#   export PYTHONNOUSERSITE=1
#
# LAMMPS e.g.:
#   lmp -in <name of lammps input file>
#
# author: akashgpt and jinalee
###############################

# =============================
conda_env_name="ALCHEMY_env__PT_SOAP" # name of the conda environment to create and install everything in
# PLUMED custom CV patches directory — set by cluster detection below
dir_w_plumed_patches=""
asap_repo_url="https://github.com/akashgpt/ASAP.git" # patched ASAP (asaplib) for dscribe>=2.0 / NumPy 2.x
asap_branch="ALCHEMY"
# Build a second KOKKOS-enabled LAMMPS (lmp_kk) alongside the plain lmp for
# A/B benchmarking. Set to 0 to skip.
BUILD_KOKKOS_LAMMPS=1
# KOKKOS GPU arch — AMPERE80 covers A100 on Della & Delta.
KOKKOS_ARCH_FLAG="-DKokkos_ARCH_AMPERE80=yes"
# =============================



deepmd_plmd_lmp_misc__folder_name="deepmd-kit_and_others__${conda_env_name}"
conda_env="${conda_env_name}"
lmp_exec_name="lmp"
lmp_kk_exec_name="lmp_kk"


# send all output to log file
exec > >(tee -i log.${deepmd_plmd_lmp_misc__folder_name})
exec 2>&1

echo "====================="
echo "Date|Time: $(date)"
echo "Hostname: $(hostname)"
echo "conda_env_name: ${conda_env_name}"
echo "deepmd_plmd_lmp_misc__folder_name: ${deepmd_plmd_lmp_misc__folder_name}"
echo "conda_env: ${conda_env}"
echo "lmp_exec_name: ${lmp_exec_name}"
echo "====================="


# check if deepmd_plmd_lmp_misc__folder_name already exists and exit if so
if [ -d "$deepmd_plmd_lmp_misc__folder_name" ]; then
    echo "Directory ${deepmd_plmd_lmp_misc__folder_name} already exists. Exiting..."
    # end script with error without closing the respective terminal
    return 1
fi

parent_dir=`pwd`
cd ${parent_dir}


# if cluster della9, then ... else if stellar, then ...
if [[ $(hostname) == *"della"* ]]; then
    module purge
    echo "# ========================== #"
    echo "Loading modules for Della"
    echo "# ========================== #"
    module load gcc-toolset/14
    module load openmpi/gcc/4.1.6
    module load cudatoolkit/12.8
    module load fftw/gcc/openmpi-4.1.6/3.3.10
    module load anaconda3/2025.12
    dir_w_plumed_patches="/projects/BURROWS/akashgpt/lammps_patches__akashgpt"
    mpi_cxx_env_var="OMPI_CXX"
elif [[ $(hostname) == *"stellar"* ]]; then
    module purge
    echo "# ========================== #"
    echo "Loading modules for Stellar"
    echo "# ========================== #"
    module load gcc-toolset/10
    module load openmpi/gcc/4.1.6
    module load cudatoolkit/12.4
    module load fftw/gcc/openmpi-4.1.6/3.3.10
    module load anaconda3/2025.12
    dir_w_plumed_patches="/projects/BURROWS/akashgpt/lammps_patches__akashgpt"
    mpi_cxx_env_var="OMPI_CXX"
elif [[ $(hostname) == *"delta"* ]]; then
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
    dir_w_plumed_patches="/projects/bguf/akashgpt/lammps_patches__akashgpt"
    mpi_cxx_env_var="MPICH_CXX"
elif [[ $(hostname) == *"tiger"* ]]; then
    echo "Run the following command for Tiger (no access to GPUs on Tiger):" 
    echo "module purge && module load anaconda3/2025.12 && conda create -n ${conda_env_name} -c conda-forge -y deepmd-kit lammps horovod ase parallel dpdata"
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



# echo required modules
echo "====================="
echo "Required modules"
module list
echo "====================="


## DeePMD-kit installation
echo "====================="
echo "Installing DeePMD-kit"
echo "====================="
git clone https://github.com/deepmodeling/deepmd-kit.git ${deepmd_plmd_lmp_misc__folder_name} # can choose folder name
cd ${deepmd_plmd_lmp_misc__folder_name}
# IMPORTANT: Use v3.1.3 stable tag — master has DeepPotPTExpt.cc which calls
# AOTIModelPackageLoader with 5 args not available in PyTorch 2.6.
git checkout v3.1.3
deepmd_source_dir=`pwd`
cd $parent_dir

# check if $conda_env already exists, and remove it if so
# conda env -y remove -n $conda_env

conda create -y --name $conda_env python=3.11
conda activate $conda_env
pip install --upgrade pip

# All installed libraries
# conda install -y -c conda-forge cuda-toolkit
conda install -y -c conda-forge cudnn
# conda install -y -c conda-forge cudatoolkit-dev
# conda install -y -c conda-forge cuda-cudart cuda-version=12 nccl
# # conda install -y -c conda-forge openmpi
conda install -y -c conda-forge gsl fftw

# NCCL — needed for PyTorch DDP multi-GPU communication
conda install -y -c conda-forge nccl
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



# =============================================
# PyTorch — needed for multi-GPU training via DDP (torchrun)
# =============================================
# IMPORTANT: Must use conda-forge PyTorch, NOT PyPI wheels.
# PyPI wheels are built with CXX11_ABI_FLAG=0, but DeePMD-kit built from
# source with gcc-toolset/14 uses CXX11_ABI_FLAG=1. These are incompatible
# and cause "undefined symbol" errors when loading libdeepmd_op_pt.so.
# conda-forge builds use CXX11_ABI_FLAG=1, matching our compiler.
#
# CONDA_OVERRIDE_CUDA is needed when installing on a non-GPU login node —
# without it, the solver can't detect the __cuda virtual package and refuses
# to install CUDA-enabled builds.
echo "====================="
echo "Installing PyTorch (conda-forge, CUDA 12)"
echo "====================="
_cuda_ver=$(module list 2>&1 | grep -oP 'cudatoolkit/\K[0-9.]+')
if [ -z "${_cuda_ver}" ]; then
    echo "WARNING: Could not detect cudatoolkit version from loaded modules."
    echo "         Defaulting to CONDA_OVERRIDE_CUDA=12.8"
    _cuda_ver="12.8"
fi
CONDA_OVERRIDE_CUDA="${_cuda_ver}" conda install -y -c conda-forge "pytorch=2.*=cuda12*"


cd $deepmd_source_dir/source

echo "====================="
echo "Downloading LAMMPS and PLUMED"
echo "====================="

# Save the current stdout (file descriptor 1) to FD 3
exec 3>&1
# Redirect stdout to /dev/null to stop output to the terminal
exec 1>/dev/null
# download LAMMPS
wget https://github.com/lammps/lammps/archive/stable_2Aug2023_update3.tar.gz
tar -zxvf stable_2Aug2023_update3.tar.gz
# download PLUMED2
wget https://github.com/plumed/plumed2/archive/refs/tags/v2.8.2.tar.gz
tar -zxvf v2.8.2.tar.gz
# Restore stdout from FD 3 to resume output to the terminal
exec 1>&3


cd $deepmd_source_dir
export DP_VARIANT=cuda
# Point CMake to the system CUDA toolkit (from 'module load cudatoolkit/...')
# so it doesn't pick up the incomplete conda-forge CUDA headers (pulled in by
# PyTorch) instead. Without this, CMake finds cuda 12.9 from conda but can't
# locate cuda_runtime.h or cudart.
export CUDAToolkit_ROOT=$CUDA_HOME

# Enable PyTorch backend, disable TensorFlow.
# DP_ENABLE_PYTORCH=1 builds the PT interface for `dp --pt train` (including DDP).
# DP_ENABLE_TENSORFLOW=0 prevents any TF build attempts (avoids protobuf ABI conflict).
export DP_ENABLE_PYTORCH=1
export DP_ENABLE_TENSORFLOW=0

pip install .

# will possibly fail ^ in Della >> if so, do the following
# dscribe>=2.0 is the NumPy-2-compatible line; the ASAP source below is
# patched for dscribe 2.x SOAP/ACSF keyword changes and NumPy 2.x.
conda install -y -c conda-forge "dscribe>=2.0,<3"
conda install -y -c conda-forge "click>=7.0"
conda install -y -c conda-forge scipy scikit-learn ase umap-learn pyyaml tqdm pandas
pip install .

## ASAP (asaplib) installation
# Clone the patched ASAP (akashgpt/ASAP, branch ALCHEMY -- modernized for
# NumPy 2.x / Python 3.10+ / dscribe 2.x) and install via pip --no-deps.
# --no-deps because runtime dependencies are already covered by the conda
# installs above. Putting asap here keeps dp train, LAMMPS+deepmd MD, and
# asap gen_desc in one PyTorch-backend ALCHEMY environment.
echo "====================="
echo "Installing ASAP (asaplib) from ${asap_repo_url} (branch ${asap_branch})"
echo "====================="
asap_source_dir="${parent_dir}/ASAP_${conda_env_name}"
if [ -d "${asap_source_dir}" ]; then
    echo "Directory ${asap_source_dir} already exists; reusing without re-cloning."
else
    git clone --branch "${asap_branch}" --single-branch "${asap_repo_url}" "${asap_source_dir}" || return 1
fi
if [ ! -f "${asap_source_dir}/setup.py" ]; then
    echo "ERROR: Missing ASAP setup.py in ${asap_source_dir}"
    return 1
fi
python -m pip install --no-deps "${asap_source_dir}" || return 1
python -c "import asaplib; print('asaplib', getattr(asaplib, '__version__', 'version_unknown'))" || return 1
command -v asap >/dev/null 2>&1 || { echo "ERROR: asap command not found after ASAP install."; return 1; }
asap --help | sed -n '1,80p'

cd $deepmd_source_dir/source
mkdir build
cd build

# Build DeePMD C++ library with PyTorch backend.
# Key flags:
#   -DENABLE_PYTORCH=ON / -DUSE_PT_PYTHON_LIBS=TRUE: use PT from conda-forge
#   -DCUDAToolkit_ROOT=$CUDA_HOME: use system CUDA (not conda stubs)
#   -DCMAKE_CXX_FLAGS="-Wno-stringop-overflow": suppress GCC 14 false-positive
#     warnings in PyTorch headers (std::vector<bool> overflow)
#   -DCMAKE_EXE_LINKER_FLAGS / CMAKE_SHARED_LINKER_FLAGS: ensure conda's
#     libstdc++ (with GLIBCXX_3.4.30+) is found at link time instead of the
#     system RHEL9 libstdc++ (which only has GLIBCXX_3.4.3)
cmake3 \
    -DENABLE_PYTORCH=ON \
    -DUSE_PT_PYTHON_LIBS=TRUE \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DUSE_CUDA_TOOLKIT=TRUE \
    -DCUDAToolkit_ROOT=$CUDA_HOME \
    -DLAMMPS_SOURCE_ROOT="${deepmd_source_dir}/source/lammps-stable_2Aug2023_update3" \
    -DDP_USING_C_API=OFF \
    -DCMAKE_CXX_FLAGS="-Wno-stringop-overflow" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib" \
    ..

make -j16 install && make lammps

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
cp $dir_w_plumed_patches/* src/colvar/
# ==============================

./configure --prefix=$CONDA_PREFIX --enable-modules=all CXX=mpicxx CXXFLAGS="-Ofast"
make -j16 install


# activate Plumed2 relevant env variables when conda env gets activated
# echo "Add the text at the end of this script in ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"


# ==============================
cat << 'EOF' > "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
# Activate PLUMED environment variables and isolate from user site-packages

# Prevent ~/.local/lib/pythonX.Y/site-packages from shadowing conda packages
export PYTHONNOUSERSITE=1

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

# LAMMPS cmake — uses DeePMD C API (deepmd_c) which supports PT backend.
# Linker flags ensure conda's libstdc++ (GLIBCXX_3.4.30+) is used for
# libc10.so and libnccl.so dependencies from PyTorch.
cmake3 \
        -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
        -DCMAKE_INSTALL_PATH=${CONDA_PREFIX} \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_CXX_FLAGS="-Wno-stringop-overflow" \
        -DCMAKE_EXE_LINKER_FLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib" \
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
        ../cmake


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

make -j16 install  




# Making a symbolic link to the lmp_mpi executable in the LAMMPS source directory. While this executable is technically
# accessible from any conda environment since it's in ~/.local/bin, it's not guaranteed that the auxiliary packages will work.
# ln -s $PWD/<name of the executable> ~/.local/bin/lmp_plmd # likely "lmp"
rm -f ~/.local/bin/${lmp_exec_name}
ln -s $PWD/lmp ~/.local/bin/${lmp_exec_name}


# =======================================================================
# KOKKOS LAMMPS build (lmp_kk)
# =======================================================================
# Builds a second LAMMPS executable with the KOKKOS package enabled,
# targeting CUDA + OpenMP on A100 (AMPERE80). The plain `lmp` above is
# left untouched so you can A/B compare: `lmp` vs `lmp_kk -k on g 1 -sf kk
# -pk kokkos newton on neigh half`.
#
# Notes:
#  - DeepMD's pair_style deepmd has NO KOKKOS variant (DeepMD already runs
#    on GPU via its own PT kernels). KOKKOS here accelerates integrators,
#    neighbor lists, comm, KSPACE, etc.
#  - nvcc_wrapper is injected into the MPI CXX compiler via the
#    cluster-specific env var (OMPI_CXX or MPICH_CXX) set above.
#  - Uses the same DEEPMD C API + PLUMED static install from the first
#    build — no rebuild of those pieces.
# =======================================================================
if [ "${BUILD_KOKKOS_LAMMPS}" = "1" ]; then
    echo ""
    echo "====================="
    echo "Installing LAMMPS (KOKKOS variant -> lmp_kk)"
    echo "====================="

    cd $deepmd_source_dir/source/lammps-stable_2Aug2023_update3
    mkdir -p build_kk
    cd build_kk

    # Point MPI CXX wrapper at kokkos nvcc_wrapper so that `mpicxx` invokes
    # nvcc under the hood for CUDA kernels. This is standard practice for
    # KOKKOS+CUDA LAMMPS builds.
    kokkos_nvcc_wrapper="${deepmd_source_dir}/source/lammps-stable_2Aug2023_update3/lib/kokkos/bin/nvcc_wrapper"
    export ${mpi_cxx_env_var}="${kokkos_nvcc_wrapper}"
    echo "Using ${mpi_cxx_env_var}=${kokkos_nvcc_wrapper}"

    # conda-forge PyTorch ships a stub `nvcc` in $CONDA_PREFIX/bin without a
    # full toolkit (no targets/x86_64-linux/nvvm/bin/cicc). nvcc_wrapper
    # resolves `nvcc` via PATH, so we must prepend the module-provided
    # $CUDA_HOME/bin to PATH. Otherwise the KOKKOS cmake check errors with
    # "nvvm/bin/cicc: No such file or directory".
    _prev_PATH="$PATH"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "Prepended \$CUDA_HOME/bin to PATH: $(which nvcc)"

    # BUILD_SHARED_LIBS=no: static link so we don't install an
    # overlapping liblammps.so into CONDA_PREFIX/lib (the plain `lmp`
    # build already put one there). Kokkos+CUDA also prefers static.
    cmake3 \
            -DCMAKE_CXX_COMPILER=${kokkos_nvcc_wrapper} \
            -DCMAKE_CXX_FLAGS="-Wno-stringop-overflow" \
            -DCMAKE_EXE_LINKER_FLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib" \
            -DBUILD_MPI=yes \
            -DBUILD_OMP=yes \
            -DPKG_KOKKOS=yes \
            -DKokkos_ENABLE_CUDA=yes \
            -DKokkos_ENABLE_OPENMP=yes \
            ${KOKKOS_ARCH_FLAG} \
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
            -DBUILD_SHARED_LIBS=no \
            ../cmake

    # Build only the lmp binary — do NOT `make install` here (would clobber
    # libraries from the first build). Symlink it into ~/.local/bin as lmp_kk.
    make -j16 lmp

    if [ -f lmp ]; then
        rm -f ~/.local/bin/${lmp_kk_exec_name}
        ln -s "$(pwd)/lmp" ~/.local/bin/${lmp_kk_exec_name}
        echo "KOKKOS LAMMPS binary: $(pwd)/lmp"
        echo "Symlink:             ~/.local/bin/${lmp_kk_exec_name}"
    else
        echo "WARNING: KOKKOS LAMMPS build did not produce lmp binary — skipping lmp_kk install."
    fi

    # Unset the MPI CXX override so it doesn't leak into any later commands.
    unset ${mpi_cxx_env_var}
    export PATH="${_prev_PATH}"
    unset _prev_PATH
fi
# =======================================================================


## Verify PyTorch backend is available
echo ""
echo "====================="
echo "Verifying PyTorch backend"
echo "====================="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" 2>&1 || \
    echo "ERROR: PyTorch import failed. Multi-GPU training will NOT work."
python -c "from deepmd.pt.train import training; print('DeePMD PT backend: OK')" 2>&1 || \
    echo "ERROR: DeePMD PT backend import failed. Check DP_ENABLE_PYTORCH was set during build."
python -c "import dscribe; print('dscribe', getattr(dscribe, '__version__', 'version_unknown'))" 2>&1 || \
    echo "ERROR: dscribe import failed. SOAP descriptor workflows will NOT work."
python -c "import asaplib; print('asaplib', getattr(asaplib, '__version__', 'version_unknown'))" 2>&1 || \
    echo "ERROR: asaplib import failed. ASAP/SOAP workflows will NOT work."
command -v asap >/dev/null 2>&1 || \
    echo "ERROR: asap command not found. ASAP/SOAP workflows will NOT work."


echo ""
echo "========================================================"
echo "  INSTALLATION COMPLETE"
echo "  Date|Time: $(date)"
echo "========================================================"
echo ""
echo "Conda environment:  ${conda_env}"
echo "LAMMPS executable:  ${lmp_exec_name}"
echo "ASAP/SOAP tooling:  asap command + asaplib Python package"
if [ "${BUILD_KOKKOS_LAMMPS}" = "1" ]; then
    echo "LAMMPS (KOKKOS):    ${lmp_kk_exec_name}"
    echo "  Usage: ${lmp_kk_exec_name} -k on g <N_GPU> -sf kk -pk kokkos newton on neigh half -in in.lammps"
fi
echo ""
echo "--- Multi-GPU training (PyTorch DDP) ---"
echo ""
echo "  Single-GPU:"
echo "    dp --pt train input.json"
echo ""
echo "  Single node, 4 GPUs:"
echo "    python -m torch.distributed.run --nproc_per_node=4 --no-python \\"
echo "        dp --pt train input.json"
echo ""
echo "  Multi-node (Slurm):"
echo "    python -m torch.distributed.run --nnodes=\${SLURM_NNODES} --nproc_per_node=4 \\"
echo "             --rdzv_id=\${SLURM_JOB_ID} --rdzv_backend=c10d \\"
echo "             --rdzv_endpoint=\$(hostname):29500 \\"
echo "             --no-python dp --pt train input.json"
echo ""
echo "  Remember: effective batch_size = batch_size * N_GPUs."
echo "    -> Manually reduce decay_steps by ~N_GPUs in input.json."
echo "    -> Check log for 'world size: N' to confirm distributed mode."
echo ""
echo "--- LAMMPS with PT models ---"
echo "  LAMMPS uses DeePMD C API with PT backend directly."
echo "  Use .pt models as-is: pair_style deepmd model.pt"
echo ""
echo "========================================================"
echo "REMINDER: Always add 'export PYTHONNOUSERSITE=1' to your sbatch scripts"
echo "          before running dp or lmp commands."
