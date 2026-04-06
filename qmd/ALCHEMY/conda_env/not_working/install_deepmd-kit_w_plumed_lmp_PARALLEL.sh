#!/bin/bash

###############################
# Summary:
# This script installs DeePMD-kit, PLUMED, and LAMMPS together with required PLUMED patches.
# It also creates a conda environment with the necessary packages.
# The script is intended to be run on a cluster with GPU support (used for DeePMD calculations)
# but can also run a purely CPU system.
#
# Supports MULTI-GPU TRAINING via:
#   - TensorFlow backend: Horovod (data-parallel, MPI-based)
#   - PyTorch backend:    torchrun (PyTorch native distributed)
# Set INSTALL_MULTI_GPU and MULTI_GPU_BACKEND below to configure.
#
# NOTE: 
# If no need of PLUMED patches, go for the "easy install" option on deepmodelling website.
# If no need of PLUMED, then best to go for the APPTAINER/DOCKER version!
# 
# Usage: source <name of this script>
# log file will be created in the same directory as log.deepmd-kit${ALCHEMY_env_suffix}.sh
#
# To run lammps or deepmd simulations add the following to submit script:
# module load gcc-toolset/14
# module load openmpi/gcc/4.1.6
# module load cudatoolkit/12.8
# module load fftw/gcc/openmpi-4.1.6/3.3.10
# module load anaconda3/2025.12
# conda activate dp_plmd${ALCHEMY_env_suffix}
#
# LAMMPS e.g.:
# lmp_plmd${ALCHEMY_env_suffix} -in <name of lammps input file> # for lammps 
# where you will have to manually specify ${ALCHEMY_env_suffix}, e.g. lmp_plmd_09
#
# ==========================================
# MULTI-GPU TRAINING (after installation):
# ==========================================
#
# --- PyTorch backend (recommended for DPA-2/DPA-3 and newer models) ---
#   Use "python -m torch.distributed.run" instead of "torchrun" to avoid
#   shebang issues (torchrun scripts can have hardcoded Python paths that
#   don't exist on compute nodes). They are functionally identical.
#
#   Single node, 4 GPUs:
#     python -m torch.distributed.run --nproc_per_node=4 --no-python dp --pt train input.json
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
# --- TensorFlow backend (legacy se_e2_a / DPA-1 models) ---
#   Single node, 4 GPUs:
#     CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 -H localhost:4 \
#         dp train --mpi-log=workers input.json
#
#   Or via mpirun:
#     CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 \
#         dp train --mpi-log=workers input.json
#
# IMPORTANT: Multi-GPU uses data parallelism, so effective batch size = batch_size * N_GPUs.
#   The learning rate is auto-scaled, but you should manually reduce decay_steps by ~N_GPUs
#   in your input.json for equivalent convergence.
#
# Verify distributed mode in the training log:
#   DEEPMD INFO distributed
#   DEEPMD INFO world size: 4    # should be > 1
#
# ==========================================
# MULTI-GPU LAMMPS MD (after installation):
# ==========================================
# Requires LAMMPS_MULTI_GPU="yes" during build (enables BUILD_MPI + BUILD_OMP).
# Each MPI rank binds to one GPU and handles a spatial sub-domain of the simulation box.
#
#   Single node, 4 GPUs (Slurm):
#     srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 \
#         lmp_plmd_della -in input.lammps
#
#   Single node, 4 GPUs (mpirun, explicit GPU binding):
#     mpirun -np 4 --map-by slot \
#         bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; \
#                  lmp_plmd_della -in input.lammps'
#
#   Multi-node, 2 nodes x 4 GPUs (Slurm):
#     srun --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 \
#         lmp_plmd_della -in input.lammps
#
# IMPORTANT: Multi-GPU MD is beneficial mainly for large systems (>~5k atoms).
#   For small systems the MPI halo-exchange overhead can exceed the speedup.
#   Profile with a short run before committing to multi-GPU production jobs.
#   Also set in your LAMMPS input (recommended for DeePMD):
#     package omp 1              # 1 OMP thread per rank — avoids oversubscription
#     processors * * *           # let LAMMPS auto-decompose the box
#
# Usage: source install_deepmd-kit_w_plumed_lmp_PARALLEL.sh &
#
# author: akashgpt and jinalee
###############################

# =============================
# USER CONFIGURATION
# =============================
ALCHEMY_env_suffix="_della_all_pll" # can simply be "" or "_09"
conda_env_name="ALCHEMY_pll_env" # the prefix for the conda env name; the full env name will be ${ALCHEMY_env_prefix}${ALCHEMY_env_suffix}
dir_w_plumed_patches="/projects/JIEDENG/akashgpt/lammp*"

# --- Multi-GPU training support (dp train) ---
INSTALL_MULTI_GPU="yes"         # "yes" to install multi-GPU dependencies, "no" to skip
MULTI_GPU_BACKEND="both"        # "tensorflow" = Horovod only
                                # "pytorch"    = PyTorch + torchrun only
                                # "both"       = install both (recommended if disk space is not a concern)

# --- Multi-GPU LAMMPS MD support ---
LAMMPS_MULTI_GPU="yes"          # "yes" to build LAMMPS with explicit MPI + OpenMP for multi-GPU MD
                                #   Each MPI rank gets one GPU; atoms are spatially decomposed.
                                #   Best for large systems (>~5k atoms). For small systems a
                                #   single GPU is usually faster due to MPI comm overhead.
                                # "no"  to build the basic single-GPU version (your original "version 1")
# =============================



deepmd_plmd_lmp_misc__folder_name="kit_and_others__${conda_env_name}"
conda_env="${conda_env_name}"
lmp_exec_name="lmp"


# send all output to log file
exec > >(tee -i log.${deepmd_plmd_lmp_misc__folder_name})
exec 2>&1

INSTALL_START_TIME=$SECONDS

echo "====================="
echo "Installation started at $(date)"
echo "conda_env_name: ${conda_env_name}"
echo "deepmd_plmd_lmp_misc__folder_name: ${deepmd_plmd_lmp_misc__folder_name}"
echo "conda_env: ${conda_env}"
echo "lmp_exec_name: ${lmp_exec_name}"
echo "INSTALL_MULTI_GPU: ${INSTALL_MULTI_GPU}"
echo "MULTI_GPU_BACKEND: ${MULTI_GPU_BACKEND}"
echo "LAMMPS_MULTI_GPU:  ${LAMMPS_MULTI_GPU}"
echo "====================="


# check if deepmd_plmd_lmp_misc__folder_name already exists and exit if so
if [ -d "$deepmd_plmd_lmp_misc__folder_name" ]; then
    echo "Directory ${deepmd_plmd_lmp_misc__folder_name} already exists. Exiting..."
    # end script with error without closing the respective terminal
    return 1
fi

parent_dir=`pwd`
cd ${parent_dir}

# All required modules

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
elif [[ $(hostname) == *"tiger"* ]]; then
    echo "Run the following command for Tiger (no access to GPUs on Tiger):" 
    echo "module purge && module load anaconda3/2025.12 && conda create -n ALCHEMY_env -c conda-forge -y deepmd-kit lammps horovod ase parallel dpdata"
    echo "Exiting."
    exit 0
else
    echo "Unknown cluster. Please load the required modules manually."
    exit 1
fi



if [[ $(hostname) == *"delta"* ]]; then
    # Delta's miniforge module places conda on PATH but does not initialize shell
    # activation, so make the conda shell function available before using it.
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    fi
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
deepmd_source_dir=`pwd`
cd $parent_dir

# check if $conda_env already exists, and remove it if so
if conda info --envs | grep -q "${conda_env}"; then
    echo "Conda environment ${conda_env} already exists. Removing it to proceed with a clean installation."
    echo "You may try: conda remove -y --name ${conda_env} --all"
fi

conda create -y --name $conda_env python=3.11
conda activate $conda_env
pip install --upgrade pip

# All installed libraries
# conda install -y -c conda-forge cuda-toolkit
conda install -y -c conda-forge cudnn
# conda install -y -c conda-forge cudatoolkit-dev
# conda install -y -c conda-forge cuda-cudart cuda-version=12 nccl
# # conda install -y -c conda-forge openmpi
# conda install -y -c conda-forge fftw
# conda install -y -c conda-forge clang-format

# # to fix the issue: cmake3: symbol lookup error: /lib64/libldap.so.2: undefined symbol: EVP_md2, version OPENSSL_3.0.0
conda install -y -c conda-forge openldap openssl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# only if cuda over conda
# mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# export CUDA_HOME=$CONDA_PREFIX


# =============================================
# Install NCCL — needed for multi-GPU comms
# (both Horovod/NCCL and PyTorch DDP use it)
# =============================================
if [ "${INSTALL_MULTI_GPU}" == "yes" ]; then
    echo "====================="
    echo "Installing NCCL for multi-GPU communication"
    echo "====================="
    conda install -y -c conda-forge nccl
fi


pip install --upgrade tensorflow --no-cache-dir


# =============================================
# PyTorch backend — needed for torchrun multi-GPU
# and for DPA-2/DPA-3 models in general
# =============================================
if [ "${INSTALL_MULTI_GPU}" == "yes" ] && [ "${MULTI_GPU_BACKEND}" == "pytorch" -o "${MULTI_GPU_BACKEND}" == "both" ]; then
    echo "====================="
    echo "Installing PyTorch (CUDA 12.x) for PyTorch backend multi-GPU support"
    echo "====================="
    # IMPORTANT: Must use conda-forge PyTorch, NOT PyPI wheels.
    # PyPI wheels are built with CXX11_ABI_FLAG=0, but DeePMD-kit built from
    # source with gcc-toolset/14 uses CXX11_ABI_FLAG=1. These are incompatible
    # and cause "undefined symbol" errors when loading libdeepmd_op_pt.so.
    # conda-forge builds use CXX11_ABI_FLAG=1, matching our compiler.
    #
    # CONDA_OVERRIDE_CUDA is needed when installing on a non-GPU login node —
    # without it, the solver can't detect the __cuda virtual package and refuses
    # to install CUDA-enabled builds. The value should match the cluster's
    # cudatoolkit version (e.g. 12.8 for cudatoolkit/12.8).
    #
    # The build string pattern "=cuda12*" pins to CUDA 12 builds on conda-forge,
    # where the CUDA variant is encoded in the build string (e.g. cuda126_generic_py311).
    # Ref: https://researchcomputing.princeton.edu/support/knowledge-base/pytorch
    #
    # OLD (broken ABI):
    #   pip install torch --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
    # Derive CUDA version from the loaded cudatoolkit module (e.g. "12.8" from cudatoolkit/12.8)
    _cuda_ver=$(module list 2>&1 | grep -oP 'cudatoolkit/\K[0-9.]+')
    CONDA_OVERRIDE_CUDA="${_cuda_ver}" conda install -y -c conda-forge "pytorch=2.*=cuda12*"
fi


# =============================================
# Horovod — needed for TensorFlow backend multi-GPU
# =============================================
if [ "${INSTALL_MULTI_GPU}" == "yes" ] && [ "${MULTI_GPU_BACKEND}" == "tensorflow" -o "${MULTI_GPU_BACKEND}" == "both" ]; then
    echo "====================="
    echo "Installing Horovod with NCCL support for TensorFlow backend multi-GPU"
    echo "====================="

    # mpi4py is required for Horovod's MPI controller
    pip install mpi4py --no-cache-dir

    # ---------------------------------------------------------
    # Pre-flight: locate NCCL headers and libs before building.
    # Horovod's build often fails silently when it can't find
    # nccl.h — we detect this up front and set explicit paths.
    # ---------------------------------------------------------

    # Search order for NCCL:
    #   1. $CONDA_PREFIX  (from conda install -c conda-forge nccl)
    #   2. System paths commonly set by cudatoolkit modules
    #   3. Common manual-install locations
    NCCL_HEADER_FOUND=""
    NCCL_LIB_FOUND=""

    for candidate_dir in \
        "${CONDA_PREFIX}" \
        "${CUDA_HOME}" \
        "${NCCL_HOME}" \
        "/usr/local/cuda" \
        "/usr/local" \
        "/usr"; do

        # skip empty or nonexistent candidates
        [ -z "${candidate_dir}" ] && continue
        [ -d "${candidate_dir}" ] || continue

        # check for the header (could be in include/ or include/nccl/)
        if [ -f "${candidate_dir}/include/nccl.h" ]; then
            NCCL_HEADER_FOUND="${candidate_dir}/include"
            NCCL_LIB_FOUND="${candidate_dir}/lib"
            echo "  [NCCL] Found nccl.h at ${candidate_dir}/include/nccl.h"
            break
        fi
    done

    if [ -z "${NCCL_HEADER_FOUND}" ]; then
        echo ""
        echo "================================================================"
        echo "  ERROR: Could not find nccl.h in any of the searched paths."
        echo ""
        echo "  Searched: \$CONDA_PREFIX/include, \$CUDA_HOME/include,"
        echo "            /usr/local/cuda/include, /usr/local/include, /usr/include"
        echo ""
        echo "  To fix, try ONE of:"
        echo "    1. conda install -y -c conda-forge nccl   (should have been done above)"
        echo "    2. module load nccl  (if your cluster provides a module)"
        echo "    3. Set NCCL_HOME=/path/to/nccl  before running this script,"
        echo "       where /path/to/nccl/include/nccl.h exists."
        echo "================================================================"
        echo ""
        echo "Skipping Horovod installation. TF multi-GPU will NOT be available."
        echo "You can still use PyTorch multi-GPU via torchrun (if installed)."
    else
        # Also verify the shared library exists alongside the header
        if [ ! -f "${NCCL_LIB_FOUND}/libnccl.so" ] && [ ! -f "${NCCL_LIB_FOUND}/libnccl.so.2" ]; then
            echo ""
            echo "================================================================"
            echo "  WARNING: Found nccl.h at ${NCCL_HEADER_FOUND}/nccl.h"
            echo "           but libnccl.so was NOT found in ${NCCL_LIB_FOUND}/"
            echo ""
            echo "  Will attempt Horovod build anyway — it may fail at link time."
            echo "  If it does, ensure the NCCL shared library is installed."
            echo "================================================================"
            echo ""
        fi

        # Set all NCCL-related env vars explicitly so Horovod's build system
        # doesn't have to guess.  Belt-and-suspenders: set both the granular
        # INCLUDE/LIB vars AND the coarser HOME var.
        export HOROVOD_NCCL_HOME="$(dirname ${NCCL_HEADER_FOUND})"
        export HOROVOD_NCCL_INCLUDE="${NCCL_HEADER_FOUND}"
        export HOROVOD_NCCL_LIB="${NCCL_LIB_FOUND}"

        echo "  [NCCL] HOROVOD_NCCL_HOME    = ${HOROVOD_NCCL_HOME}"
        echo "  [NCCL] HOROVOD_NCCL_INCLUDE = ${HOROVOD_NCCL_INCLUDE}"
        echo "  [NCCL] HOROVOD_NCCL_LIB     = ${HOROVOD_NCCL_LIB}"

        # Core Horovod build flags
        export HOROVOD_WITHOUT_GLOO=1           # skip Gloo — we use MPI
        export HOROVOD_WITH_TENSORFLOW=1        # build TF bindings
        export HOROVOD_GPU_OPERATIONS=NCCL      # use NCCL for GPU allreduce/allgather

        # Also build PT bindings if PyTorch is installed (no harm if not)
        if python -c "import torch" 2>/dev/null; then
            export HOROVOD_WITH_PYTORCH=1
        fi

        echo ""
        echo "  Building Horovod (this may take a few minutes)..."
        pip install horovod --no-cache-dir
        HOROVOD_EXIT_CODE=$?

        if [ ${HOROVOD_EXIT_CODE} -ne 0 ]; then
            echo ""
            echo "================================================================"
            echo "  ERROR: Horovod pip install failed (exit code ${HOROVOD_EXIT_CODE})."
            echo ""
            echo "  Common causes:"
            echo "    - NCCL version mismatch with CUDA. Check:"
            echo "        \$ ls ${NCCL_LIB_FOUND}/libnccl*"
            echo "        \$ nvcc --version"
            echo "    - MPI not found. Ensure 'mpicc' and 'mpicxx' are on PATH:"
            echo "        \$ which mpicc"
            echo "    - Compiler mismatch. TF was built with a specific GCC; check:"
            echo "        \$ python -c \"import tensorflow; print(tensorflow.version.COMPILER_VERSION)\""
            echo ""
            echo "  Horovod NOT installed. TF multi-GPU will NOT be available."
            echo "  You can still use PyTorch multi-GPU via torchrun."
            echo "================================================================"
        else
            # Sanity check: print what Horovod was built with
            echo "====================="
            echo "Horovod build check:"
            horovodrun --check-build
            HVD_CHECK=$?
            if [ ${HVD_CHECK} -ne 0 ]; then
                echo ""
                echo "  WARNING: horovodrun --check-build returned non-zero."
                echo "  Horovod may have been installed without NCCL or TF support."
                echo "  Run 'horovodrun --check-build' manually to diagnose."
            fi
            echo "====================="
        fi
    fi
fi



# for dpdata
echo "======================="
echo "Installing dpdata, ase, parallel, and other python packages"
echo "======================="
conda install -y conda-forge::dpdata
#################
# Install ase and parallel for post-processing and running multiple simulations in parallel
conda install -y -c conda-forge ase parallel
#################




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

# =============================================
# Build DeePMD-kit Python interface
# =============================================
export DP_VARIANT=cuda
# Point CMake to the system CUDA toolkit (from 'module load cudatoolkit/...')
# so it doesn't pick up the incomplete conda-forge CUDA headers instead.
export CUDAToolkit_ROOT=$CUDA_HOME

# Enable PyTorch C++ OPs if we installed PyTorch
# This lets the LAMMPS plugin load the PT backend for inference too.
if [ "${INSTALL_MULTI_GPU}" == "yes" ] && [ "${MULTI_GPU_BACKEND}" == "pytorch" -o "${MULTI_GPU_BACKEND}" == "both" ]; then
    export DP_ENABLE_PYTORCH=1
fi

pip install .

# will possibly fail ^ in Della >> if so, do the following
conda install -y -c conda-forge dscribe==1.2.2
conda install -y -c conda-forge click>=7.0
conda install -y -c conda-forge scipy scikit-learn ase umap-learn pyyaml tqdm pandas
pip install .

cd $deepmd_source_dir/source
mkdir build
cd build

cmake3 \
    -DUSE_TF_PYTHON_LIBS=TRUE \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DUSE_CUDA_TOOLKIT=TRUE \
    -DLAMMPS_SOURCE_ROOT="${deepmd_source_dir}/source/lammps-stable_2Aug2023_update3" \
    -DDP_USING_C_API=OFF \
    ..

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

make -j24 install && make lammps

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
make -j24 install


# activate Plumed2 relevant env variables when conda env gets activated
# echo "Add the text at the end of this script in ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"


# ==============================
cat << 'EOF' > "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
# Activate PLUMED environment variables

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

# =============================================
# LAMMPS cmake — conditionally enable MPI+OpenMP for multi-GPU MD
# =============================================
#
# Physics note: LAMMPS spatial decomposition splits the simulation box across
# MPI ranks. Each rank calls DeePMD inference for its local atoms on its own GPU.
# This is beneficial for large systems (>~5k atoms) where the per-rank atom count
# is still large enough that GPU utilization stays high. For small systems, the
# MPI halo exchange overhead can exceed the speedup — profile before committing.

if [ "${LAMMPS_MULTI_GPU}" == "yes" ]; then
    echo "  [LAMMPS] Building with BUILD_MPI=yes, BUILD_OMP=yes for multi-GPU MD"

    cmake3 \
            -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
            -DCMAKE_INSTALL_PATH=${CONDA_PREFIX} \
            -DCMAKE_CXX_FLAGS_RELEASE="-Ofast -fopenmp" \
            -DBUILD_OMP=yes \
            -DBUILD_MPI=yes \
            -DCMAKE_CXX_COMPILER=mpicxx \
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
else
    echo "  [LAMMPS] Building single-GPU version (LAMMPS_MULTI_GPU=no)"

    cmake3 \
            -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
            -DCMAKE_INSTALL_PATH=${CONDA_PREFIX} \
            -DCMAKE_CXX_COMPILER=mpicxx \
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
fi

make -j24 install  


# Making a symbolic link to the lmp_mpi executable in the LAMMPS source directory. While this executable is technically
# accessible from any conda environment since it's in ~/.local/bin, it's not guaranteed that the auxiliary packages will work.
# ln -s $PWD/<name of the executable> ~/.local/bin/lmp_plmd # likely "lmp"
rm -f ~/.local/bin/${lmp_exec_name}
ln -s $PWD/lmp ~/.local/bin/${lmp_exec_name}


# =============================================
# POST-INSTALL SUMMARY
# =============================================
echo ""
echo "========================================================"
echo "  INSTALLATION COMPLETE"
echo "========================================================"
echo ""
echo "Conda environment:  ${conda_env}"
echo "LAMMPS executable:  ${lmp_exec_name}"
echo ""

if [ "${INSTALL_MULTI_GPU}" == "yes" ]; then
    echo "--- Multi-GPU training support ---"
    if [ "${MULTI_GPU_BACKEND}" == "pytorch" -o "${MULTI_GPU_BACKEND}" == "both" ]; then
        echo ""
        echo "[PyTorch backend]  (recommended for DPA-2, DPA-3)"
        echo "  Single-node, 4 GPUs:"
        echo "    python -m torch.distributed.run --nproc_per_node=4 --no-python dp --pt train input.json"
        echo ""
        echo "  Multi-node (Slurm):"
        echo "    python -m torch.distributed.run --nnodes=\${SLURM_NNODES} --nproc_per_node=4 \\"
        echo "             --rdzv_id=\${SLURM_JOB_ID} --rdzv_backend=c10d \\"
        echo "             --rdzv_endpoint=\$(hostname):29500 \\"
        echo "             --no-python dp --pt train input.json"
    fi
    if [ "${MULTI_GPU_BACKEND}" == "tensorflow" -o "${MULTI_GPU_BACKEND}" == "both" ]; then
        echo ""
        echo "[TensorFlow backend]  (legacy se_e2_a, DPA-1)"
        echo "  Single-node, 4 GPUs:"
        echo "    CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 -H localhost:4 \\"
        echo "        dp train --mpi-log=workers input.json"
    fi
    echo ""
    echo "Remember: effective batch_size = batch_size * N_GPUs."
    echo "  -> Manually reduce decay_steps by ~N_GPUs in input.json."
    echo "  -> Check log for 'world size: N' to confirm distributed mode."
    echo "========================================================"
fi

if [ "${LAMMPS_MULTI_GPU}" == "yes" ]; then
    echo ""
    echo "--- Multi-GPU LAMMPS MD support ---"
    echo ""
    echo "  Built with BUILD_MPI=yes + BUILD_OMP=yes."
    echo "  Each MPI rank gets one GPU via spatial decomposition."
    echo ""
    echo "  Slurm (recommended — automatic GPU binding):"
    echo "    srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 \\"
    echo "        ${lmp_exec_name} -in input.lammps"
    echo ""
    echo "  mpirun (manual GPU binding via rank → device mapping):"
    echo "    mpirun -np 4 --map-by slot \\"
    echo "        bash -c 'export CUDA_VISIBLE_DEVICES=\$OMPI_COMM_WORLD_LOCAL_RANK; \\"
    echo "                 ${lmp_exec_name} -in input.lammps'"
    echo ""
    echo "  Tip: add to your LAMMPS input for DeePMD multi-GPU:"
    echo "    package omp 1        # 1 OMP thread per rank"
    echo "    processors * * *     # auto spatial decomposition"
    echo ""
    echo "  Best for large systems (>~5k atoms). Profile short runs first."
    echo "========================================================"
fi

cd ${parent_dir}

# --- Installation timer: print elapsed time ---
INSTALL_ELAPSED=$(( SECONDS - INSTALL_START_TIME ))
INSTALL_MIN=$(( INSTALL_ELAPSED / 60 ))
INSTALL_SEC=$(( INSTALL_ELAPSED % 60 ))
echo ""
echo "========================================================"
echo "  Total installation time: ${INSTALL_MIN}m ${INSTALL_SEC}s"
echo "  Finished at $(date)"
echo "========================================================"