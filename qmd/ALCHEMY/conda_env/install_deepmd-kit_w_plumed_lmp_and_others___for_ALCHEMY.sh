#!/bin/bash

###############################
# Summary:
# This script installs DeePMD-kit, PLUMED, and LAMMPS together with required
# PLUMED patches into a single conda environment (ALCHEMY_env).
# The script is intended to be run on a cluster with GPU support (used for
# DeePMD calculations) but can also run on a purely CPU system.
#
# Installation order:
#   1. Create conda env with Python 3.11 + conda packages (cudnn, gsl, fftw,
#      dpdata, ase, etc.)
#   2. Install TensorFlow via pip (needed for DeePMD build)
#   3. Build DeePMD-kit from source (pip install . + cmake/make)
#   4. Additional conda installs (dscribe, scipy, scikit-learn, etc.)
#      -- these may clobber TF pip dependencies
#   5. Build PLUMED from source (with custom CV patches)
#   6. Build LAMMPS from source (with DEEPMD + PLUMED packages)
#   7. Reinstall TensorFlow via pip (--force-reinstall) to repair any
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
# LAMMPS e.g.:
#   lmp -in <name of lammps input file>
#
# author: akashgpt and jinalee
###############################

# =============================
conda_env_name="ALCHEMY_env" # name of the conda environment to create and install everything in
dir_w_plumed_patches="/projects/BURROWS/akashgpt/lammp*"
# dir_w_plumed_patches="/projects/bguf/akashgpt/lammp*"
# =============================



deepmd_plmd_lmp_misc__folder_name="deepmd-kit_and_others__${conda_env_name}"
conda_env="${conda_env_name}"
lmp_exec_name="lmp"


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
pip install --upgrade tensorflow --no-cache-dir




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
# export CUDAToolkit_ROOT=$CUDA_HOME
pip install .

# will possibly fail ^ in Della >> if so, do the following
conda install -y -c conda-forge "dscribe==1.2.2"
conda install -y -c conda-forge "click>=7.0"
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

# version 1 -- works but without multiple cores + uses gpu for deepmd
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
pip install --upgrade --force-reinstall tensorflow --no-cache-dir

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
echo "REMINDER: Always add 'export PYTHONNOUSERSITE=1' to your sbatch scripts"
echo "          before running dp or lmp commands."
