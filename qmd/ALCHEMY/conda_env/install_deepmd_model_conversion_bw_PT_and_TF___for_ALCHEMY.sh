#!/bin/bash

###############################
# Summary:
# This script creates a lightweight conda environment for converting DeePMD-kit
# model files between PyTorch (.pth) and TensorFlow (.pb) backends using the
# `dp convert-backend` command.
#
# PURPOSE:
# The ALCHEMY_env (TF) and ALCHEMY_env__PT (PT) environments are each built
# from source with only ONE backend's C++ ops compiled. This means neither
# environment can directly convert models between formats — the PT env can't
# write .pb files (no TF C++ ops), and the TF env can't read .pth files (no PT
# C++ ops).
#
# This environment uses the OFFICIAL DeePMD-kit pip wheel (v3.1.3), which ships
# with BOTH TF and PT C++ ops pre-compiled (ENABLE_TENSORFLOW=1 and
# ENABLE_PYTORCH=1 in the wheel's run_config.ini). This makes it the only
# environment that can do one-step model conversion in either direction.
#
# WHY A SEPARATE ENVIRONMENT:
# The pip wheel bundles its own C++ libraries (libdeepmd_op.so, libdeepmd_op_pt.so)
# that are linked against specific TF/PT versions. These can conflict with the
# from-source builds in ALCHEMY_env and ALCHEMY_env__PT. A dedicated conversion
# env avoids any library conflicts and is small (no LAMMPS, no PLUMED, no
# source build needed).
#
# SUPPORTED CONVERSIONS:
#   dp convert-backend model.pth model.pb    # PT → TF
#   dp convert-backend model.pb  model.pth   # TF → PT
#   dp convert-backend model.pth model.dp    # PT → DP (backend-agnostic)
#   dp convert-backend model.dp  model.pb    # DP → TF
#   dp convert-backend model.pb  model.dp    # TF → DP
#   dp convert-backend model.dp  model.pth   # DP → PT
#
# LIMITATION — COMPRESSED MODELS:
#   Compressed models (.pth from `dp --pt compress` or .pb from `dp compress`)
#   CANNOT be converted — the compression data tensors cause state_dict shape
#   mismatches during deserialization. You must convert the UNCOMPRESSED frozen
#   model first, then compress in the target backend's native environment:
#
#     # Example workflow: Train PT → Convert → Compress in TF → LAMMPS
#     # Step 1: Train and freeze in ALCHEMY_env__PT
#     conda activate ALCHEMY_env__PT
#     dp --pt train input.json
#     dp --pt freeze -o model.pth
#
#     # Step 2: Convert in dp_model_conv_PT_TF
#     conda activate dp_model_conv_PT_TF
#     export PYTHONNOUSERSITE=1
#     dp convert-backend model.pth model.pb
#
#     # Step 3: Compress in ALCHEMY_env (TF) — needs GPU node
#     conda activate ALCHEMY_env
#     dp compress -i model.pb -o model_comp.pb
#     #   NOTE: If `dp compress` fails with "start_lr not found", provide the
#     #   original training input.json via:
#     #     dp compress -i model.pb -o model_comp.pb -t input.json
#     #   The embedded training script in converted models may be incomplete.
#     #   If this still fails, use the uncompressed model.pb directly in LAMMPS
#     #   (slightly slower but works fine).
#
#     # Step 4: Run LAMMPS with compressed TF model (fastest)
#     conda activate ALCHEMY_env
#     lmp -in in.lammps   # pair_style deepmd model_comp.pb
#
# ALTERNATIVE TWO-STEP CONVERSION (without this environment):
#   If you prefer not to create this env, you can use the .dp intermediate
#   format through the existing environments:
#     conda activate ALCHEMY_env__PT && dp convert-backend model.pth model.dp
#     conda activate ALCHEMY_env     && dp convert-backend model.dp  model.pb
#   This works because .dp is a pure HDF5 serialization format that doesn't
#   need any ML framework C++ ops to read/write.
#
# Installation order:
#   1. Create conda env with Python 3.11
#   2. Install DeePMD-kit v3.1.3 from pip (official wheel with both backends)
#   3. Install PyTorch from pip (needed by PT backend ops)
#   4. Install remaining dependencies (mpmath, pyyaml, pandas, mpich)
#   5. Verify both backends work and test a conversion
#
# Usage: source <name of this script>
# Log file will be created in the same directory as log.deepmd-kit_and_others__<env_name>
#
# IMPORTANT: Always set PYTHONNOUSERSITE=1 when using this environment to
# prevent stale user-local packages (from ~/.local) from interfering.
#
# author: akashgpt (with Claude Code assistance)
###############################

# =============================
conda_env_name="dp_model_conv_PT_TF"	# name of the conda environment
deepmd_kit_version="3.1.3"				# DeePMD-kit version to install
# =============================


deepmd_plmd_lmp_misc__folder_name="deepmd-kit_and_others__${conda_env_name}"

# send all output to log file
exec > >(tee -i log.${deepmd_plmd_lmp_misc__folder_name})
exec 2>&1

echo "=============================================="
echo "  DeePMD-kit Model Conversion Environment"
echo "  (PT <-> TF backend conversion)"
echo "=============================================="
echo "Date|Time: $(date)"
echo "Hostname: $(hostname)"
echo "conda_env_name: ${conda_env_name}"
echo "deepmd_kit_version: ${deepmd_kit_version}"
echo "=============================================="


# Load minimal modules (only need anaconda for conda)
echo ""
echo "====================="
echo "Loading modules"
echo "====================="
if [[ $(hostname) == *"della"* ]]; then
	module purge
	module load anaconda3/2025.12
elif [[ $(hostname) == *"stellar"* ]]; then
	module purge
	module load anaconda3/2025.12
elif [[ $(hostname) == *"delta"* ]]; then
	module reset
	module load miniforge3-python
else
	echo "Unknown cluster. Please load anaconda module manually."
	return 1
fi

# Ensure the conda shell function is available for 'conda activate'.
if command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook)"
fi

echo ""
echo "Required modules:"
module list
echo "====================="


# Check if env already exists
if conda env list 2>/dev/null | grep -q "^${conda_env_name} "; then
	echo ""
	echo "WARNING: Conda environment '${conda_env_name}' already exists."
	echo "         Remove it first with: conda env remove -n ${conda_env_name}"
	echo "         Exiting."
	return 1
fi


# Step 1: Create conda env with Python 3.11
echo ""
echo "====================="
echo "Step 1: Creating conda environment '${conda_env_name}' with Python 3.11"
echo "====================="
conda create -y --name ${conda_env_name} python=3.11
conda activate ${conda_env_name}
pip install --upgrade pip

echo "Python location: $(which python3)"
echo "Python version: $(python3 --version)"
echo "Pip location: $(which pip)"


# Step 2: Install DeePMD-kit from pip (official wheel with both TF+PT C++ ops)
echo ""
echo "====================="
echo "Step 2: Installing DeePMD-kit v${deepmd_kit_version} from pip"
echo "        (official wheel with ENABLE_TENSORFLOW=1, ENABLE_PYTORCH=1)"
echo "====================="
pip install "deepmd-kit[gpu,cu12]==${deepmd_kit_version}"

echo "DeePMD-kit installed: $(pip show deepmd-kit 2>/dev/null | grep Version)"


# Step 3: Install PyTorch (not included as a deepmd-kit dependency by default)
echo ""
echo "====================="
echo "Step 3: Installing PyTorch (pip, CUDA)"
echo "====================="
pip install torch

echo "PyTorch installed: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'FAILED')"


# Step 4: Install remaining dependencies that deepmd-kit needs at runtime
echo ""
echo "====================="
echo "Step 4: Installing remaining dependencies"
echo "====================="
pip install mpmath pyyaml pandas mpich

echo "All dependencies installed."


# Step 5: Set up PYTHONNOUSERSITE in activation script
echo ""
echo "====================="
echo "Step 5: Setting up activation script (PYTHONNOUSERSITE=1)"
echo "====================="
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat << 'EOF' > "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
#!/bin/bash
# Prevent ~/.local/lib/pythonX.Y/site-packages from leaking stale packages
# into this conda environment (common source of NumPy/pyarrow/pandas errors).
export PYTHONNOUSERSITE=1
EOF
echo "Created ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"


# Step 6: Verify both backends
echo ""
echo "====================="
echo "Step 6: Verifying both backends are available"
echo "====================="
export PYTHONNOUSERSITE=1

python3 -c "
from deepmd.env import GLOBAL_CONFIG
print(f'  enable_tensorflow: {GLOBAL_CONFIG[\"enable_tensorflow\"]}')
print(f'  enable_pytorch:    {GLOBAL_CONFIG[\"enable_pytorch\"]}')
print(f'  dp_variant:        {GLOBAL_CONFIG[\"dp_variant\"]}')
print(f'  model_version:     {GLOBAL_CONFIG[\"model_version\"]}')
import tensorflow as tf
print(f'  TensorFlow:        {tf.__version__}')
import torch
print(f'  PyTorch:           {torch.__version__}')
" 2>&1 | grep -v "^I0000\|^WARNING\|^2026.*oneDNN\|^To enable\|^E0000"

# Check that enable_tensorflow and enable_pytorch are both "1"
_check=$(python3 -c "
from deepmd.env import GLOBAL_CONFIG
tf_ok = GLOBAL_CONFIG['enable_tensorflow'] == '1'
pt_ok = GLOBAL_CONFIG['enable_pytorch'] == '1'
print('PASS' if (tf_ok and pt_ok) else 'FAIL')
" 2>/dev/null)

if [ "${_check}" != "PASS" ]; then
	echo ""
	echo "ERROR: Both backends are NOT enabled. Something went wrong."
	echo "       The pip wheel should have ENABLE_TENSORFLOW=1 and ENABLE_PYTORCH=1."
	echo "       Check the log above for errors."
	return 1
else
	echo ""
	echo "Both backends verified: TF=1, PT=1"
fi


# Step 7: Test conversion
echo ""
echo "====================="
echo "Step 7: Quick self-test"
echo "====================="
echo "  Testing dp convert-backend --help ..."
dp convert-backend -h > /dev/null 2>&1 && echo "  dp convert-backend: OK" || echo "  dp convert-backend: FAILED"


echo ""
echo "=============================================="
echo "  INSTALLATION COMPLETE"
echo "  Date|Time: $(date)"
echo "=============================================="
echo ""
echo "Conda environment:  ${conda_env_name}"
echo "DeePMD-kit version: ${deepmd_kit_version}"
echo ""
echo "--- Usage ---"
echo ""
echo "  conda activate ${conda_env_name}"
echo "  export PYTHONNOUSERSITE=1    # also set automatically on activation"
echo ""
echo "  # Convert PT model to TF:"
echo "  dp convert-backend model.pth model.pb"
echo ""
echo "  # Convert TF model to PT:"
echo "  dp convert-backend model.pb model.pth"
echo ""
echo "  # Convert to backend-agnostic format:"
echo "  dp convert-backend model.pth model.dp"
echo ""
echo "--- Full workflow: Train (PT) → Convert → Compress (TF) → LAMMPS ---"
echo ""
echo "  # 1. Train + freeze in ALCHEMY_env__PT"
echo "  conda activate ALCHEMY_env__PT"
echo "  dp --pt train input.json && dp --pt freeze -o model.pth"
echo ""
echo "  # 2. Convert in ${conda_env_name}"
echo "  conda activate ${conda_env_name}"
echo "  dp convert-backend model.pth model.pb"
echo ""
echo "  # 3. Compress in ALCHEMY_env (on GPU node)"
echo "  conda activate ALCHEMY_env"
echo "  dp compress -i model.pb -o model_comp.pb"
echo ""
echo "  # 4. Run LAMMPS with TF model (fastest)"
echo "  conda activate ALCHEMY_env"
echo "  lmp -in in.lammps   # pair_style deepmd model_comp.pb"
echo ""
echo "=============================================="
echo "NOTE: Compressed models cannot be converted. Always convert the"
echo "      uncompressed frozen model, then compress in the target env."
echo "=============================================="
