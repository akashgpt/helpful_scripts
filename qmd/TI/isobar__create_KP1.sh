#!/bin/bash
# set -euo pipefail

# Usage: source $HELP_SCRIPTS_TI/isobar__create_KP1.sh > log.isobar__create_KP1 2>&1 &
#        nohup $HELP_SCRIPTS_TI/isobar__create_KP1.sh > log.isobar__create_KP1 2>&1 &
# Author: Akash Gupta

#--------------------------------------------------------------
# Script to generate KP1* directories and submit NVT VASP runs
#--------------------------------------------------------------






#######
CELL_SIZE_P50_T3500_w_Fe=8.5
CELL_SIZE_P250_T6500_w_Fe=7.8
CELL_SIZE_P500_T9000_w_Fe=7.4
CELL_SIZE_P1000_T13000_w_Fe=7.0

CELL_SIZE_P50_T3500_w_MgSiO3=10.9
CELL_SIZE_P250_T6500_w_MgSiO3=9.7
CELL_SIZE_P500_T9000_w_MgSiO3=9.0
CELL_SIZE_P1000_T13000_w_MgSiO3=8.4
#######


#-------------------------
# Free energy scaling factor
#-------------------------
SCALEE_CHOSEN=1.0                    # Scale factor for free energy calculations

#-------------------------
# Derived constants
#-------------------------
kB=0.00008617333262145               # Boltzmann constant in eV/K

#-------------------------
# Working directory
#-------------------------
ISOBAR_CALC_dir=$(pwd)
CONFIG_dir=$(dirname "$ISOBAR_CALC_dir") # CONFIG_dir is the parent directory of isobar_calc
PT_dir=$(dirname "$CONFIG_dir") # PT_dir is the parent directory of CONFIG_dir
PT_dir_name=$(basename "$PT_dir")

# parent directory
COMPOSITION_dir=$(dirname "$PT_dir")
COMPOSITION_dir_name=$(basename "$COMPOSITION_dir")

echo "Current time: $(date)"
echo "PT directory: $PT_dir"
echo "PT directory name: $PT_dir_name"
echo "COMPOSITION directory: $COMPOSITION_dir"
echo "COMPOSITION directory name: $COMPOSITION_dir_name"
echo "ISOBAR_CALC_dir: $ISOBAR_CALC_dir"
echo ""


SETUP_dir=$PT_dir/master_setup_TI


# read all the above from input.calculate_GFE file where each line is a key-value pair, e.g. TEMP_CHOSEN=13000
PARAMETER_FILE=${SETUP_dir}/input.calculate_GFE
if [ -f "$PARAMETER_FILE" ]; then
    while IFS='=' read -r key value; do
        case $key in
            TEMP_CHOSEN) TEMP_CHOSEN="$value" ;;
            PSTRESS_CHOSEN_GPa) PSTRESS_CHOSEN_GPa="$value" ;;
            NPAR_CHOSEN) NPAR_CHOSEN="$value" ;;
            POTIM_CHOSEN) POTIM_CHOSEN="$value" ;;
            NBANDS_CHOSEN) NBANDS_CHOSEN="$value" ;;
            KPAR_CHOSEN_111) KPAR_CHOSEN_111="$value" ;;
            KPAR_CHOSEN_222) KPAR_CHOSEN_222="$value" ;;
            WAIT_TIME_VLONG) WAIT_TIME_VLONG="$value" ;;
            WAIT_TIME_LONG) WAIT_TIME_LONG="$value" ;;
            WAIT_TIME_SHORT) WAIT_TIME_SHORT="$value" ;;
            N_FRAMES_hp_calculations) N_FRAMES_hp_calculations="$value" ;;
        esac
    done < "$PARAMETER_FILE"
else
    echo "Parameter file not found: $PARAMETER_FILE"
    exit 1
fi


# if TEMP_CHOSEN and PSTRESS_CHOSEN_GPa are not a number, exit
if ! [[ "$TEMP_CHOSEN" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "TEMP_CHOSEN is not a number: $TEMP_CHOSEN. Exiting. Please check the input.calculate_GFE file."
    exit 1
fi
if ! [[ "$PSTRESS_CHOSEN_GPa" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "PSTRESS_CHOSEN_GPa is not a number: $PSTRESS_CHOSEN_GPa. Exiting. Please check the input.calculate_GFE file."
    exit 1
fi

# Extract TEMP_CHOSEN_ARRAY from the names of the directories in the current directory -- their format is "T<TEMP_CHOSEN_i>"
TEMP_CHOSEN_ARRAY=($(ls -d T* | sed 's/T//g' | sort -n))
echo "TEMP_CHOSEN_ARRAY: ${TEMP_CHOSEN_ARRAY[@]}"

# SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN" | bc -l)  # Gaussian smearing sigma

PSTRESS_CHOSEN=$(echo "$PSTRESS_CHOSEN_GPa * 10" | bc -l)  # Convert GPa to kBar

#-------------------------
# Print the parameters
echo "------------------------"
echo "Simulation parameters:"
echo "TEMP_CHOSEN_ARRAY: ${TEMP_CHOSEN_ARRAY[@]}"
echo "PSTRESS_CHOSEN_GPa: $PSTRESS_CHOSEN_GPa"
echo "PSTRESS_CHOSEN (kBar): $PSTRESS_CHOSEN"
echo "NPAR_CHOSEN: $NPAR_CHOSEN"
echo "POTIM_CHOSEN: $POTIM_CHOSEN"
echo "NBANDS_CHOSEN: $NBANDS_CHOSEN"
echo "KPAR_CHOSEN_111: $KPAR_CHOSEN_111"
echo "KPAR_CHOSEN_222: $KPAR_CHOSEN_222"
echo "WAIT_TIME_VLONG: $WAIT_TIME_VLONG"
echo "WAIT_TIME_LONG: $WAIT_TIME_LONG"
echo "WAIT_TIME_SHORT: $WAIT_TIME_SHORT"
echo "N_FRAMES_hp_calculations: $N_FRAMES_hp_calculations"
echo "SCALEE_CHOSEN: $SCALEE_CHOSEN"
echo "-------------------------"
echo ""
#-------------------------

# Check if the folder name contains both TEMP_CHOSEN and PSTRESS_CHOSEN_GPa
if [[ "$PT_dir_name" == *"${TEMP_CHOSEN}"* && "$PT_dir_name" == *"${PSTRESS_CHOSEN_GPa}"* ]]; then
    echo "Folder name contains TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
else
    echo "Folder name does NOT contain both TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
    exit 1
fi

#-------------------------
# Main loop: find KP1 dirs
#-------------------------
while IFS= read -r -d '' parent; do
    # echo "Processing $parent"
    cd "$parent" || exit


    KP1_dir=$(pwd)
    V_est_dir=$(dirname "$KP1_dir") # parent directory is V_est_dir
    ISOBAR_T_dir=$(dirname "$V_est_dir") # parent directory is ISOBAR_T_dir

    cd $KP1_dir || exit

    echo ""
    echo "Processing KP1 directory: $KP1_dir"
    echo "V_est_dir: $V_est_dir"
    echo "ISOBAR_T_dir: $ISOBAR_T_dir"
    echo ""



    # Submit the VASP job
    sbatch RUN_VASP.sh

    # list just the job IDs for your user, sort them numerically, and take the last one
    LAST_JOB_ID=$(squeue -u $USER -h -o "%i" | sort -n | tail -1)
    echo "Started VASP in $KP1_dir"
    echo "JOB_ID_KP1: $LAST_JOB_ID"
    echo ""

    # Return to root directory for next iteration
    cd $ISOBAR_CALC_dir || exit

done < <(find . -type d -name KP1 -print0)

# Final status message
echo "All ISOBAR VASP jobs started in KP1 directories in ${ISOBAR_CALC_dir}."
