#!/bin/bash
# set -euo pipefail

# Usage: source $HELP_SCRIPTS_TI/create_KP1.sh > log.create_KP1 2>&1 &
# Author: Akash Gupta

#--------------------------------------------------------------
# Script to generate KP1* directories and submit NVT VASP runs
#--------------------------------------------------------------



# big warning message saying better to do 4 - KP 222 calculations rather than 111
echo ""
echo ""
echo ""
echo "#==========================================#"
echo "#==========================================#"
echo "#==========================================#"
echo "WARNING: It is better to do 4 {KP 222} calculations rather than 4 {KP 111} + hp_calculcations"
echo "Not sure though ..."
echo "Thermal pressure correction is likely larger than a simple external pressure offset between KPOINTS 222 vs 111. Not 100% sure though."
echo "#==========================================#"
echo "#==========================================#"
echo "#==========================================#"
echo ""
echo ""
echo ""



#-------------------------
# Simulation parameters
#-------------------------
# TEMP_CHOSEN=6500                     # Target temperature (K)
# PSTRESS_CHOSEN_GPa=250               # Pressure offset (GPa), not used here
# NBANDS_CHOSEN=784                    # Number of bands: 784 for MgSiO3-He; use 560 for Fe-He
# POTIM_CHOSEN=0.5                     # MD time step (fs)
# NPAR_CHOSEN=14                       # Parallelization: cores per node; TIGER3=14, STELLAR=16
# KPAR_CHOSEN_111=1                    # K-point parallelization for KPOINTS 1×1×1
# KPAR_CHOSEN_222=4                    # K-point parallelization for KPOINTS 2×2×2

# #-------------------------
# # Timing controls (in minutes)
# #-------------------------
# WAIT_TIME_VLONG=600                  # Very long jobs
# WAIT_TIME_LONG=60                    # Long jobs
# WAIT_TIME_SHORT=10                   # Short jobs


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
PT_dir=$(pwd)
PT_dir_name=$(basename "$PT_dir")

# parent directory
COMPOSITION_dir=$(dirname "$PT_dir")
COMPOSITION_dir_name=$(basename "$COMPOSITION_dir")

echo "Current time: $(date)"
echo "Current PT directory: $PT_dir"
echo "Current PT directory name: $PT_dir_name"
echo "Current COMPOSITION directory: $COMPOSITION_dir"
echo "Current COMPOSITION directory name: $COMPOSITION_dir_name"
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



SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN" | bc -l)  # Gaussian smearing sigma

PSTRESS_CHOSEN=$(echo "$PSTRESS_CHOSEN_GPa * 10" | bc -l)  # Convert GPa to kBar

#-------------------------
# Print the parameters
echo "------------------------"
echo "Simulation parameters:"
echo "TEMP_CHOSEN: $TEMP_CHOSEN"
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

    # parent directory is V_est_dir
    V_est_dir=$(dirname "$KP1_dir")
    cd "$V_est_dir" || exit
    # parent directory is CONFIG_dir
    CONFIG_dir=$(dirname "$V_est_dir")
    cd "$CONFIG_dir" || exit
    LOCAL_SETUP_dir=$CONFIG_dir/setup_TI
    cd $KP1_dir || exit

    echo "CONFIG_dir: $CONFIG_dir"
    echo "LOCAL_SETUP_dir: $LOCAL_SETUP_dir"
    echo "Processing KP1 directory: $KP1_dir"


    # Enter parent directory and perform analysis

    # echo current directory
    # echo "Current directory: ${KP1_dir}"

    # check if POSCAR file exists
    if [ ! -f POSCAR ]; then
        cp ${LOCAL_SETUP_dir}/POSCAR_NPT $KP1_dir/POSCAR
    fi

    # in POSCAR, replace second line with CELL_SIZE
    # sed -i "2s/.*/$CELL_SIZE/" POSCAR
    if [[ "$COMPOSITION_dir_name" == *"Fe"* ]]; then
        if [[ "$PT_dir_name" == *"P50_T3500"* ]]; then
            CELL_SIZE=$CELL_SIZE_P50_T3500_w_Fe
        elif [[ "$PT_dir_name" == *"P250_T6500"* ]]; then
            CELL_SIZE=$CELL_SIZE_P250_T6500_w_Fe
        elif [[ "$PT_dir_name" == *"P500_T9000"* ]]; then
            CELL_SIZE=$CELL_SIZE_P500_T9000_w_Fe
        elif [[ "$PT_dir_name" == *"P1000_T13000"* ]]; then
            CELL_SIZE=$CELL_SIZE_P1000_T13000_w_Fe
        else
            echo "Unknown folder name: $PT_dir_name"
            exit 1
        fi
    elif [[ "$COMPOSITION_dir_name" == *"MgSiO3"* ]]; then
        if [[ "$PT_dir_name" == *"P50_T3500"* ]]; then
            CELL_SIZE=$CELL_SIZE_P50_T3500_w_MgSiO3
        elif [[ "$PT_dir_name" == *"P250_T6500"* ]]; then
            CELL_SIZE=$CELL_SIZE_P250_T6500_w_MgSiO3
        elif [[ "$PT_dir_name" == *"P500_T9000"* ]]; then
            CELL_SIZE=$CELL_SIZE_P500_T9000_w_MgSiO3
        elif [[ "$PT_dir_name" == *"P1000_T13000"* ]]; then
            CELL_SIZE=$CELL_SIZE_P1000_T13000_w_MgSiO3
        else
            echo "Unknown folder name: $PT_dir_name"
            exit 1
        fi
    else
        echo "Unknown folder name: $COMPOSITION_dir_name"
        exit 1
    fi
    echo "POSCAR CELL_SIZE: $CELL_SIZE"
    sed -i "2s/.*/$CELL_SIZE/" POSCAR




    # Copy VASP run scripts and input templates
    cp ${SETUP_dir}/RUN_VASP_NPT.sh $KP1_dir/RUN_VASP.sh
    cp ${SETUP_dir}//POTCAR         $KP1_dir/POTCAR
    cp ${SETUP_dir}//KPOINTS_111    $KP1_dir/KPOINTS   # KPOINTS 1×1×1
    cp ${SETUP_dir}/INCAR_NPT    $KP1_dir/INCAR

    # Populate INCAR placeholders
    sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" $KP1_dir/INCAR
    sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" $KP1_dir/INCAR
    sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" $KP1_dir/INCAR
    sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" $KP1_dir/INCAR
    sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_111}/" $KP1_dir/INCAR
    sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" $KP1_dir/INCAR
    sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" $KP1_dir/INCAR
    sed -i "s/__PSTRESS_CHOSEN__/${PSTRESS_CHOSEN}/" $KP1_dir/INCAR

    # Submit the VASP job
    sbatch RUN_VASP.sh

    # list just the job IDs for your user, sort them numerically, and take the last one
    LAST_JOB_ID=$(squeue -u $USER -h -o "%i" | sort -n | tail -1)
    echo "Started VASP in $KP1_dir"
    echo "JOB_ID_KP1: $LAST_JOB_ID"
    echo ""

    # Return to root directory for next iteration
    cd $PT_dir || exit

done < <(find . -type d -name KP1 -print0)

# Final status message
echo "All VASP jobs started in KP1 directories in ${PT_dir}."
