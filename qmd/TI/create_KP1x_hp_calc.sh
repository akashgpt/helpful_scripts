#!/bin/bash
# set -euo pipefail

# Usage: source $HELP_SCRIPTS_TI/create_KP1x_hp_calc.sh > log.create_KP1x_hp_calc 2>&1 &
# Author: Akash Gupta


# while IFS= read -r -d '' parent; do
#     # …look for its immediate subdirectories starting with “KP1”
#     for child in "$parent"/KP1*; do
#         if [ -d "$child" ]; then
#         touch "$child/done_KP1x"
#         # P_RUN=$(grep Pressure analysis/peavg.out  | awk '{print $3}')
#         # l_ase; python $HELP_SCRIPTS_vasp/eos* -p $P_RUN -m 0 -e 0.01 -hp 1
#         echo "Touched $child/done_KP1x"
#         fi
#     done
# done < <(find . -type d -name KP1 -print0)



#--------------------------------------------------------------
# Driver script to set up and launch VASP runs in KP1*/hp_calculations
#--------------------------------------------------------------

# # Thermostat and MD settings
# TEMP_CHOSEN=13000            # Target temperature (K)
# PSTRESS_CHOSEN_GPa=1000     # Pressure offset (GPa), not used here
# NBANDS_CHOSEN=784            # Number of electronic bands (e.g., 560 for Fe-He; 784 for MgSiO3-He)
# POTIM_CHOSEN=0.5             # MD timestep (fs)
# NPAR_CHOSEN=14               # Parallelization: cores/node × nodes (e.g., TIGER3=14, STELLAR=16)

# # K-point parallelization choices
# KPAR_CHOSEN_111=1            # KPAR for 1×1×1 k-point mesh
# KPAR_CHOSEN_222=4            # KPAR for 2×2×2 k-point mesh

# # Job wait-time thresholds (mins)
# WAIT_TIME_VLONG=600          # Very long jobs
# WAIT_TIME_LONG=60            # Long jobs
# WAIT_TIME_SHORT=10           # Short jobs

# Scaling parameter for alchemical transformations
SCALEE_CHOSEN=1.0            # Lambda scaling factor

# Physical constants
kB=0.00008617333262145       # Boltzmann constant (eV/K)
# Compute sigma = k_B * T (for smearing)
# SIGMA_CHOSEN=$(echo "${kB} * ${TEMP_CHOSEN}" | bc -l)

# Save the current working directory for later
PT_dir=$(pwd)
PT_dir_name=$(basename "$PT_dir")
echo "Current time: $(date)"
echo "Current working directory: $PT_dir"
echo "Current working directory name: $PT_dir_name"


SETUP_dir=$PT_dir/master_setup_TI
LOCAL_SETUP_dir=$CONFIG_dir/setup_TI


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

SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN" | bc -l)  # Gaussian smearing sigma


########################################
NPAR_CHOSEN=8 # For single point calculations, set NPAR_CHOSEN to 8
########################################


#-------------------------
# Print the parameters
echo "------------------------"
echo "Simulation parameters:"
echo "TEMP_CHOSEN: $TEMP_CHOSEN"
echo "PSTRESS_CHOSEN_GPa: $PSTRESS_CHOSEN_GPa"
echo "NPAR_CHOSEN: $NPAR_CHOSEN (special case for single point calculations)"
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



# Check if the folder name contains both TEMP_CHOSEN and PSTRESS_CHOSEN_GPa
if [[ "$PT_dir_name" == *"${TEMP_CHOSEN}"* && "$PT_dir_name" == *"${PSTRESS_CHOSEN_GPa}"* ]]; then
    echo "Folder name contains TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
else
    echo "Folder name does NOT contain both TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
    exit 1
fi


#--------------------------------------------------------------
# Loop over every directory named exactly 'KP1'
#--------------------------------------------------------------
while IFS= read -r -d '' parent; do
    echo "#----------------------------------------"
    echo "#----------------------------------------"
    echo ""
    echo "Processing parent directory: ${parent}"
    echo ""
    echo "#----------------------------------------"
    echo "#----------------------------------------"

    # Iterate over immediate subdirectories beginning with 'KP1'
    for child in "${parent}"/KP1*; do
        if [ -d "${child}" ]; then
            # Enter the child directory
            cd "${child}" || exit
            child_abs=$(pwd)
            echo " → Entered ${child_abs}"

            # Copy and source analysis script
            cp "${HELP_SCRIPTS_vasp}/data_4_analysis.sh" "${child_abs}/"
            source data_4_analysis.sh

            # Check for average pressure file
            if [[ ! -f analysis/peavg.out ]]; then
                echo "ERROR: analysis/peavg.out not found in ${child}" >&2
                exit 1
            fi
            # Extract the third field (pressure) from peavg.out
            P_RUN=$(grep Pressure analysis/peavg.out | awk '{print $3}')

            # Run ASE setup (assumed alias/function)
            l_ase
            # Generate high-precision calculations using eos script
            python ${HELP_SCRIPTS_vasp}/eos* \
                -p ${P_RUN} -m 0 -e 0.1 -hp 1 -nt 20

            #----------------------------------------------------------
            # Now enter hp_calculations for KPOINTS_222 runs
            #----------------------------------------------------------
            cd hp_calculations
            hp_calculations_dir=$(pwd)
            echo " → In hp_calculations at ${hp_calculations_dir}"

            # Copy VASP run scripts and input templates
            cp ${PT_dir}/master_setup_TI/RUN_VASP_relax.sh \
                ${hp_calculations_dir}/RUN_VASP.sh
            cp ${PT_dir}/master_setup_TI/POTCAR \
                ${hp_calculations_dir}/POTCAR
            cp ${PT_dir}/master_setup_TI/KPOINTS_222 \
                ${hp_calculations_dir}/KPOINTS
            cp ${PT_dir}/master_setup_TI/INCAR_relax \
                ${hp_calculations_dir}/INCAR

            # Substitute placeholders in INCAR
            sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_222}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" ${hp_calculations_dir}/INCAR
            sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" ${hp_calculations_dir}/INCAR

            # Distribute input files to all subdirectories
            find . -type d | xargs -I {} cp INCAR KPOINTS POTCAR RUN_VASP.sh {}

            # Clean up templates in current and analysis directories
            rm -f KPOINTS INCAR POTCAR RUN_VASP.sh
            cd analysis
            rm -r KPOINTS INCAR POTCAR RUN_VASP.sh
            cd ..

            #----------------------------------------------------------
            # Submit VASP jobs in each subfolder
            #----------------------------------------------------------
            for subchild in *; do
                if [ -d "${subchild}" ]; then
                    cd "${subchild}" || exit
                    sbatch RUN_VASP.sh
                    echo "   → Started VASP in ${subchild}"
                    cd ..
                fi
            done

            echo "Started hp_calculations in ${child}"
            # Return to the original driver directory
            cd ${PT_dir} || exit
        fi
    done
done < <(find . -type d -name KP1 -print0)

# Final confirmation message
echo ""
echo ""
echo ""
echo "################################"
echo "################################"
echo "################################"
echo "All VASP jobs started in all KP1/*/hp_calculations directories under ${PT_dir}."
echo "################################"
echo "################################"
echo "################################"
echo ""
echo ""
echo ""
