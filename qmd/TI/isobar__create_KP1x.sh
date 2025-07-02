#!/bin/bash
# set -euo pipefail

# Usage: source $HELP_SCRIPTS_TI/isobar__create_KP1x.sh > log.isobar__create_KP1x 2>&1 &
#        nohup $HELP_SCRIPTS_TI/isobar__create_KP1x.sh > log.isobar__create_KP1x 2>&1 &
# Author: Akash Gupta




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
    echo "#----------------------------------------"
    echo "#----------------------------------------"
    echo ""
    echo "Processing parent directory: ${parent}"
    echo ""
    echo "#----------------------------------------"
    echo "#----------------------------------------"
    # Enter parent directory
    cd "$parent" || exit

    KP1_dir=$(pwd)
    V_est_dir=$(dirname "$KP1_dir") # parent directory is V_est_dir
    ISOBAR_T_dir=$(dirname "$V_est_dir") # parent directory is ISOBAR_T_dir
    ISOBAR_T_dir_name=$(basename "$ISOBAR_T_dir") # name of the ISOBAR_T_dir directory

    # check if $V_est_dir/done_estimating_V exists -- if yes, skip this directory
    if [ -f "$V_est_dir/done_estimating_V" ]; then
        echo ""
        echo "============================"
        echo "Skipping $V_est_dir as done_estimating_V exists in V_est."
        echo "============================"
        echo ""
        echo ""
        cd "$ISOBAR_CALC_dir" || exit 1  # Return to ISOBAR_CALC_dir
        continue
    fi


    # Extract TEMP_CHOSEN_ISOBAR from the name of the ISOBAR_T_dir directory (ISOBAR_T_dir_name) -- the format is "T<TEMP_CHOSEN_i>"
    TEMP_CHOSEN_ISOBAR=$(echo "$ISOBAR_T_dir_name" | sed 's/T//g')
    SIGMA_CHOSEN_ISOBAR=$(echo "$kB * $TEMP_CHOSEN_ISOBAR" | bc -l)  # Gaussian smearing sigma
    echo "KP1_dir: $KP1_dir"
    echo "V_est_dir: $V_est_dir"
    echo "ISOBAR_T_dir: $ISOBAR_T_dir"
    echo "ISOBAR_T_dir_name: $ISOBAR_T_dir_name"
    echo "==========================="
    echo "TEMP_CHOSEN_ISOBAR: $TEMP_CHOSEN_ISOBAR"
    echo "SIGMA_CHOSEN_ISOBAR: $SIGMA_CHOSEN_ISOBAR"
    echo "==========================="
    echo ""



    ISOBAR_CALC_dir__test=$(dirname "$ISOBAR_T_dir") # parent directory is ISOBAR_CALC_test_dir
    # if ISOBAR_CALC_dir__test is not the same as ISOBAR_CALC_dir, exit
    if [ "$ISOBAR_CALC_dir__test" != "$ISOBAR_CALC_dir" ]; then
        echo "ERROR: ISOBAR_CALC_dir__test ($ISOBAR_CALC_dir__test) is not the same as ISOBAR_CALC_dir ($ISOBAR_CALC_dir)"
        exit 1
    else
        echo "ISOBAR_CALC_dir__test ($ISOBAR_CALC_dir__test) is the same as ISOBAR_CALC_dir ($ISOBAR_CALC_dir)"
        echo ""
    fi



    # # echo "###########################"
    # # echo 
    # # echo "###########################"
    # # echo "UNCOMMENT THE FOLLOWING LINES IN THE CODE!!!"
    # # echo "###########################"
    # # echo ""
    # # echo "###########################"
    # # 0) file log.isobar__create_KP1 has numbers next to "JOB_ID_KP1:" -- grab all those numbers and check if they are running
    # #    if they are running, sleep and wait for them to finish
    # #    if they are not running, continue to the next step
    # #    if there are no numbers, prompt error and exit
    # # 0.1) Extract all tokens after "JOB_ID_KP1:" 
    # mapfile -t job_ids < <(grep -oP 'JOB_ID_KP1:\s*\K\S+' $ISOBAR_CALC_dir/log.isobar__create_KP1)

    # # # 0.2) Error if none found
    # if [ ${#job_ids[@]} -eq 0 ]; then
    #     echo "ERROR: no JOB_ID_KP1 entries found in log.isobar__create_KP1" >&2
    #     exit 1
    # fi

    # # # 0.3) Validate that each is a pure integer
    # for jid in "${job_ids[@]}"; do
    #     if ! [[ $jid =~ ^[0-9]+$ ]]; then
    #         echo "ERROR: invalid job ID '$jid' extracted from log.isobar__create_KP1" >&2
    #         exit 1
    #     fi
    # done

    # echo "Found JOB_ID_KP1 IDs: ${job_ids[*]}"

    # # # 0.4) For each valid job ID, wait until SLURM no longer lists it
    # for jid in "${job_ids[@]}"; do
    #     echo -n "Waiting for SLURM job $jid to finish"
    #     while squeue -h -j "$jid" &>/dev/null; do
    #         echo -n "."      # progress dot
    #         sleep "$WAIT_TIME_LONG"
    #     done
    #     echo " done."
    # done

    # echo "All JOB_ID_KP1 jobs have completed; proceeding with the next steps."
    # echo
    # echo
    # #################
    # # UNCOMMENT THE ABOVE LINES!!!



    # # backup check
    # # 1) extract JOB_ID_KP1 from the first line of log.run_sim
    # JOB_ID_KP1=$(awk 'NR==1 {print $3}' log.run_sim)

    # if [[ -z "$JOB_ID_KP1" ]]; then
    #     echo "ERROR: could not read JOB_ID_KP1 from log.run_sim"
    #     exit 1
    # fi

    # echo "JOB_ID_KP1: $JOB_ID_KP1"
    # echo -n "Waiting for job $JOB_ID_KP1 to finish "

    # # 2) loop until squeue no longer reports it
    # while squeue -h -j "$JOB_ID_KP1" >/dev/null; do
    #     # print a dot and sleep
    #     echo -n "."
    #     sleep "$WAIT_TIME_LONG"
    # done

    # echo    # newline after the dots
    # echo "Job $JOB_ID_KP1 has completed."
    # echo


    # perform analysis
    cp $HELP_SCRIPTS_vasp/data_4_analysis.sh .
    source data_4_analysis.sh

    # Generate KP1* subdirectories with EOS script
    module load anaconda3/2024.6; conda activate ase_env
    python $HELP_SCRIPTS_vasp/eos* \
        -p $PSTRESS_CHOSEN_GPa -m 0 -e 0.025 -hp -1  # create KP1a, KP1b, etc.

    # Return to root
    cd $ISOBAR_CALC_dir || exit

    # Loop over each KP1* subdirectory
    for child in "$parent"/KP1*; do
        if [ -d "$child" ]; then
            cd "$child" || exit
            KP1x_dir=$(pwd)  # KP1x_dir is the current directory

            # Copy VASP run scripts and input templates
            cp ${SETUP_dir}/RUN_VASP_NVT.sh $KP1x_dir/RUN_VASP.sh

            # cp $KP1_dir/POTCAR           $KP1x_dir/POTCAR
            CONFIG_dir_name=$(basename "$CONFIG_dir")
            # if "_0" in the CONFIG_dir_name and COMPOSITION_dir_name contains "MgSiO3", copy corresponding POTCARs
            if [[ "$CONFIG_dir_name" == *_0* && "$COMPOSITION_dir_name" == *"MgSiO3"* ]]; then
                echo "Copying POTCAR for MgSiO3"
                cp $SETUP_dir/POTCAR_MgSiO3 "$KP1x_dir/POTCAR"
            elif [[ "$CONFIG_dir_name" == *_0* && "$COMPOSITION_dir_name" == *"Fe"* ]]; then
                echo "Copying POTCAR for Fe"
                cp $SETUP_dir/POTCAR_Fe "$KP1x_dir/POTCAR"
            else
                echo "Copying default POTCAR"
                cp $SETUP_dir/POTCAR "$KP1x_dir/POTCAR"
            fi

            cp $SETUP_dir/KPOINTS_111         $KP1x_dir/KPOINTS   # KPOINTS 1×1×1
            cp ${SETUP_dir}/INCAR_SCALEE $KP1x_dir/INCAR

            # Populate INCAR placeholders
            sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN_ISOBAR}/" $KP1x_dir/INCAR
            sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" $KP1x_dir/INCAR
            sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" $KP1x_dir/INCAR
            sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" $KP1x_dir/INCAR
            sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_111}/" $KP1x_dir/INCAR
            sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN_ISOBAR}/" $KP1x_dir/INCAR
            sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" $KP1x_dir/INCAR

            # Submit the VASP job
            sbatch RUN_VASP.sh
            LAST_JOB_ID=$(squeue -u $USER -h -o "%i" | sort -n | tail -1)
            echo "   → Started VASP in ${child}"
            echo "JOB_ID_KP1x: $LAST_JOB_ID"
            echo ""

            # Return to root directory for next iteration
            cd $ISOBAR_CALC_dir || exit
        fi
    done
    echo "++++++++++++++++++++++++++++++"
    echo 
    echo

done < <(find . -type d -name KP1 -print0)

# Final status message
echo "All ISOBAR VASP jobs started in KP1 directories in ${ISOBAR_CALC_dir}."
