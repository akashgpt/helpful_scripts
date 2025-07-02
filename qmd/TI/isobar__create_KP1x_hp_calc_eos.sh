#!/bin/bash
# set -euo pipefail

# Usage: source $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos.sh > log.isobar__create_KP1x_hp_calc_eos 2>&1 &
#        nohup $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos.sh > log.isobar__create_KP1x_hp_calc_eos 2>&1 &
# Author: Akash Gupta



# Scaling parameter for alchemical transformations
SCALEE_CHOSEN=1.0            # Lambda scaling factor
MLDP_SCRIPTS="/projects/BURROWS/akashgpt/misc_libraries/scripts_Jie/mldp"

# Physical constants
kB=0.00008617333262145       # Boltzmann constant (eV/K)
kBar_to_GPa=0.1   
# Compute sigma = k_B * T (for smearing)
# SIGMA_CHOSEN=$(echo "${kB} * ${TEMP_CHOSEN}" | bc -l)

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
echo "CONFIG directory: $CONFIG_dir"
echo "ISOBAR_CALC_dir: $ISOBAR_CALC_dir"
echo ""


SETUP_dir=$PT_dir/master_setup_TI
# # LOCAL_SETUP_dir=$CONFIG_dir/setup_TI


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

########################################
# NPAR_CHOSEN=8 # For single point calculations, set NPAR_CHOSEN to 8
########################################


#-------------------------
# Print the parameters
echo "------------------------"
echo "Simulation parameters:"
echo "TEMP_CHOSEN_ARRAY: ${TEMP_CHOSEN_ARRAY[@]}"
echo "PSTRESS_CHOSEN_GPa: $PSTRESS_CHOSEN_GPa"
echo "PSTRESS_CHOSEN: $PSTRESS_CHOSEN (in kBar)"
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
echo "MLDP_SCRIPTS: $MLDP_SCRIPTS"
echo "-------------------------"



# Check if the folder name contains both TEMP_CHOSEN and PSTRESS_CHOSEN_GPa
if [[ "$PT_dir_name" == *"${TEMP_CHOSEN}"* && "$PT_dir_name" == *"${PSTRESS_CHOSEN_GPa}"* ]]; then
    echo "Folder name contains TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
else
    echo "Folder name does NOT contain both TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
    exit 1
fi

counter_incomplete_runs=0

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

    cd "${parent}" || exit
    KP1_dir=$(pwd)
    V_est_dir=$(dirname "$KP1_dir") # parent directory is V_est_dir
    ISOBAR_T_dir=$(dirname "$V_est_dir") # parent directory is ISOBAR_T_dir
    ISOBAR_T_dir_name=$(basename "$ISOBAR_T_dir") # name of the ISOBAR_T_dir directory
    ISOBAR_CALC_dir__test=$(dirname "$ISOBAR_T_dir") # parent directory is ISOBAR_CALC_test_dir
    
    # if ISOBAR_CALC_dir__test is not the same as ISOBAR_CALC_dir, exit
    if [ "$ISOBAR_CALC_dir__test" != "$ISOBAR_CALC_dir" ]; then
        echo "ERROR: ISOBAR_CALC_dir__test ($ISOBAR_CALC_dir__test) is not the same as ISOBAR_CALC_dir ($ISOBAR_CALC_dir)"
        exit 1
    else
        echo "ISOBAR_CALC_dir__test ($ISOBAR_CALC_dir__test) is the same as ISOBAR_CALC_dir ($ISOBAR_CALC_dir)"
    fi

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


    analysis_KP1_dir=$KP1_dir/analysis
    corrected_pressure_file="${analysis_KP1_dir}/corrected_pressure.dat"
    pressure_correction_file="${analysis_KP1_dir}/pressure_correction.dat"
    corrected_volume_file="${analysis_KP1_dir}/corrected_volume.dat"
    # create corrected_pressure.dat and corrected_volume.dat files and add the header: "# after KP1, KP1X, hp_calculations"
    mkdir -p "${analysis_KP1_dir}"
    rm -f "${corrected_pressure_file}" "${corrected_volume_file}" "${pressure_correction_file}"
    echo "# after KP1, KP1X, hp_calculations (in kBar)" > "${corrected_pressure_file}"
    echo "# after KP1, KP1X, hp_calculations (in kBar)" > "${corrected_volume_file}"
    echo "# after KP1, KP1X, hp_calculations (in kBar)" > "${pressure_correction_file}"
    cd "${ISOBAR_CALC_dir}" || exit


    # counter_incomplete_runs=0

    # Iterate over immediate subdirectories beginning with 'KP1'
    for child in "${parent}"/KP1*; do
        if [ -d "${child}" ]; then
            # Enter the child directory
            cd "${child}" || exit
            KP1x_dir=$(pwd)
            echo " → Entered ${KP1x_dir}"

            # # Copy and source analysis script
            cp "${HELP_SCRIPTS_vasp}/data_4_analysis.sh" "${KP1x_dir}/"
            source data_4_analysis.sh


            #----------------------------------------------------------
            # Now enter hp_calculations for KPOINTS_222 runs
            #----------------------------------------------------------
            cd hp_calculations
            hp_calculations_dir=$(pwd)
            echo " → In hp_calculations"


            #########################################################
            #########################################################
            python ${MLDP_SCRIPTS}/post_recal_rerun.py -ip all -v -ss $INPUT_FILES_DIR/sub_vasp_xtra.sh > log.recal_test 2>&1

            logfile="log.recal_test"
            line_count=$(wc -l < "$logfile")    # Count the number of lines in the file
            num_failed_recal_frames=$((${line_count}-14)) # name says it all ...
            echo "Number of failed recal frames: $num_failed_recal_frames"
            counter_incomplete_runs=$(($counter_incomplete_runs + $num_failed_recal_frames))

            echo ""
            if [ $num_failed_recal_frames -eq 0 ]; then
                echo "All recal frames completed successfully in ${KP1x_dir}/hp_calculations (num_failed_recal_frames: $num_failed_recal_frames)."
            else
                echo "WARNING: Number of failed recal frames in ${KP1x_dir}/hp_calculations: $num_failed_recal_frames"
            fi
            echo ""

            ##########################################################
            ##########################################################


            # grep pressure (external pressure == total pressure in relaxation) from OUTCAR and save it in a file called pressure.txt
            mkdir -p $hp_calculations_dir/analysis
            rm -f $hp_calculations_dir/analysis/pressure.dat
            rm -f $hp_calculations_dir/analysis/external_pressure_KP2.dat
            rm -f $hp_calculations_dir/analysis/volume.dat
            echo "# external pressure (GPa)" > $hp_calculations_dir/analysis/external_pressure_KP2.dat

            # for all directories in hp_calculations
            for subchild in $(ls -1v); do
                if [ -d "${subchild}" ]; then

                    # if subchild is "analysis", skip it
                    if [[ "${subchild}" == "analysis" ]]; then
                        continue
                    fi

                    cd "${subchild}" || exit
                    echo "   → Entered ${subchild}"

                    # grep external pressure from OUTCAR, multiply it by 0.1 and save it in a file called external_pressure_KP2.dat
                    grep "external pressure" OUTCAR \
                        | awk "{ print \$4 * $kBar_to_GPa }"  >> "$hp_calculations_dir/analysis/external_pressure_KP2.dat"
                    grep "pressure" OUTCAR | awk '{print $4}' >> $hp_calculations_dir/analysis/pressure.dat
                    grep -m 1 volume OUTCAR | awk '{print $5}' >> $hp_calculations_dir/analysis/volume.dat
                    cd ..
                fi
            done

            # external_pressure_KP2.dat, multiply by 0.1


            # # if first line of external_pressure_KP1.dat contains "#", remove it
            # if [[ $(head -n 1 analysis/external_pressure_KP1.dat) == \#* ]]; then
            #     sed -i '1d' analysis/external_pressure_KP1.dat
            # fi

            # subtract the lines in external_pressure_KP1.dat from the lines in external_pressure_KP2.dat
            # and save the result in a file called diff_external_pressure.dat. Skip the first line of each file.
            # paste -d ' ' analysis/external_pressure_KP1.dat analysis/external_pressure_KP2.dat | awk '{print $1-$2}' > analysis/diff_external_pressure.dat
            paste \
                <(tail -n +2 $hp_calculations_dir/analysis/external_pressure_KP1.dat) \
                <(tail -n +2 $hp_calculations_dir/analysis/external_pressure_KP2.dat) \
                | awk '{ print $2 - $1 }' \
                > $hp_calculations_dir/analysis/diff_external_pressure.dat

            # average values in diff_external_pressure.dat and save it to avg_diff_external_pressure.dat
            awk '{ sum += $1 } END { if (NR > 0) print sum / NR }' $hp_calculations_dir/analysis/diff_external_pressure.dat > $hp_calculations_dir/analysis/avg_diff_external_pressure.dat

            total_pressure_KP1=$(awk 'NR==15 {print $3}' ${KP1x_dir}/analysis/peavg.out)
            cell_volume=$(awk 'NR==21 {print $5}' ${KP1x_dir}/analysis/peavg.out)
            # corrected_pressure = total_pressure_KP1 + avg_diff_external_pressure
            avg_diff_external_pressure=$(awk 'NR==1 {print $1}' $hp_calculations_dir/analysis/avg_diff_external_pressure.dat)
            avg_diff_external_pressure_kBar=$(echo "$avg_diff_external_pressure / $kBar_to_GPa" | bc -l)
            corrected_pressure=$(echo "$total_pressure_KP1 + $avg_diff_external_pressure" | bc -l)
            corrected_pressure_kBar=$(echo "$corrected_pressure / $kBar_to_GPa " | bc -l)
            echo $corrected_pressure_kBar >> $corrected_pressure_file # pressure required in kBar there
            echo $cell_volume >> $corrected_volume_file
            echo $avg_diff_external_pressure_kBar >> $pressure_correction_file
            # echo "Started hp_calculations in ${child}"
            # Return to the original driver directory
            cd ${ISOBAR_CALC_dir} || exit
        fi
    done

    cd "${KP1_dir}" || exit
    echo ""
    echo ""
    echo "----------------------------------------"
    echo "Estimating final EoS parameters for ${KP1_dir}"
    echo "----------------------------------------"
    python $HELP_SCRIPTS_vasp/eos_fit__V_at_P.py -p $PSTRESS_CHOSEN_GPa -m 2 -e 0.2 -hp 0

    CELL_FILE="$KP1_dir/../cell_sizes_KPX.dat"

    # 1) create the file if it doesn't exist
    if [ ! -f "$CELL_FILE" ]; then
        cat > "$CELL_FILE" <<EOF
    cell_sizes_KP1; cell_sizes_KP2 (or the hp_calculations equivalent)
    <cell_size_KP1>
    <cell_size_KP2>
EOF
        echo "cell_sizes_KPX.dat not found. Creating a new one at $CELL_FILE."
    fi

    # ensure at least 3 lines in the file
    line_count=$(wc -l < "$CELL_FILE")
    if [ "$line_count" -lt 3 ]; then
    # append blank lines until there are 3
    for n in $(seq $((line_count + 1)) 3); do
        echo "" >> "$CELL_FILE"
    done
    fi

    # make sure the log files exist before you try to awk them
    LOG2="$KP1_dir/analysis/log.eos_fit_data_mode_2"
    LOG1="$KP1_dir/analysis/log.eos_fit"
    for f in "$LOG2" "$LOG1"; do
        [ -r "$f" ] || { echo "Error: cannot read $f"; exit 1; }
    done

    # 2) extract volumes
    estimated_cell_volume_KP2=$(awk 'NR==9 {print $7}' "$LOG2")
    estimated_cell_volume_KP1=$(awk 'NR==9 {print $7}' "$LOG1")

    # 3) patch line 3 and line 2
    sed -i "3s/.*/$estimated_cell_volume_KP2/" "$CELL_FILE"
    sed -i "2s/.*/$estimated_cell_volume_KP1/" "$CELL_FILE"

    echo "cell_sizes_KPX.dat updated with estimated cell volumes."
    echo "  estimated_cell_volume_KP1: $estimated_cell_volume_KP1"
    echo "  estimated_cell_volume_KP2: $estimated_cell_volume_KP2"
    echo "----------------------------------------"

    touch $V_est_dir/done_estimating_V

    echo ""
    echo ""
    cd "${ISOBAR_CALC_dir}" || exit


done < <(find . -type d -name KP1 -print0)

# Final confirmation message
echo ""
echo ""
echo ""
echo "################################"
echo "################################"
echo "################################"
echo "All EoS data "corrected" in all KP1/*/hp_calculations directories under ${ISOBAR_CALC_dir} @ $(date)."
echo ""
echo "Total number of incomplete runs: $counter_incomplete_runs"
echo "################################"
echo "################################"
echo "################################"
echo ""
echo ""
echo ""
