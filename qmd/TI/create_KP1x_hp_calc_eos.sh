#!/bin/bash
# set -euo pipefail

# Usage: source $HELP_SCRIPTS_TI/create_KP1x_hp_calc_eos.sh > log.create_KP1x_hp_calc_eos 2>&1 &
#        nohup "$HELP_SCRIPTS_TI/create_KP1x_hp_calc_eos.sh" > log.create_KP1x_hp_calc_eos 2>&1 &
# Author: Akash Gupta

# while IFS= read -r -d '' parent; do
#     # …look for its immediate subdirectories starting with “KP1”
#     for child in "$parent"/KP1*; do
#         if [ -d "$child" ]; then
#         touch "$child/done_KP1x"
#         # P_RUN=$(grep Pressure analysis/peavg.out  | awk '{print $3}')
#         # module load anaconda3/2024.6; conda activate ase_env; python $HELP_SCRIPTS_vasp/eos* -p $P_RUN -m 0 -e 0.01 -hp 1
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



# echo process ID
echo "Process ID: $$"
echo ""


# Scaling parameter for alchemical transformations
SCALEE_CHOSEN=1.0            # Lambda scaling factor
MLDP_SCRIPTS="/projects/BURROWS/akashgpt/misc_libraries/scripts_Jie/mldp"

# Physical constants
kB=0.00008617333262145       # Boltzmann constant (eV/K)
kBar_to_GPa=0.1   
# Compute sigma = k_B * T (for smearing)
# SIGMA_CHOSEN=$(echo "${kB} * ${TEMP_CHOSEN}" | bc -l)

# Save the current working directory for later
PT_dir=$(pwd)
PT_dir_name=$(basename "$PT_dir")

COMPOSITION_dir=$(dirname "$PT_dir")
COMPOSITION_dir_name=$(basename "$COMPOSITION_dir")

echo "Current time: $(date)"
echo "Current PT directory: $PT_dir"
echo "Current PT directory name: $PT_dir_name"
echo "Current COMPOSITION directory: $COMPOSITION_dir"
echo "Current COMPOSITION directory name: $COMPOSITION_dir_name"
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

SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN" | bc -l)  # Gaussian smearing sigma


########################################
# NPAR_CHOSEN=8 # For single point calculations, set NPAR_CHOSEN to 8
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
echo "MLDP_SCRIPTS: $MLDP_SCRIPTS"
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

    cd "${parent}" || exit
    KP1_dir=$(pwd)
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
    cd "${PT_dir}" || exit


    counter_incomplete_runs=0

    # Iterate over immediate subdirectories beginning with 'KP1'
    for child in "${parent}"/KP1*; do
        if [ -d "${child}" ]; then
            # Enter the child directory
            cd "${child}" || exit
            child_abs=$(pwd)
            KP1x_dir=$child_abs
            echo " → Entered ${child_abs}"

            # # Copy and source analysis script
            cp "${HELP_SCRIPTS_vasp}/data_4_analysis.sh" "${child_abs}/"
            source data_4_analysis.sh

            # # Check for average pressure file
            # if [[ ! -f analysis/peavg.out ]]; then
            #     echo "ERROR: analysis/peavg.out not found in ${child}" >&2
            #     exit 1
            # fi
            # # Extract the third field (pressure) from peavg.out
            # P_RUN=$(grep Pressure analysis/peavg.out | awk '{print $3}')

            # # Run ASE setup (assumed alias/function)
            # module load anaconda3/2024.6; conda activate ase_env
            # # Generate high-precision calculations using eos script
            # python ${HELP_SCRIPTS_vasp}/eos* \
            #     -p ${P_RUN} -m 0 -e 0.1 -hp 1 -nt 20

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

            # backup to old.log.recal_test with a timestamp
            # touch old.log.recal_test
            # echo "##################################################################" >> old.log.recal_test
            # echo "### ~ 1 ~ ### $(date) ###" >> old.log.recal_test
            # echo "##################################################################" >> old.log.recal_test
            # cat log.recal_test >> old.log.recal_test


            # if [ "$num_failed_recal_frames" -gt 0 ]; then

            #     echo "Problem with recal phase ($((${line_count}-14))) at $(date). Or so it seems but sleeping to see if it gets resolved."
                
            #     #10 times check if line_count is still > 14 or num_failed_recal_frames > 0 and sleep for 600 seconds each time if not
            #     for i in {1..6}; do
            #         echo "Seeing if it completes ~ $i ..."
            #         sleep ${WAIT_TIME_LONG}
            #         python ${MLDP_SCRIPTS}/post_recal_rerun.py -ip all -v -ss $INPUT_FILES_DIR/sub_vasp_xtra.sh > log.recal_test 2>&1
            #         line_count=$(wc -l < "$logfile")    # Count the number of lines in the file
            #         num_failed_recal_frames=$((${line_count}-14))
            #         if [ "$num_failed_recal_frames" -le 0 ]; then
            #             break
            #         fi
            #     done

            #     # append to old.log.recal_test with a timestamp
            #     echo "##################################################################" >> old.log.recal_test
            #     echo "### ~ 2 ~ ### $(date) ###" >> old.log.recal_test
            #     echo "##################################################################" >> old.log.recal_test
            #     cat log.recal_test >> old.log.recal_test

            #     if [ "$num_failed_recal_frames" -gt 0 ]; then
            #         echo ""
            #         echo "Problem with recal phase persists ($((${line_count}-14))) -- check then rerun."
            #         echo "Exiting script."
            #         echo ""
            #         exit 1
            #     else
            #         echo ""
            #         echo "Recal phase completed successfully after waiting."
            #         echo ""
            #         # exit 0
            #     fi
            # fi




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
            cd ${PT_dir} || exit
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

    touch $KP1_dir/../done_estimating_V

    echo ""
    echo ""
    cd "${PT_dir}" || exit


done < <(find . -type d -name KP1 -print0)

# Final confirmation message
echo ""
echo ""
echo ""
echo "################################"
echo "################################"
echo "################################"
echo "All EoS data "corrected" in all KP1/*/hp_calculations directories under ${PT_dir} @ $(date)."
echo ""
echo "Total number of incomplete runs: $counter_incomplete_runs"
echo "################################"
echo "################################"
echo "################################"
echo ""
echo ""
echo ""
