#!/usr/bin/env bash
shopt -s nullglob   # so that `for dir in */` does nothing if there are no folders

# Usage: source $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos_SCALEE.sh 0 > log.isobar__create_KP1x_hp_calc_eos_SCALEE 2>&1 &
#        nohup bash $HELP_SCRIPTS_TI/isobar__create_KP1x_hp_calc_eos_SCALEE.sh 0 > log.isobar__create_KP1x_hp_calc_eos_SCALEE 2>&1 &


run_switch=${1:-0} # set to 0 to not run the calculations, just create the directories and files




# Constants
kB=0.00008617333262145  # Boltzmann constant in eV/K
SCALEE_CHOSEN=1.0 # Default value for SCALEE_CHOSEN, can be overridden by the input file


# for each folder in the current directory, except, master_setup_TI, go inside and run source $HELP_SCRIPTS_TI/calculate_GFE_v2.sh 1 2 > log.calculate_GFE 2>&1 &

# Save the current working directory for later
ISOBAR_CALC_dir=$(pwd)
CONFIG_dir=$(dirname "$ISOBAR_CALC_dir") # CONFIG_dir is the parent directory of isobar_calc
PT_dir=$(dirname "$CONFIG_dir") # PT_dir is the parent directory of CONFIG_dir
PT_dir_name=$(basename "$PT_dir")

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

PSTRESS_CHOSEN=$(echo "$PSTRESS_CHOSEN_GPa * 10" | bc -l)  # Convert GPa to kBar


#-------------------------
# Print the parameters
echo "------------------------"
echo "Simulation parameters:"
echo "TEMP_CHOSEN: $TEMP_CHOSEN"
echo "PSTRESS_CHOSEN_GPa: $PSTRESS_CHOSEN_GPa"
echo "PSTRESS_CHOSEN: $PSTRESS_CHOSEN (kBar)"
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
echo "MLDP_SCRIPTS: $MLDP_SCRIPTS"
echo "-------------------------"
echo ""



# Check if the folder name contains both TEMP_CHOSEN and PSTRESS_CHOSEN_GPa
if [[ "$PT_dir_name" == *"${TEMP_CHOSEN}"* && "$PT_dir_name" == *"${PSTRESS_CHOSEN_GPa}"* ]]; then
    echo "Folder name contains TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
else
    echo "Folder name does NOT contain both TEMP_CHOSEN (${TEMP_CHOSEN}) and PSTRESS_CHOSEN_GPa (${PSTRESS_CHOSEN_GPa})"
    exit 1
fi



echo ""
echo ""
echo ""




cd $ISOBAR_CALC_dir || exit 1

for dir in */; do
    # skip the master_setup_TI folder
    [[ "$dir" == "master_setup_TI/" ]] && continue

    cd "$dir" || exit 1

    ISOBAR_T_dir=$(pwd)
    ISOBAR_T_dir_name=$(basename "$ISOBAR_T_dir")
    V_est_dir=$ISOBAR_T_dir/V_est
    KP1_dir=$V_est_dir/KP1

    LOCAL_SETUP_dir=$ISOBAR_T_dir/setup_TI

    # # check if $ISOBAR_T_dir/done_SCALEE_0 exists -- if yes, skip this directory
    if [ -f "$ISOBAR_T_dir/done_SCALEE_0" ]; then
        echo ""
        echo "============================"
        echo "Skipping $ISOBAR_T_dir as done_SCALEE_0 exists."
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
    echo 
    echo "==========================="
    echo "TEMP_CHOSEN_ISOBAR: $TEMP_CHOSEN_ISOBAR"
    echo "SIGMA_CHOSEN_ISOBAR: $SIGMA_CHOSEN_ISOBAR"
    echo "SCALEE_CHOSEN: $SCALEE_CHOSEN"
    echo "==========================="
    echo ""




    # read the cell sizes from cell_sizes_KPX.dat
    cell_size_KP1=$(awk 'NR==2' $V_est_dir/cell_sizes_KPX.dat)
    cell_size_KP2=$(awk 'NR==3' $V_est_dir/cell_sizes_KPX.dat)

    echo "cell_size_KP1: ${cell_size_KP1}"
    echo "cell_size_KP2: ${cell_size_KP2}"
    echo ""




    KP2_dir=$KP1_dir/KP1b # when mode == 2
    echo "KP2_dir (effective; the KP1/KP1b directory): ${KP2_dir}"



    # Create SCALEE_0
    cd $ISOBAR_T_dir || exit 1
    SCALEE_0_dir=$ISOBAR_T_dir/SCALEE_0

    # check if there exists SCALEE_0/done_SCALEE_0 file
    if [ -f "$SCALEE_0_dir/done_SCALEE_0" ]; then
        echo "SCALEE_0 already exists in $SCALEE_0_dir. Skipping creation."
        continue
    else
        echo "done_SCALEE_0 file does not exist in $SCALEE_0_dir. Creating it."
    fi

    echo "Creating SCALEE_0 (i.e., NVT simulation for isobaric estimation of GFE) in $ISOBAR_T_dir"
    mkdir -p SCALEE_0
    echo "SCALEE_0_dir: $SCALEE_0_dir"
    cp $SETUP_dir/KPOINTS_111 "$SCALEE_0_dir/KPOINTS" # KPOINTS 111 for SCALEE
    cp $SETUP_dir/INCAR_SCALEE "$SCALEE_0_dir/INCAR" || echo "Error: INCAR_SCALEE not found in $SETUP_dir"
    cp $SETUP_dir/RUN_VASP_SCALEE.sh "$SCALEE_0_dir/RUN_VASP.sh"
    cp $HELP_SCRIPTS/qmd/vasp/data_4_analysis.sh "$SCALEE_0_dir/data_4_analysis.sh"


    CONFIG_dir_name=$(basename "$CONFIG_dir")
    # if "_0" in the CONFIG_dir_name and COMPOSITION_dir_name contains "MgSiO3", copy corresponding POTCARs
    if [[ "$CONFIG_dir_name" == *_0* && "$COMPOSITION_dir_name" == *"MgSiO3"* ]]; then
        echo "Copying POTCAR for MgSiO3"
        cp $SETUP_dir/POTCAR_MgSiO3 "$SCALEE_0_dir/POTCAR"
    elif [[ "$CONFIG_dir_name" == *_0* && "$COMPOSITION_dir_name" == *"Fe"* ]]; then
        echo "Copying POTCAR for Fe"
        cp $SETUP_dir/POTCAR_Fe "$SCALEE_0_dir/POTCAR"
    else
        echo "Copying default POTCAR -- H+Fe | H+MgSiO3"
        cp $SETUP_dir/POTCAR "$SCALEE_0_dir/POTCAR"
    fi


    # prep POSCAR for SCALEE
    cp $KP2_dir/CONTCAR $SCALEE_0_dir/POSCAR
    sed -i "2s/.*/$cell_size_KP2/" $SCALEE_0_dir/POSCAR
    sed -i "3s/.*/1.0000000000000000 0.0000000000000000 0.0000000000000000/" $SCALEE_0_dir/POSCAR
    sed -i "4s/.*/0.0000000000000000 1.0000000000000000 0.0000000000000000/" $SCALEE_0_dir/POSCAR
    sed -i "5s/.*/0.0000000000000000 0.0000000000000000 1.0000000000000000/" $SCALEE_0_dir/POSCAR




    echo ""



    cd "$SCALEE_0_dir" || exit 1
    rm -f slurm*
    cd $ISOBAR_T_dir || exit 1

    # Replace __SCALEE_CHOSEN__ with the SCALEE_CHOSEN
    sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" "$SCALEE_0_dir/INCAR"

    # Replace __TEMP_CHOSEN__ and __SIGMA_CHOSEN__ based on the chosen temperature
    sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN_ISOBAR}/" "$SCALEE_0_dir/INCAR"
    sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN_ISOBAR}/" "$SCALEE_0_dir/INCAR"
    sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" "$SCALEE_0_dir/INCAR"

    # Replace __NPAR_CHOSEN__ with the chosen NPAR
    sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" "$SCALEE_0_dir/INCAR"
    sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_111}/" "$SCALEE_0_dir/INCAR"

    # Replace __NBANDS_CHOSEN__ with the chosen NBANDS
    sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" "$SCALEE_0_dir/INCAR"

    if [ $run_switch -eq 1 ]; then
        cd "$SCALEE_0_dir" || exit 1
        rm -rf $SCALEE_0_dir/to_RUN  # remove any existing to_RUN file
        sbatch RUN_VASP.sh
        echo "SCALEE_0 submitted to SLURM in $SCALEE_0_dir"
        # list just the job IDs for your user, sort them numerically, and take the last one
        LAST_JOB_ID=$(squeue -u $USER -h -o "%i" | sort -n | tail -1)
        echo "JOB_ID_SCALEE_0: $LAST_JOB_ID"
        echo ""
        cd $ISOBAR_T_dir || exit 1
    else
        echo "run_switch is set to 0, not running SCALEE_0."
        touch "$SCALEE_0_dir/to_RUN"
        echo "Created <to_RUN> file as a marker for a later run."
    fi

    cd $ISOBAR_CALC_dir || exit 1

    sleep ${WAIT_TIME_SHORT}
done

echo ""
echo ""
echo ""
echo "All calculations started/created in the background by $(date)."
echo ""