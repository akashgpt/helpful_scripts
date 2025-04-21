#!/bin/bash

# Summary:
# This script creates directories for different SCALEE values and copies the necessary files for VASP simulations.
# Before that, it estimates an accurate cell size based on KPOINTS 111 and KPOINTS 222 simulations.
# Usage: source setup_TI/calculate_GFE_v2.sh 1 1 > log.calculate_GFE 2>&1 &


# =======================================================================================
# =======================================================================================
# =======================================================================================
run_switch=${1:-1} # 0: create directories, 1: create directories and run VASP simulations
mode=${2:-1} # 0: run in normal mode (KP1+KP2+...), 1: run in high accuracy mode (KP1+hp_calculations)
# mode 0 is where you first figure out V_est (low accuracy; KP1) and then do a high accuracy KP2 given this better V_est and CONTCAR from KP1 sim
# mode 1 is where you do high accuracy calculations for a select number of frames from KP1 sim as in the DPAL recal calculations
# Input parameters
SCALEE=(1.0 0.71792289 0.3192687 0.08082001 0.00965853 0.00035461 0.00000108469)
MLDP_SCRIPTS="/projects/BURROWS/akashgpt/misc_libraries/scripts_Jie/mldp"
# SCALEE=(0.71792289)
# TEMP_CHOSEN=13000
# PSTRESS_CHOSEN_GPa=1000 # in GPa
# NBANDS_CHOSEN=784 # number of bands to be used in the calculation
# POTIM_CHOSEN=0.5
# NPAR_CHOSEN=14 # choose based on CLUSTER || the number of cores per node and the number of nodes; Preferred values: TIGER3: 14, STELLAR: 16
# KPAR_CHOSEN_111=1 # for KPOINTS 111
# KPAR_CHOSEN_222=4 # for KPOINTS 222
# WAIT_TIME_VLONG=600
# WAIT_TIME_LONG=60
# WAIT_TIME_SHORT=10
# =======================================================================================
# =======================================================================================
# =======================================================================================



#########
# KPOINTS_CHOSEN_111="1 1 1" # for KPOINTS 111
# KPOINTS_CHOSEN_222="2 2 2" # for KPOINTS 222

CONFIG_dir=$(pwd)
TP_dir=$(dirname "$CONFIG_dir")
SETUP_dir=$TP_dir/master_setup_TI
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
        esac
    done < "$PARAMETER_FILE"
else
    echo "Parameter file not found: $PARAMETER_FILE"
    exit 1
fi



# Constants
kB=0.00008617333262145  # Boltzmann constant in eV/K



# check if the following files exist in the setup directory: POTCAR, KPOINTS_111, KPOINTS_222, INCAR_NPT, RUN_VASP_NPT.sh, INCAR_SCALEE, RUN_VASP_SCALEE.sh, POSCAR_NPT, RUN_VASP_SCALEE_hp.sh
check_files=(
    "POTCAR"
    "KPOINTS_111"
    "KPOINTS_222"
    "INCAR_NPT"
    "RUN_VASP_NPT.sh"
    "INCAR_SCALEE"
    "RUN_VASP_SCALEE.sh"
    "RUN_VASP_SCALEE_hp.sh"
    "input.calculate_GFE"
)
for file in "${check_files[@]}"; do
    if [ ! -f "$SETUP_dir/$file" ]; then
        echo "Error: $file not found in $SETUP_dir"
        echo "NOTE: You need the following files in the setup directory -- POTCAR, KPOINTS_111, KPOINTS_222, INCAR_NPT, RUN_VASP_NPT.sh, INCAR_SCALEE, RUN_VASP_SCALEE.sh, POSCAR_NPT, RUN_VASP_SCALEE_hp.s, input.calculate_GFE"
        exit 1
    fi
done

# check for POSCAR_NPT in the LOCAL_SETUP_dir
if [ ! -f "$LOCAL_SETUP_dir/POSCAR_NPT" ]; then
    echo "Error: POSCAR_NPT not found in $LOCAL_SETUP_dir"
    exit 1
fi




# GPa to Kbar
PSTRESS_CHOSEN=$(echo "$PSTRESS_CHOSEN_GPa * 10" | bc -l)

# Calculate SIGMA_CHOSEN based on the chosen temperature
SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN" | bc -l)

CLUSTER_NAME=$(scontrol show config | grep ClusterName | awk '{print $3}')

# STEP 1: KPOINTS 111 sim
echo "=========================="
echo "PROCESS ID: $$"
if [ $run_switch -eq 1 ]; then
    echo "run_switch: ${run_switch} (running simulations)"
else
    echo "run_switch: ${run_switch} (only creating directories)"
fi
if [ $mode -eq 0 ]; then
    echo "mode: ${mode} (KP1+KP2)"
else
    echo "mode: ${mode} (KP1+hp_calculations)"
fi
echo "TEMP_CHOSEN: ${TEMP_CHOSEN}"
echo "PSTRESS_CHOSEN: ${PSTRESS_CHOSEN}"
echo "POTIM_CHOSEN: ${POTIM_CHOSEN}"
echo "NPAR_CHOSEN: ${NPAR_CHOSEN}"
echo "KPAR_CHOSEN_111: ${KPAR_CHOSEN_111}"
echo "KPAR_CHOSEN_222: ${KPAR_CHOSEN_222}"
echo "CLUSTER_NAME: ${CLUSTER_NAME}"
echo "CONFIG_dir: ${CONFIG_dir}"
echo "SETUP_dir: ${SETUP_dir}"
echo "MLDP_SCRIPTS: ${MLDP_SCRIPTS}"
echo "SCALEE array has ${#SCALEE[@]} elements"
echo "=========================="
echo ""

# echo "WARNING: Intentionally exiting."
# exit 1

# echo "SCALEE elements are:"
mkdir -p "V_est"
cd "V_est" || exit 1
V_est_dir=$(pwd)
echo "V_est_dir: ${V_est_dir}"

KP1_dir=$V_est_dir/KP1
KP2_dir=$V_est_dir/KP2

# if done_estimating_V doesn't exist in $V_est, then run
if [ ! -f "done_estimating_V" ]; then

    rm -rf $V_est_dir/cell_sizes_KPX.dat
    echo "cell_sizes_KP1; cell_sizes_KP2 (or the hp_calculations equivalent)" > $V_est_dir/cell_sizes_KPX.dat

    ########################################################################
    ########################################################################
    ########################################################################
    mkdir -p KP1
    cd KP1 || exit 1
    KP1_dir=$(pwd)
    echo "KP1_dir: ${KP1_dir}"

    # STEP 1: KPOINTS 111 sim
    if [ ! -f "done_KP1" ]; then

        rm -f slurm*

        cp $LOCAL_SETUP_dir/POSCAR_NPT $KP1_dir/POSCAR
        cp $SETUP_dir/POTCAR $KP1_dir
        cp $SETUP_dir/KPOINTS_111 $KP1_dir/KPOINTS # KPOINTS 111
        cp $SETUP_dir/INCAR_NPT $KP1_dir/INCAR
        cp $SETUP_dir/RUN_VASP_NPT.sh $KP1_dir/RUN_VASP.sh

        # Replace __..._CHOSEN__ with the chosen values in INCAR
        sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" INCAR
        sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" INCAR
        sed -i "s/__PSTRESS_CHOSEN__/${PSTRESS_CHOSEN}/" INCAR
        sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" INCAR
        sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" INCAR
        sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_111}/" INCAR
        sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" INCAR

        ########################################################################

        sbatch RUN_VASP.sh
        job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}')
        echo ""
        echo "------------------------"
        echo "Job for KPOINT 111 (${job_id}) submitted at $(date)"

        #after submission, now wait as the job gets finished
        while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id
        do 
            # echo "Job ${master_id}${letter} (${job_id}; ${counter}) under work"
            sleep ${WAIT_TIME_LONG}
        done

        echo "Job for KPOINT 111 (${job_id}) done at $(date)"

        touch done_KP1
    else
        echo "KPOINTS 111 simulation already done. Skipping..."
    fi

    # update and source data_4_analysis.sh
    cp $HELP_SCRIPTS/qmd/vasp/data_4_analysis.sh .
    source data_4_analysis.sh

    # estimate cell size for PSTRESS_CHOSEN using conda environment with ASE and python $HELP_SCRIPTS_vasp/eos_fit__V_at_P.py -p $PSTRESS_CHOSEN_GPa -e 0.01
    module purge
    module load anaconda3/2024.6; conda activate ase_env
    echo ""
    python $HELP_SCRIPTS_vasp/eos_fit__V_at_P.py -p $PSTRESS_CHOSEN_GPa # -e 0.01
    echo ""
    module purge

    # grab the number from analysis/log.eos_fit that comes in the line that follows the one with text "Estimated cell size" -- use bash
    cell_size_KP1=$(awk '/Estimated cell size/ {match($0, /([0-9]+(\.[0-9]+)?)$/, a); print a[1]}' analysis/log.eos_fit)
    echo "cell_size_KP1: ${cell_size_KP1}"

    echo $cell_size_KP1 >> $V_est_dir/cell_sizes_KPX.dat

    ########################################################################
    ########################################################################
    ########################################################################

    if [ $mode -eq 0 ]; then
        echo "Mode 0: KPOINTS 111 + KPOINTS 222"

        echo "Now creating KPOINTS 222 simulation directory..."
        cd $V_est_dir || exit 1
        mkdir -p KP2
        cd KP2 || exit 1
        KP2_dir=$(pwd)
        echo "KP2_dir: ${KP2_dir}"

        # STEP 2: KPOINTS 222 sim
        if [ ! -f "done_KP2" ]; then
            
            rm -f slurm*

            cp $SETUP_dir/POTCAR $KP2_dir
            cp $SETUP_dir/KPOINTS_222 $KP2_dir/KPOINTS # KPOINTS 222
            cp $SETUP_dir/INCAR_NPT $KP2_dir/INCAR
            cp $SETUP_dir/RUN_VASP_NPT.sh $KP2_dir/RUN_VASP.sh

            # see if $KP1_dir/CONTCAR exists, if not, then exit with error
            if [ ! -f "$KP1_dir/CONTCAR" ]; then
                echo "Error: $KP1_dir/CONTCAR not found. Please run KPOINTS 111 simulation first."
                exit 1
            fi
            cp $KP1_dir/CONTCAR $KP2_dir/POSCAR
            # replace the second line with $cell_size_KP1, and second line with "   1.0 0.0 0.0", third with "  0.0 1.0 0.0", and fourth with "   0.0 0.0 1.0"
            sed -i "2s/.*/$cell_size_KP1/" POSCAR
            sed -i "3s/.*/1.0000000000000000 0.0000000000000000 0.0000000000000000/" POSCAR
            sed -i "4s/.*/0.0000000000000000 1.0000000000000000 0.0000000000000000/" POSCAR
            sed -i "5s/.*/0.0000000000000000 0.0000000000000000 1.0000000000000000/" POSCAR

            # Replace __..._CHOSEN__ with the chosen values in INCAR
            sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" INCAR
            sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" INCAR
            sed -i "s/__PSTRESS_CHOSEN__/${PSTRESS_CHOSEN}/" INCAR
            sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" INCAR
            sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" INCAR
            sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_222}/" INCAR
            sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" INCAR

            ########################################################################

            sbatch RUN_VASP.sh
            job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}')
            echo ""
            echo "------------------------"
            echo "Job for KPOINT 222 (${job_id}) submitted at $(date)"

            #after submission, now wait as the job gets finished
            while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id
            do 
                # echo "Job ${master_id}${letter} (${job_id}; ${counter}) under work"
                sleep ${WAIT_TIME_LONG}
            done

            echo "Job for KPOINT 222 (${job_id}) done at $(date)"

            touch done_KP2
        else
            echo "KPOINTS 222 simulation already done. Skipping..."
        fi

        # update and source data_4_analysis.sh
        cp $HELP_SCRIPTS/qmd/vasp/data_4_analysis.sh .
        source data_4_analysis.sh

        # estimate cell size for PSTRESS_CHOSEN using l_ase conda environment and python $HELP_SCRIPTS_vasp/eos_fit__V_at_P.py -p $PSTRESS_CHOSEN_GPa -e 0.01
        module purge
        module load anaconda3/2024.6; conda activate ase_env
        echo ""
        python $HELP_SCRIPTS_vasp/eos_fit__V_at_P.py -p $PSTRESS_CHOSEN_GPa # -e 0.01
        echo ""
        module purge

        # grab the number from analysis/log.eos_fit that comes in the line that follows the one with text "Estimated cell size" -- use bash
        cell_size_KP2=$(awk '/Estimated cell size/ {match($0, /([0-9]+(\.[0-9]+)?)$/, a); print a[1]}' ${KP2_dir}/analysis/log.eos_fit)
        echo "cell_size_KP2: ${cell_size_KP2}"

        echo $cell_size_KP2 >> $V_est_dir/cell_sizes_KPX.dat

        ########################################################################
        ########################################################################
        ########################################################################

    elif [ $mode -eq 1 ]; then
        echo "Mode 1: KPOINTS 111 + hp_calculations"
        echo "Not doing KPOINTS 222 simulation in this mode."

        # move to KP1 directory
        echo "KP1_dir: ${KP1_dir}"
        cd $KP1_dir || exit 1
        # move to hp_calculations directory
        if [ ! -d "hp_calculations" ]; then
            echo "Error: hp_calculations directory doesn't exist. Please call eos_fit__V_at_P.py."
            exit 1
        else
            cd hp_calculations
            hp_calculations_dir=$(pwd)
            echo "hp_calculations_dir: ${hp_calculations_dir}"
        fi

        if [ ! -f "done_hp_calculations" ]; then

            cp $SETUP_dir/POTCAR .
            cp $SETUP_dir/KPOINTS_222 KPOINTS # KPOINTS 222
            cp $SETUP_dir/INCAR_relax INCAR
            cp $SETUP_dir/RUN_VASP_relax.sh RUN_VASP.sh

            # find the number right after ntasks-per-node= in the slurm script and nothing else
            NPAR_CHOSEN_hp_calculations=$(grep -oP '(?<=ntasks-per-node=)\d+' $SETUP_dir/RUN_VASP_relax.sh)

            # Replace __..._CHOSEN__ with the chosen values in INCAR
            sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" INCAR
            sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" INCAR
            sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN_hp_calculations}/" INCAR
            sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_222}/" INCAR
            sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" INCAR

            # copy POTCAR, KPOINTS, INCAR and RUN_VASP.sh to all folders in this directory
            find . -type d -exec cp POTCAR {} \;
            find . -type d -exec cp KPOINTS {} \;
            find . -type d -exec cp INCAR {} \;
            find . -type d -exec cp RUN_VASP.sh {} \;
            rm POTCAR KPOINTS INCAR RUN_VASP.sh 

            # go in each folder and run the following command: "sbatch RUN_VASP.sh"
            find . -type d -exec bash -c 'cd "$0" && sbatch RUN_VASP.sh' {} \;
            job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}') #grabbing the last job that was submitted
            job_id_2=$(squeue --user=$USER --sort=i --format=%i | tail -n 2 | head -n 1 | awk '{print $1}') #second last job

            echo "Running sbatch RUN_VASP.sh in all folders in hp_calculations directory. $(date)"
            echo "Keeping track of the last two jobs submitted in hp_calculations directory: ${job_id} and ${job_id_2}"

            # after submission, now wait as the last of the recal jobs gets finished
            while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id; do
                sleep ${WAIT_TIME_LONG}
            done
            sleep ${WAIT_TIME_SHORT}

            while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id_2; do
                sleep ${WAIT_TIME_LONG}
            done
            sleep ${WAIT_TIME_VLONG}

            echo "Jobs for hp_calculations (${job_id}) done at $(date)"

            python ${MLDP_SCRIPTS}/post_recal_rerun.py -ip all -v -ss $INPUT_FILES_DIR/sub_vasp_xtra.sh > log.recal_test 2>&1
            logfile="log.recal_test"
            line_count=$(wc -l < "$logfile")    # Count the number of lines in the file
            num_failed_recal_frames=$((${line_count}-14)) # name says it all ...

            if [ $num_failed_recal_frames -gt 0 ]; then
                echo "Error: $num_failed_recal_frames frames failed in the hp_calculations. Please check the log file."
                exit 1
            else
                echo "All frames passed in the hp_calculations."
                touch done_hp_calculations
            fi

        else
            echo "hp_calculations already done. Skipping..."
        fi

        # grep pressure (external pressure == total pressure in relaxation) from OUTCAR and save it in a file called pressure.txt
        mkdir -p $hp_calculations_dir/analysis
        grep "pressure" */OUTCAR | awk '{print $5}' > analysis/pressure.dat
        # grep vol
        grep -m 1 "volume" */OUTCAR | awk '{print $6}' > analysis/volume.dat

        module purge
        module load anaconda3/2024.6; conda activate ase_env
        echo ""
        python $HELP_SCRIPTS_vasp/eos_fit__V_at_P.py -p $PSTRESS_CHOSEN_GPa -m 1 # -e 0.01 # m/mode=1 since hp_calculations mode
        echo ""
        module purge

        # grab the number from analysis/log.eos_fit that comes in the line that follows the one with text "Estimated cell size" -- use bash
        cell_size_KP2=$(awk '/Estimated cell size/ {match($0, /([0-9]+(\.[0-9]+)?)$/, a); print a[1]}' analysis/log.eos_fit)
        echo "cell_size_KP2 (based on hp_calculations): ${cell_size_KP2}"

        echo $cell_size_KP2 >> $V_est_dir/cell_sizes_KPX.dat

        # In all folders in this directory

        ########################################################################
        ########################################################################
        ########################################################################
    fi


    ########################################################################
    cd $V_est_dir || exit 1
    touch done_estimating_V
    ########################################################################
    ########################################################################
    ########################################################################
else 
    echo "V_est already estimated. Skipping..."

    # read the cell sizes from cell_sizes_KPX.dat
    cell_size_KP1=$(awk 'NR==2' $V_est_dir/cell_sizes_KPX.dat)
    cell_size_KP2=$(awk 'NR==3' $V_est_dir/cell_sizes_KPX.dat)

    echo "cell_size_KP1: ${cell_size_KP1}"
    echo "cell_size_KP2: ${cell_size_KP2}"

    KP1_dir=$V_est_dir/KP1
    echo "KP1_dir: ${KP1_dir}"

fi






# STEP 3: SCALEE sim
cd $LOCAL_SETUP_dir || exit 1
echo "Now creating SCALEE simulation directories..."


# if mode is 0
if [ $mode -eq 0 ]; then

    KP2_dir=$V_est_dir/KP2
    echo "KP2_dir: ${KP2_dir}"

    # prep POSCAR for SCALEE
    cp $KP2_dir/CONTCAR $LOCAL_SETUP_dir/POSCAR_SCALEE
    sed -i "2s/.*/$cell_size_KP2/" $LOCAL_SETUP_dir/POSCAR_SCALEE
    sed -i "3s/.*/1.0000000000000000 0.0000000000000000 0.0000000000000000/" $LOCAL_SETUP_dir/POSCAR_SCALEE
    sed -i "4s/.*/0.0000000000000000 1.0000000000000000 0.0000000000000000/" $LOCAL_SETUP_dir/POSCAR_SCALEE
    sed -i "5s/.*/0.0000000000000000 0.0000000000000000 1.0000000000000000/" $LOCAL_SETUP_dir/POSCAR_SCALEE

    echo "Created POSCAR for SCALEE"

else

    KP2_dir=$KP1_dir # when mode == 1 
    echo "KP2_dir: ${KP2_dir}"

    # prep POSCAR for SCALEE
    cp $KP2_dir/CONTCAR $LOCAL_SETUP_dir/POSCAR_SCALEE
    sed -i "2s/.*/$cell_size_KP2/" $LOCAL_SETUP_dir/POSCAR_SCALEE
    sed -i "3s/.*/1.0000000000000000 0.0000000000000000 0.0000000000000000/" $LOCAL_SETUP_dir/POSCAR_SCALEE
    sed -i "4s/.*/0.0000000000000000 1.0000000000000000 0.0000000000000000/" $LOCAL_SETUP_dir/POSCAR_SCALEE
    sed -i "5s/.*/0.0000000000000000 0.0000000000000000 1.0000000000000000/" $LOCAL_SETUP_dir/POSCAR_SCALEE

    echo "Created POSCAR for SCALEE"
fi


cd $CONFIG_dir || echo "WARNING: Issue w cd ${CONFIG_dir}"

# echo "SCALEE elements are:"
echo "SCALEE array has ${#SCALEE[@]} elements"
echo "Creating/Updating dir for:"
counter=0

for SCALEE_CHOSEN in "${SCALEE[@]}"; do

    counter=$((counter+1))
    echo "SCALEE_CHOSEN ${counter}: ${SCALEE_CHOSEN}"

    mkdir -p "SCALEE_${counter}"
    # cp setup_TI/* "SCALEE_${counter}"
    cp $LOCAL_SETUP_dir/POSCAR_SCALEE "SCALEE_${counter}/POSCAR"
    cp $SETUP_dir/POTCAR "SCALEE_${counter}/POTCAR"
    cp $SETUP_dir/KPOINTS_111 "SCALEE_${counter}/KPOINTS" # KPOINTS 111 for SCALEE
    cp $SETUP_dir/INCAR_SCALEE "SCALEE_${counter}/INCAR"
    cp $SETUP_dir/RUN_VASP_SCALEE.sh "SCALEE_${counter}/RUN_VASP.sh"
    cp $HELP_SCRIPTS/qmd/vasp/data_4_analysis.sh "SCALEE_${counter}/data_4_analysis.sh"
    cd "SCALEE_${counter}" || exit 1
    rm -f slurm*

    # Replace __SCALEE_CHOSEN__ with the SCALEE_CHOSEN
    sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" "SCALEE_${counter}/INCAR"

    # Replace __TEMP_CHOSEN__ and __SIGMA_CHOSEN__ based on the chosen temperature
    sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" "SCALEE_${counter}/INCAR"
    sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" "SCALEE_${counter}/INCAR"
    sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" "SCALEE_${counter}/INCAR"

    # Replace __NPAR_CHOSEN__ with the chosen NPAR
    sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" "SCALEE_${counter}/INCAR"
    sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_111}/" "SCALEE_${counter}/INCAR"

    # Replace __NBANDS_CHOSEN__ with the chosen NBANDS
    sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" "SCALEE_${counter}/INCAR"

    if [ $run_switch -eq 1 ]; then
        sbatch RUN_VASP.sh
    fi

    cd $CONFIG_dir || exit 1

done


echo "Done submitting jobs for SCALEE simulations."






########################################################################
########################################################################
########################################################################
# start the final hp run (NPT)
cd $V_est_dir
mkdir -p "SCALEE_1__hp"
cd "SCALEE_1__hp" || exit 1
SCALEE_hp_dir=$(pwd)
echo "SCALEE_hp_dir: ${SCALEE_hp_dir}"

if [ ! -f "done_SCALEE_hp" ]; then
    if [ ! -f "running_SCALEE_hp" ]; then

        echo "Now creating SCALEE=1.0 - High accuracy simulation directory..."
        cp "$CONFIG_dir/SCALEE_1"/* $SCALEE_hp_dir/
        rm -f slurm*
        cp $SETUP_dir/RUN_VASP_SCALEE_hp.sh $SCALEE_hp_dir/RUN_VASP.sh
        cp $SETUP_dir/KPOINTS_222 $SCALEE_hp_dir/KPOINTS # KPOINTS 222 for SCALEE hp
        cp $HELP_SCRIPTS/qmd/vasp/data_4_analysis.sh $SCALEE_hp_dir/

        cp $SETUP_dir/INCAR_SCALEE INCAR
        # Replace __SCALEE_CHOSEN__ with the SCALEE_CHOSEN
        sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" $SCALEE_hp_dir/INCAR
        # Replace __TEMP_CHOSEN__ and __SIGMA_CHOSEN__ based on the chosen temperature
        sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN}/" $SCALEE_hp_dir/INCAR
        sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" $SCALEE_hp_dir/INCAR
        sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" $SCALEE_hp_dir/INCAR
        # Replace __NPAR_CHOSEN__ with the chosen NPAR
        sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" $SCALEE_hp_dir/INCAR
        sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_222}/" $SCALEE_hp_dir/INCAR
        # Replace __NBANDS_CHOSEN__ with the chosen NBANDS
        sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" $SCALEE_hp_dir/INCAR


        sbatch RUN_VASP.sh
        touch $SCALEE_hp_dir/running_SCALEE_hp

        job_id=$(squeue --user=$USER --sort=i --format=%i | tail -n 1 | awk '{print $1}')
        echo ""
        echo "------------------------"
        echo "Job for SCALEE=1.0 - High accuracy | KPOINT 222 (${job_id}) submitted at $(date)"

        #after submission, now wait as the job gets finished
        while  squeue --user=$USER --sort=i --format=%i | grep -q $job_id
        do 
            # echo "Job ${master_id}${letter} (${job_id}; ${counter}) under work"
            sleep ${WAIT_TIME_LONG}
        done

        echo "Job for SCALEE=1.0 - High accuracy | KPOINT 222 (${job_id}) done at $(date)"


        # update and source data_4_analysis.sh
        source data_4_analysis.sh
        rm $SCALEE_hp_dir/running_SCALEE_hp
        touch done_SCALEE_hp
    else
        echo "SCALEE=1.0 - High accuracy simulation already running. Exiting."
        exit 1
    fi
else
    echo "SCALEE=1.0 - High accuracy simulation already done. Skipping."
fi

########################################################################
########################################################################
########################################################################


cd $CONFIG_dir || exit 1


echo "Done."

