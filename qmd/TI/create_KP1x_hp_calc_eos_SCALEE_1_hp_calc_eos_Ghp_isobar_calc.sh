#!/usr/bin/env bash
shopt -s nullglob   # so that `for dir in */` does nothing if there are no folders

# Usage: source $HELP_SCRIPTS_TI/create_KP1x_hp_calc_eos_SCALEE_1_hp_calc_eos_Ghp_isobar_calc.sh > log.create_KP1x_hp_calc_eos_SCALEE_1_hp_calc_eos_Ghp_isobar_calc 2>&1 &
#        nohup $HELP_SCRIPTS_TI/create_KP1x_hp_calc_eos_SCALEE_1_hp_calc_eos_Ghp_isobar_calc.sh > log.create_KP1x_hp_calc_eos_SCALEE_1_hp_calc_eos_Ghp_isobar_calc 2>&1 &

# for each folder in the current directory, except, master_setup_TI, go inside and run source $HELP_SCRIPTS_TI/calculate_GFE_v2.sh 1 2 > log.calculate_GFE 2>&1 &



param_run_or_not=${1:-0}  # Default to 0 if not provided, which means do not run the script
num_isobar_calculations=${2:-4}  # Default to 6 if not provided
temp_gap_percentage=${3:-10}  # Default to 5 if not provided -- e.g. if TEMP_CHOSEN=13000, then the isobar calculations will be at 12000, 12500, 13000, 13500, 14000 for default values of both these variables



# Constants
kB=0.00008617333262145  # Boltzmann constant in eV/K


# throw error if num_isobar_calculations is not even
if (( num_isobar_calculations % 2 != 0 )); then
    echo "Error: num_isobar_calculations must be an even number."
    exit 1
fi

# make sure temp_gap_percentage is a positive integer
if ! [[ "$temp_gap_percentage" =~ ^[0-9]+$ ]] || (( temp_gap_percentage <= 0 )); then
    echo "Error: temp_gap_percentage must be a positive integer."
    exit 1
fi

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



echo 
echo
echo




# TEMP_CHOSEN_ISOBAR = [TEMP_CHOSEN * (1 - (temp_gap_percentage/100) * i) for i in (range(num_isobar_calculations)/2)] and [TEMP_CHOSEN * (1 + (temp_gap_percentage/100) * i) for i in (range(num_isobar_calculations)/2)]
# determine half‐count
half=$(( num_isobar_calculations / 2 ))

# initialize array
TEMP_CHOSEN_ISOBAR=()

# lower‐side: TEMP_CHOSEN * (1 - (gap%/100) * i), rounded to nearest 10
for (( i=1; i<=half; i++ )); do
    val=$(awk -v T="$TEMP_CHOSEN" -v p="$temp_gap_percentage" -v i="$i" \
        'BEGIN {
            res = T * (1 - (p/100)*i);
            # round to nearest 10:
            printf "%d", int((res + 5) / 10) * 10
        }')
    TEMP_CHOSEN_ISOBAR+=("$val")
done

# upper‐side: TEMP_CHOSEN * (1 + (gap%/100) * i), rounded to nearest 10
for (( i=1; i<=half; i++ )); do
    val=$(awk -v T="$TEMP_CHOSEN" -v p="$temp_gap_percentage" -v i="$i" \
        'BEGIN {
            res = T * (1 + (p/100)*i);
            printf "%d", int((res + 5) / 10) * 10
        }')
    TEMP_CHOSEN_ISOBAR+=("$val")
done

# display
# printf 'Isobar temperatures (%d values): %s\n' "${#TEMP_CHOSEN_ISOBAR[@]}" "${TEMP_CHOSEN_ISOBAR[*]}"

echo ""
echo ""
echo ""
echo "=========================="
printf 'Isobar temperatures (%d values): %s\n' "${#TEMP_CHOSEN_ISOBAR[@]}" "${TEMP_CHOSEN_ISOBAR[*]}"
echo "=========================="
echo ""
echo ""
echo ""

for dir in */; do
    # skip the master_setup_TI folder
    [[ "$dir" == "master_setup_TI/" ]] && continue


    # skip if the CONFIG_dir does not contain _0 or _8
    # skip unless there's “_0” or “_8” anywhere in the string
    if [[ ! $dir =~ _[08] ]]; then
        echo "Skipping $dir – it doesn't contain _0 or _8."
        cd "$PT_dir" || exit 1
        continue
    fi


    (
        cd "$dir" || exit 1
        dir_address=$(pwd)
        CONFIG_dir=$dir_address
        echo ""
        echo "Current directory: $dir_address"
        echo ""


        # if CONFIG_dir contains "_0", then check if SCALEE_1 directory exists. If it does, move on.
        # if SCALEE_1 directory doesn't exist, then create it + copy the POSCAR file from directory ../*_1H*/SCALEE_1
        # in this POSCAR, replace the last " 1" with " 0" in the seventh line
        # add all the numbers in the seventh line to a variable called num_atoms and 
        # then remove all lines from this copied POSCAR that are after the (8+num_atoms) number of lines
        if [[ "$CONFIG_dir" == *_0* ]]; then
            dest="$CONFIG_dir/SCALEE_1"

            if [[ -d "$dest" ]]; then
                echo "SCALEE_1 already exists in $CONFIG_dir, skipping."
            else
                echo "Creating $dest"
                mkdir -p "$dest"

                # find first matching source SCALEE_1 under any ../*_1H*/SCALEE_1
                src=$(echo ../*_1H*/SCALEE_1 2>/dev/null | awk '{print $1}')
                if [[ ! -d "$src" ]]; then
                echo "Error: no source dir found for SCALEE_1" >&2
                continue
                fi

                # copy CONTCAR if present, else POSCAR
                if [[ -f "$src/CONTCAR" ]]; then
                echo "Using CONTCAR from $src"
                cp "$src/CONTCAR" "$dest/POSCAR"
                elif [[ -f "$src/POSCAR" ]]; then
                echo "Using POSCAR from $src"
                cp "$src/POSCAR" "$dest/POSCAR"
                else
                echo "Error: neither CONTCAR nor POSCAR in $src" >&2
                continue
                fi

                # if COMPOSITION_dir_name contains "MgSiO3", then
                if [[ "$COMPOSITION_dir_name" == *"MgSiO3"* ]]; then
                    echo "Primary component is MgSiO3, modifying POSCAR accordingly."
                    # change the last ' 1' on line 7 to ' '
                    sed -i '7s/\(.*\) 1$/\1 /' "$dest/POSCAR"

                    # delete the last word on sixth line
                    awk 'NR==6 { $NF=""; sub(/[[:space:]]+$/, ""); }1' \
                        "$dest/POSCAR" > "$dest/POSCAR.tmp" && mv "$dest/POSCAR.tmp" "$dest/POSCAR"

                    # count atoms by adding all numbers in the seventh line
                    num_atoms=$(awk 'NR==7 {sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum}' "$dest/POSCAR")
                    echo "Number of atoms in POSCAR: $num_atoms"


                    echo "Truncating to $((8 + num_atoms)) lines (8 header + $num_atoms atoms)"

                    # truncate the file
                    head -n $((8 + num_atoms)) "$dest/POSCAR" > "$dest/POSCAR.tmp"
                    echo ""
                elif [[ "$COMPOSITION_dir_name" == *"Fe"* ]]; then
                    echo "Primary component is Fe, modifying POSCAR accordingly."
                    # change the first ' 1' on line 7 to ' '
                    sed -i '7s/ 1 / /' "$dest/POSCAR"

                    # delete the first word on sixth line
                    awk 'NR==6 { $1=""; sub(/^[[:space:]]+/, ""); }1' \
                        "$dest/POSCAR" > "$dest/POSCAR.tmp" && mv "$dest/POSCAR.tmp" "$dest/POSCAR"
                    # count atoms by adding all numbers in the seventh line
                    num_atoms=$(awk 'NR==7 {sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum}' "$dest/POSCAR")
                    echo "Number of atoms in POSCAR: $num_atoms"

                    echo "Truncating to $((8 + num_atoms)) lines (8 header + $num_atoms atoms)"
                    # truncate the file by deleting the ninth line only
                    awk 'NR!=9' "$dest/POSCAR" > "$dest/POSCAR.tmp" && mv "$dest/POSCAR.tmp" "$dest/POSCAR"
                    echo ""
                else
                    echo "Unknown composition: $COMPOSITION_dir_name"
                    exit 1
                fi

                mv "$dest/POSCAR.tmp" "$dest/POSCAR"
                echo "POSCAR copied and modified in $dest"
            fi
        fi

        echo ""


        mkdir -p isobar_calc  # Create the isobar_calculations directory if it doesn't exist
        ISOBAR_CALC_dir="$CONFIG_dir/isobar_calc"

        # make as many isobar calculation directories as specified by num_isobar_calculations with a gap of temp_gap_percentage and format the names as T{TEMP_CHOSEN_ISOBAR}}
        for TEMP_CHOSEN_ISOBAR_i in "${TEMP_CHOSEN_ISOBAR[@]}"; do
            # Create the directory name
            TEMP_CHOSEN_ISOBAR_dirname="T${TEMP_CHOSEN_ISOBAR_i}"
            TEMP_CHOSEN_ISOBAR_dir="$ISOBAR_CALC_dir/$TEMP_CHOSEN_ISOBAR_dirname"

            # Check if the directory already exists
            if [ -d "$TEMP_CHOSEN_ISOBAR_dir" ]; then
                echo "Directory $TEMP_CHOSEN_ISOBAR_dir already exists, skipping..."
                continue
            fi

            # Create the directory
            mkdir -p "$TEMP_CHOSEN_ISOBAR_dir"
            echo "Created directory: $TEMP_CHOSEN_ISOBAR_dir"

            cd "$TEMP_CHOSEN_ISOBAR_dir" || exit 1

            cp $CONFIG_dir/SCALEE_1/POSCAR $TEMP_CHOSEN_ISOBAR_dir/POSCAR
            num_atoms=$(python -c "from ase.io import read; print(len(read('$TEMP_CHOSEN_ISOBAR_dir/POSCAR').get_chemical_symbols()))")
            echo "Number of atoms in POSCAR: $num_atoms"
            # echo "Number of atoms in POSCAR: $num_atoms"
            # Remove all lines after (8 + num_atoms) lines
            awk -v num_atoms="$num_atoms" 'NR <= (8 + num_atoms)' "$TEMP_CHOSEN_ISOBAR_dir/POSCAR" > "$TEMP_CHOSEN_ISOBAR_dir/POSCAR.tmp"
            mv "$TEMP_CHOSEN_ISOBAR_dir/POSCAR.tmp" "$TEMP_CHOSEN_ISOBAR_dir/POSCAR"

            CONFIG_dir_name=$(basename "$CONFIG_dir")
            # if "_0" in the CONFIG_dir_name and COMPOSITION_dir_name contains "MgSiO3", copy corresponding POTCARs
            if [[ "$CONFIG_dir_name" == *_0* && "$COMPOSITION_dir_name" == *"MgSiO3"* ]]; then
                echo "Copying POTCAR for MgSiO3"
                cp $SETUP_dir/POTCAR_MgSiO3 "$TEMP_CHOSEN_ISOBAR_dir/POTCAR"
            elif [[ "$CONFIG_dir_name" == *_0* && "$COMPOSITION_dir_name" == *"Fe"* ]]; then
                echo "Copying POTCAR for Fe"
                cp $SETUP_dir/POTCAR_Fe "$TEMP_CHOSEN_ISOBAR_dir/POTCAR"
            else
                echo "Copying default POTCAR"
                cp $SETUP_dir/POTCAR "$TEMP_CHOSEN_ISOBAR_dir/POTCAR"
            fi
            # cp $SETUP_dir/POTCAR "$TEMP_CHOSEN_ISOBAR_dir/POTCAR"
            cp $SETUP_dir/KPOINTS_111 "$TEMP_CHOSEN_ISOBAR_dir/KPOINTS" # KPOINTS 111
            cp ${SETUP_dir}/INCAR_NPT    $TEMP_CHOSEN_ISOBAR_dir/INCAR

            # GPa to Kbar
            PSTRESS_CHOSEN=$(echo "$PSTRESS_CHOSEN_GPa * 10" | bc -l)
            # Calculate SIGMA_CHOSEN based on the chosen temperature
            SIGMA_CHOSEN=$(echo "$kB * $TEMP_CHOSEN_ISOBAR_i" | bc -l)

            # Populate INCAR placeholders
            sed -i "s/__TEMP_CHOSEN__/${TEMP_CHOSEN_ISOBAR_i}/" $TEMP_CHOSEN_ISOBAR_dir/INCAR
            sed -i "s/__NBANDS_CHOSEN__/${NBANDS_CHOSEN}/" $TEMP_CHOSEN_ISOBAR_dir/INCAR
            sed -i "s/__POTIM_CHOSEN__/${POTIM_CHOSEN}/" $TEMP_CHOSEN_ISOBAR_dir/INCAR
            sed -i "s/__NPAR_CHOSEN__/${NPAR_CHOSEN}/" $TEMP_CHOSEN_ISOBAR_dir/INCAR
            sed -i "s/__KPAR_CHOSEN__/${KPAR_CHOSEN_111}/" $TEMP_CHOSEN_ISOBAR_dir/INCAR
            sed -i "s/__SIGMA_CHOSEN__/${SIGMA_CHOSEN}/" $TEMP_CHOSEN_ISOBAR_dir/INCAR
            sed -i "s/__SCALEE_CHOSEN__/${SCALEE_CHOSEN}/" $TEMP_CHOSEN_ISOBAR_dir/INCAR
            sed -i "s/__PSTRESS_CHOSEN__/${PSTRESS_CHOSEN}/" $TEMP_CHOSEN_ISOBAR_dir/INCAR

            cp $SETUP_dir/RUN_VASP_NPT.sh "$TEMP_CHOSEN_ISOBAR_dir/RUN_VASP.sh"

            mkdir -p V_est/KP1 # follow the usual structure from before
            cp $TEMP_CHOSEN_ISOBAR_dir/* $TEMP_CHOSEN_ISOBAR_dir/V_est/KP1/  # copy all files to V_est/KP1
            rm $TEMP_CHOSEN_ISOBAR_dir/*  # remove all files (not dirs, e.g., V_est) from the current directory

            cd "$TEMP_CHOSEN_ISOBAR_dir/V_est/KP1" || exit 1
            # Submit the VASP job
            if [[ "$param_run_or_not" == "1" ]]; then
                echo "Submitting VASP job in $TEMP_CHOSEN_ISOBAR_dir"
                # sbatch --job-name="isobar_calc_${TEMP_CHOSEN_ISOBAR_i}" --output="log.isobar_calc_${TEMP_CHOSEN_ISOBAR_i}" --error="log.isobar_calc_${TEMP_CHOSEN_ISOBAR_i}" "$TEMP_CHOSEN_ISOBAR_dir/RUN_VASP.sh"
                sbatch RUN_VASP.sh

                # list just the job IDs for your user, sort them numerically, and take the last one
                LAST_JOB_ID=$(squeue -u $USER -h -o "%i" | sort -n | tail -1)
                echo "Started VASP in $KP1_dir"
                echo "JOB_ID_KP1: $LAST_JOB_ID"
                echo ""
            else
                echo "Skipping VASP job submission in $TEMP_CHOSEN_ISOBAR_dir (param_run_or_not is set to 0)"
            fi

        done



        echo ""
        cd "$PT_dir" || exit 1
    )
    sleep ${WAIT_TIME_SHORT}
done



# find all isobar_calc directories and run the isobar calculations in parallel
while IFS= read -r -d '' isobar_calc_dir; do
    (
        cd "$isobar_calc_dir" || exit 1
        echo "Processing isobar_calc directory: $isobar_calc_dir"

        nohup $HELP_SCRIPTS_TI/isobar__create.sh > log.isobar__create 2>&1 &
        
        echo "Initiated isobar calculation in $isobar_calc_dir"

        sleep ${WAIT_TIME_SHORT}

        cd "$PT_dir" || exit 1
    )
done < <(find . -type d -name 'isobar_calc' -print0)



echo ""
echo "All calculations/directories initiated. $(date)."
echo ""