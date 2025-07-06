#!/bin/bash

# Find all folders and sub-folders starting with "SCALEE" and run "source data_analysis.sh" in each
# Also, marks directories with specific conditions such as those that stopped prematurely, e.g., with less than 1000 time steps
# Usage:    nohup $HELP_SCRIPTS_TI/source_all_SCALEE.sh > log.source_all_SCALEE 2>&1 &
#           nohup $LOCAL_HELP_SCRIPTS_TI/source_all_SCALEE.sh > log.source_all_SCALEE 2>&1 &

# Check if the script is being run from the correct directory
current_dir=$(pwd)
parent_dir=$current_dir

echo "======================================================"
echo "Running source_all_SCALEE.sh in $parent_dir"
echo "Current time: $(date)"
echo "======================================================"
# echo ""

pids=()



# ========================================================
# ========================================================
# ========================================================

KEYWORD_TO_EVALUATE="SCALEE_1" # or "SCALEE_0" if you want to run it in SCALEE_0 directories

mode_data_4_analysis=1 # 0: don't run data_4_analysis.sh everywhere, 1: run
mode_track=1 # 0: don't track, 1: track how long all sims ran 
mode_restart_extend=1 # 0: don't extend, 1: extend the simulations (if SCALEE_0b last, create SCALEE_0c, etc.)
ENABLE_CREATION=false # if true, create directories and copy contents; if false, just print what would be done

# ========================================================
# ========================================================
# ========================================================





# if mode_restart_extend=1 but mode_track=0, then set mode_track=1
if [ $mode_restart_extend -eq 1 ] && [ $mode_track -eq 0 ]; then
    echo "======================================================="
    echo "NOTE: Setting mode_track=1 because mode_restart_extend=1"
    echo "======================================================="
    echo ""
    mode_track=1
fi


# echo all parameters
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Parameters:"
echo "KEYWORD_TO_EVALUATE: $KEYWORD_TO_EVALUATE"
echo "mode_data_4_analysis: $mode_data_4_analysis"
echo "mode_track: $mode_track"
echo "mode_restart_extend: $mode_restart_extend"
echo "ENABLE_CREATION: $ENABLE_CREATION"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""



# # ========================================================
# # run data_analysis.sh in all SCALEE folders
# # =======================================================
if [ $mode_data_4_analysis -eq 1 ]; then
    for dir in $(find . -type d -name "${KEYWORD_TO_EVALUATE}"); do ### exact match directories ###
    # for dir in $(find . -type d -name "${KEYWORD_TO_EVALUATE}*"); do ### all related directories ###
    # for dir in $(find . -type d -name "SCALEE*"); do
        echo ""
        if [ -d "$dir" ]; then

            # skip if path has "misc"
            if [[ "$dir" == *"misc"* ]]; then
                echo "Skipping $dir, path contains 'misc'."
                echo ""
                continue
            fi

            # check if OUTCAR file exists
            if [ ! -f "$dir/OUTCAR" ]; then
                echo "No OUTCAR file found in $dir, skipping..."
                echo ""
                continue
            fi
            echo "Running data_analysis.sh in $dir"
            cd "$dir" 
            cp $LOCAL_HELP_SCRIPTS_vasp/data_4_analysis.sh . # from the local "rsync" copy
            # cp $parent_dir/data_4_analysis.sh .
            # nohup bash data_4_analysis.sh > log.data_4_analysis 2>&1 &
            source data_4_analysis.sh > log.data_4_analysis 2>&1 &
            pids+=($!)          # $! is the PID of the backgrounded command
            cd $parent_dir
        else
            echo "$dir is not a directory."
        fi
        echo ""
    done

    echo "======================================================"
    echo "Waiting for all data_analysis.sh scripts to finish..."
    echo "======================================================"
    # Now wait only for those jobs
    wait "${pids[@]}"


    echo ""
    echo "======================================================"
    echo "All data_analysis.sh scripts have been run in all SCALEE folders in $parent_dir."
    echo "Time: $(date)"
    echo "======================================================"
fi # end of mode_data_4_analysis







echo ""
echo ""
echo "======================================================="
echo "======================================================="
echo ""
echo ""







if [ $mode_track -eq 1 ]; then
    echo "======================================================="
    echo "Running grep 'time' analysis/peavg.out in all $KEYWORD_TO_EVALUATE directories"
    echo "======================================================="
    echo ""
    counter_failed=0
    counter_dir=0
    counter_lt_1000=0
    counter_lt_5000=0
    counter_lt_10000=0
    counter_lt_20000=0
    dir_array_extend=()
    dir_array_restart=()
    # Again go to all these SCALEE_* directories and run <grep "time" analysis/peavg.out>; along with the directory address
    # for dir in $(find . -type d -name "$KEYWORD_TO_EVALUATE*"); do
    for dir in $(find . -type d -name "$KEYWORD_TO_EVALUATE"); do
    # for dir in $(find . -type d -path '*/'*0H*'/SCALEE_1'); do
    # for dir in $(find . -type d -name "SCALEE*"); do
        if [ -d "$dir" ]; then

            # echo "Looking in directory: $dir"

            # if this is SCALEE_0a, SCALEE_0b, etc., skip it
            # if [[ "$dir" =~ SCALEE_0[a-z] ]]; then
            #     continue
            # fi

            # make sure grandparent folder is "isobar_calc"
            # grandparent_dir=$(dirname "$(dirname "$dir")")
            # if [ "$(basename "$grandparent_dir")" != "isobar_calc" ]; then
            #     # echo "Skipping $dir, grandparent folder is not 'isobar_calc'."
            #     continue
            # fi

            # skip if path has "misc"
            if [[ "$dir" == *"misc"* ]]; then
                echo "Skipping $dir, path contains 'misc'."
                echo ""
                continue
            fi

            # echo "Running grep in $dir"
            cd "$dir"
            abs_dir=$(pwd)
            counter_dir=$((counter_dir + 1))
            # see if analysis/peavg.out exists
            if [ ! -f "analysis/peavg.out" ]; then
                # echo "No analysis/peavg.out file found in $dir."
                time_steps=""
            else
                time_steps=$(grep "time" analysis/peavg.out | awk '{print $5}')
            fi

            # total number of ${KEYWORD_TO_EVALUATE}* directories in $abs_dir/.. (e.g. SCALEE_0a, SCALEE_0b, etc.)
            total_child_dirs=$(find "$abs_dir/.." -type d -name "${KEYWORD_TO_EVALUATE}*" | wc -l)
            total_child_dirs=$((total_child_dirs - 1)) # subtract 1 to not count the current directory itself
            

            if [ -z "$time_steps" ]; then
                echo "Warning: time_steps not found"
                echo "Directory: $dir"
                echo "Total child directories: $total_child_dirs"
                echo ""
                counter_failed=$((counter_failed + 1))
                touch $abs_dir/to_RUN__failed
                dir_array_restart+=("$abs_dir") # add to array for restarting simulations
            elif [ "$time_steps" -lt 1000 ]; then
                echo "Warning: time_steps < 1000"
                echo "Directory: $dir"
                echo "Total child directories: $total_child_dirs"
                echo "Time steps: $time_steps"
                echo ""
                counter_lt_1000=$((counter_lt_1000 + 1))
                touch $abs_dir/to_RUN__lt_1000
                dir_array_restart+=("$abs_dir") # add to array for restarting simulations
                # cd ..
                # cp -r SCALEE_0 SCALEE_0a &
            elif [ "$time_steps" -lt 5000 ]; then
                echo "Warning: time_steps < 5000"
                echo "Directory: $dir"
                echo "Total child directories: $total_child_dirs"
                echo "Time steps: $time_steps"
                echo ""
                counter_lt_5000=$((counter_lt_5000 + 1))
                touch $abs_dir/to_RUN__1000_to_5000
                dir_array_extend+=("$abs_dir") # add to array for extending simulations
                # cd ..
                # cp -r SCALEE_0 SCALEE_0a &
            elif [ "$time_steps" -lt 10000 ]; then
                echo "Warning: time_steps < 10000"
                echo "Directory: $dir"
                echo "Total child directories: $total_child_dirs"
                echo "Time steps: $time_steps"
                echo ""
                counter_lt_10000=$((counter_lt_10000 + 1))
                touch $abs_dir/to_RUN__5000_to_10000
                dir_array_extend+=("$abs_dir") # add to array for extending simulations
                # cd ..
                # cp -r SCALEE_0 SCALEE_0a &
            elif [ "$time_steps" -lt 20000 ]; then
                echo "Warning: time_steps < 20000"
                echo "Directory: $dir"
                echo "Total child directories: $total_child_dirs"
                echo "Time steps: $time_steps"
                echo ""
                counter_lt_20000=$((counter_lt_20000 + 1))
                touch $abs_dir/to_RUN__10000_to_20000
                dir_array_extend+=("$abs_dir") # add to array for extending simulations
                # cd ..
                # cp -r SCALEE_0 SCALEE_0a &
            else
                # echo "Time steps > 20000"
                echo "Directory: $dir"
                echo "Total child directories: $total_child_dirs"
                echo "Time steps: $time_steps"
                echo ""
            fi

            #     echo "Time steps in $dir: $time_steps"
            # fi
            # echo "Directory: $dir"
            # echo ""
            cd $parent_dir
        fi
    done


    # Summarize the results
    counter_lt_20000=$((counter_lt_20000 + counter_lt_10000 + counter_lt_5000 + counter_lt_1000))
    counter_lt_10000=$((counter_lt_10000 + counter_lt_5000 + counter_lt_1000))
    counter_lt_5000=$((counter_lt_5000 + counter_lt_1000))

    counter_1000_to_5000=$((counter_lt_5000 - counter_lt_1000))
    counter_5000_to_10000=$((counter_lt_10000 - counter_lt_5000))
    counter_10000_to_20000=$((counter_lt_20000 - counter_lt_10000))

    echo ""
    echo "======================================================="
    echo "Directories checked: $counter_dir"
    echo ""
    echo "Directories with time_steps < 20000: $counter_lt_20000"
    echo "Directories with time_steps < 10000: $counter_lt_10000"
    echo "Directories with time_steps < 5000: $counter_lt_5000"
    echo "Directories with time_steps < 1000: $counter_lt_1000"
    echo ""
    echo "Directories with 10000 < time_steps < 20000 (touch to_RUN__10000_to_20000): $counter_10000_to_20000"
    echo "Directories with 5000 < time_steps < 10000 (touch to_RUN__5000_to_10000): $counter_5000_to_10000"
    echo "Directories with 1000 < time_steps < 5000 (touch to_RUN__1000_to_5000): $counter_1000_to_5000"
    echo "Directories with time_steps < 1000 (touch to_RUN__lt_1000): $counter_lt_1000"
    echo ""
    echo "Directories with time_steps not found (touch to_RUN__failed): $counter_failed"
    echo "======================================================="
    echo ""
fi # end of mode_track





echo ""
echo ""
echo "======================================================="
echo "======================================================="
echo ""
echo ""





if [ $mode_restart_extend -eq 1 ]; then


    # check if there are existing to_RUN__SCALEE_0 files, and exit if so
    if find . -type f -name "to_RUN__SCALEE_0" | grep -q .; then
        echo "ERROR: to_RUN__SCALEE_0 files already exist in some SCALEE directories. Please remove them before running this script." >&2
        exit 1
    fi
    # remove all to_RUN__SCALEE_0 files in all SCALEE directories
    # find . -type f -name "to_RUN__SCALEE_0" -exec rm -f {} \;

    echo "======================================================="
    echo "Running extend+restart simulations in all SCALEE directories"
    echo "======================================================="
    echo ""
    echo "Directories to extend simulations: ${#dir_array_extend[@]}"
    echo "Directories to restart simulations: ${#dir_array_restart[@]}"
    echo ""

    # Extend simulations in directories with 1000 < time_steps < 20000
    for abs_dir in "${dir_array_extend[@]}"; do
        echo "Extending simulation in $abs_dir"
        cd "$abs_dir" || continue
        cd ..
        # If there's a SCALEE_0{X}, etc., copy the last one to SCALEE_0{X+1} but if only SCALEE_0, copy it to SCALEE_0a
        # find last SCALEE_0* directory
        # 1) Find the last SCALEE_0* dir (sorted lexically)
        last_SCALEE_0_dir=$(ls -d SCALEE_0* 2>/dev/null | sort | tail -n1)
        if [[ ! -d "$last_SCALEE_0_dir" ]]; then
            echo "ERROR: no SCALEE_0* directory found" >&2
            exit 1
        fi

        # 2) Strip off the “SCALEE_0” prefix to get the letter suffix (if any)
        suffix=${last_SCALEE_0_dir#SCALEE_0}

        if [[ -z "$suffix" ]]; then
            # No suffix → last_SCALEE_0_dir=last_SCALEE_0_dir“a”
            if $ENABLE_CREATION; then
                cp -r "$last_SCALEE_0_dir" "$last_SCALEE_0_dir"a
            fi
            last_SCALEE_0_dir="$last_SCALEE_0_dir"a
            suffix=${last_SCALEE_0_dir#SCALEE_0}
        # else
        #     echo "This was accounted for in the last attempt."
        #     echo ""
        #     continue
        fi

        # Must be a single lowercase letter (a–z)
        if [[ ! "$suffix" =~ ^[a-z]$ ]]; then
            echo "ERROR: unexpected suffix ‘$suffix’ in $last_SCALEE_0_dir" >&2
            exit 1
        fi

        # Convert letter → ASCII code → increment → back to char
        code=$(printf "%d" "'$suffix")     # e.g. 'a' → 97
        (( code++ ))                        # 97→98
        if (( code > 122 )); then           # ‘z’ is 122
            echo "ERROR: can’t go past ‘z’" >&2
            exit 1
        fi
        # printf '\\%03o' to get back an escaped octal char
        next_letter=$(printf "\\$(printf '%03o' "$code")")
        next_SCALEE_0_dir="SCALEE_0${next_letter}"
        # fi

        echo "Creating new directory: $next_SCALEE_0_dir"


        if $ENABLE_CREATION; then
            mkdir -p "$next_SCALEE_0_dir" \
                || { echo "❌ Failed to create directory $next_SCALEE_0_dir"; echo ""; continue; }

            # copy everything (including hidden files) from last → next
            cp -a "$last_SCALEE_0_dir"/. "$next_SCALEE_0_dir"/ \
                || { echo "❌ Failed to copy contents to $next_SCALEE_0_dir"; echo ""; continue; }

            echo "✅ Contents copied to $next_SCALEE_0_dir to extend simulation."

            pushd "$next_SCALEE_0_dir" > /dev/null \
                || { echo "❌ Failed to cd into $next_SCALEE_0_dir"; echo ""; continue; }

            cp CONTCAR POSCAR \
                || { echo "❌ Failed to copy CONTCAR → POSCAR"; popd > /dev/null; continue; }

            touch to_RUN__SCALEE_0 \
                || echo "⚠️  Couldn’t create marker file to_RUN__SCALEE_0"

            popd > /dev/null
        else
            echo "[DRY RUN] mkdir -p \"$next_SCALEE_0_dir\""
            echo "[DRY RUN] cp -a \"$last_SCALEE_0_dir\"/. \"$next_SCALEE_0_dir\"/"
            echo "[DRY RUN] cp CONTCAR → POSCAR & touch to_RUN__SCALEE_0"
        fi

        cd $parent_dir || exit 1

        echo ""

    done


    # Restart simulations in directories with time_steps not found or < 1000
    for abs_dir in "${dir_array_restart[@]}"; do
        echo "Restarting simulation in $abs_dir"
        cd "$abs_dir" || continue
        cd ..
        # If there's a SCALEE_0{X}, etc., copy the last_SCALEE_0_dir one to SCALEE_0{X+1} but if only SCALEE_0, copy it to SCALEE_0a
        # find last_SCALEE_0_dir SCALEE_0* directory
        last_SCALEE_0_dir=$(ls -d SCALEE_0* 2>/dev/null | sort | tail -n1)
        if [[ ! -d "$last_SCALEE_0_dir" ]]; then
            echo "ERROR: no SCALEE_0* directory found" >&2
            exit 1
        fi
        next_SCALEE_0_dir=$last_SCALEE_0_dir # restart in the same directory
        echo "Using existing directory $next_SCALEE_0_dir for restarting simulations."

        if $ENABLE_CREATION; then
            pushd "$next_SCALEE_0_dir" > /dev/null \
            || { echo "❌ Failed to enter $next_SCALEE_0_dir"; echo ""; continue; }

            # cp CONTCAR POSCAR \ # not applicable here, as we are restarting in the same directory
            # || { echo "❌ Failed to copy CONTCAR → POSCAR"; popd > /dev/null; continue; }

            touch to_RUN__SCALEE_0 \
            || echo "⚠️  Could not create marker file to_RUN__SCALEE_0"

            popd > /dev/null
        else
            echo "[DRY RUN] pushd \"$next_SCALEE_0_dir\""
            echo "[DRY RUN] cp CONTCAR POSCAR"
            echo "[DRY RUN] touch to_RUN__SCALEE_0"
        fi

        cd $parent_dir || exit 1

        echo ""

    done

    echo ""
    echo "======================================================="
    echo "All simulations have been extended or restarted."
    echo "Time: $(date)"
    echo "======================================================="
fi # end of mode_restart_extend