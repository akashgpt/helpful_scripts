#!/bin/bash

# Find all folders and sub-folders starting with "SCALEE" and run "source data_analysis.sh" in each

# Usage: nohup $HELP_SCRIPTS_TI/source_all_SCALEE.sh > log.source_all_SCALEE 2>&1 &

# Check if the script is being run from the correct directory
current_dir=$(pwd)
parent_dir=$current_dir

echo "======================================================"
echo "Running data_analysis.sh in all SCALEE folders in $parent_dir at $(date)"
echo "======================================================"

pids=()



# for dir in $(find . -type d -name "SCALEE*"); do
#     echo ""
#     if [ -d "$dir" ]; then
#         # check if OUTCAR file exists
#         if [ ! -f "$dir/OUTCAR" ]; then
#             echo "No OUTCAR file found in $dir, skipping..."
#             continue
#         fi
#         echo "Running data_analysis.sh in $dir"
#         cd "$dir" 
#         cp $LOCAL_HELP_SCRIPTS/qmd/vasp/data_4_analysis.sh . # from the local "rsync" copy
#         # cp $parent_dir/data_4_analysis.sh .
#         # nohup bash data_4_analysis.sh > log.data_4_analysis 2>&1 &
#         source data_4_analysis.sh > log.data_4_analysis 2>&1 &
#         pids+=($!)          # $! is the PID of the backgrounded command
#         cd $parent_dir
#     else
#         echo "$dir is not a directory."
#     fi
#     echo ""
# done

echo "======================================================"
echo "Waiting for all data_analysis.sh scripts to finish..."
echo "======================================================"
# Now wait only for those jobs
wait "${pids[@]}"


echo ""
echo "======================================================"
echo "All data_analysis.sh scripts have been run in all SCALEE folders in $parent_dir at $(date)"
echo "======================================================"
echo ""
echo ""
echo "======================================================="
echo "======================================================="
echo ""
echo ""
echo "======================================================="
echo "Now running grep 'time' analysis/peavg.out in all SCALEE_* directories"
echo "======================================================="
echo ""
counter_failed=0
counter_dir=0
counter_lt_1000=0
counter_lt_5000=0
counter_lt_10000=0
counter_lt_20000=0
# Again go to all these SCALEE_* directories and run <grep "time" analysis/peavg.out>; along with the directory address
for dir in $(find . -type d -name "SCALEE*"); do
    if [ -d "$dir" ]; then
        # echo "Running grep in $dir"
        cd "$dir"
        abs_dir=$(pwd)
        counter_dir=$((counter_dir + 1))
        time_steps=$(grep "time" analysis/peavg.out | awk '{print $5}')

        if [ "$time_steps" -lt 1000 ]; then
            echo "Warning: time_steps < 1000"
            echo "Directory: $dir"
            echo "Time steps: $time_steps"
            echo ""
            counter_lt_1000=$((counter_lt_1000 + 1))
            touch $abs_dir/to_RUN__lt_1000
        elif [ "$time_steps" -lt 5000 ]; then
            echo "Warning: time_steps < 5000"
            echo "Directory: $dir"
            echo "Time steps: $time_steps"
            echo ""
            counter_lt_5000=$((counter_lt_5000 + 1))
            touch $abs_dir/to_RUN__1000_to_5000
        elif [ "$time_steps" -lt 10000 ]; then
            echo "Warning: time_steps < 10000"
            echo "Directory: $dir"
            echo "Time steps: $time_steps"
            echo ""
            counter_lt_10000=$((counter_lt_10000 + 1))
            touch $abs_dir/to_RUN__5000_to_10000
        elif [ "$time_steps" -lt 20000 ]; then
            echo "Warning: time_steps < 20000"
            echo "Directory: $dir"
            echo "Time steps: $time_steps"
            echo ""
            counter_lt_20000=$((counter_lt_20000 + 1))
            touch $abs_dir/to_RUN__10000_to_20000
        elif [ -z "$time_steps" ]; then
            echo "Warning: time_steps not found"
            echo "Directory: $dir"
            echo ""
            counter_failed=$((counter_failed + 1))
            touch $abs_dir/to_RUN__failed
        else
            echo "Time steps: $time_steps"
            echo "Directory: $dir"
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
echo "Directories with 10000 < time_steps < 20000 (touch to_RUN_10000_to_20000): $counter_10000_to_20000"
echo "Directories with 5000 < time_steps < 10000 (touch to_RUN_5000_to_10000): $counter_5000_to_10000"
echo "Directories with 1000 < time_steps < 5000 (touch to_RUN_1000_to_5000): $counter_1000_to_5000"
echo "Directories with time_steps < 1000 (touch to_RUN_lt_1000): $counter_lt_1000"
echo ""
echo "Directories with time_steps not found (touch to_RUN_failed): $counter_failed"
echo "======================================================="
echo ""
