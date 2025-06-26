#!/bin/bash

# Usage: nohup bash submit_MULTI_RUN_VASP__to_NOT_RUN_key.sh > log.submit_MULTI_RUN_VASP__to_NOT_RUN_key 2>&1 &

# to_RUN_keyword="to_RUN__"  # the directories that have this keyword in a filename will go into RUN_DIRS
to_NOT_RUN_keyword="done_SCALEE_0"  # the directories that have this keyword in a filename, right outside, will NOT go into RUN_DIRS
home_dir=$(pwd)
echo ""
echo "Home directory: $home_dir"
echo "Current time: $(date)"
echo ""
echo "Submitting MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN.sh scripts in all isobar_calc directories"
echo ""

# find "isobar_calc" folders/sub-folders and run MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN.sh script in each
find . -type d -name "isobar_calc" | while read dir; do
    # echo "Running MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN.sh in $dir"
    (
        cd "$dir" || exit 1
        abs_dir=$(pwd)
        echo ""
        echo "Current directory: $dir"

        # find total number of filenames with the keyword "to_RUN_keyword"
        # total_keyword_files=$(find . -type f -name "*$to_RUN_keyword*" | wc -l)
        total_keyword_files=$(find . -type f -name "*$to_NOT_RUN_keyword*" | wc -l)
        # echo "Total filenames with keyword '$to_RUN_keyword': $total_keyword_files"
        echo "Total filenames with keyword '$to_NOT_RUN_keyword': $total_keyword_files"

        # skip if no files with the keyword "to_RUN_keyword" are found
        if [ "$total_keyword_files" -eq 4 ]; then
            echo "4 files with keyword '$to_NOT_RUN_keyword' found. Skipping this directory."
            cd "$home_dir" || exit 1
            continue
        fi

        # # temporary -- if $total_keyword_files -eq 0, then skip this directory, as already running
        # if [ "$total_keyword_files" -eq 0 ]; then
        #     echo "No files with keyword '$to_NOT_RUN_keyword' found. Skipping this directory as these are already running."
        #     cd "$home_dir" || exit 1
        #     continue
        # fi

        cp "$home_dir/MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN__NOT.sh" $abs_dir # copy the script to the current directory

        # number of sims to run = 4 - total_keyword_files
        number_of_sims_to_run=$((4 - total_keyword_files))
        NODES_CHOSEN=$((4 * number_of_sims_to_run)) # set NODES_CHOSEN to 4 times the number of sims to run
        echo "NODES_CHOSEN: $NODES_CHOSEN"

        # replace __NODES_CHOSEN__ in the MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN__NOT.sh script
        sed -i "s/__NODES_CHOSEN__/$NODES_CHOSEN/" MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN__NOT.sh
        echo "Submitting MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN__NOT.sh script"

        sbatch MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN__NOT.sh

        cd "$home_dir" || exit 1
    )
done

echo ""
echo ""
echo "======================================================="
echo "All MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN.sh scripts submitted."
echo "Current time: $(date)"
echo "Check individual log files in each isobar_calc directory for details."
echo "======================================================="
echo ""