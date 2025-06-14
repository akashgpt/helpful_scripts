#!/bin/bash

# Usage: nohup bash submit_MULTI_RUN_VASP.sh > log.submit_MULTI_RUN_VASP 2>&1 &

to_RUN_keyword="to_RUN__"
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
        total_keyword_files=$(find . -type f -name "*$to_RUN_keyword*" | wc -l)
        echo "Total filenames with keyword '$to_RUN_keyword': $total_keyword_files"

        # skip if no files with the keyword "to_RUN_keyword" are found
        if [ "$total_keyword_files" -eq 0 ]; then
            echo "No files with keyword '$to_RUN_keyword' found. Skipping this directory."
            cd "$home_dir" || exit 1
            continue
        fi

        cp "$home_dir/MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN.sh" $abs_dir # copy the script to the current directory

        NODES_CHOSEN=$((4 * total_keyword_files)) # set NODES_CHOSEN to 4 times the number of files with the keyword "to_RUN_keyword"
        echo "NODES_CHOSEN: $NODES_CHOSEN"

        # replace __NODES_CHOSEN__ in the MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN.sh script
        sed -i "s/__NODES_CHOSEN__/$NODES_CHOSEN/" MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN.sh
        echo "Submitting MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN.sh script"

        sbatch MULTI_RUN_VASP__FRONTERA_Vfind_isobar_calc_to_RUN.sh

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