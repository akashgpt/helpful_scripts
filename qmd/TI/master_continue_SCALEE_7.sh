#!/bin/bash

# Usage: nohup bash master_continue_SCALEE_7.sh > log.master_continue_SCALEE_7 2>&1 &

# find all SCALEE_7 directories that do not have a "misc" or "isobar_calc" on path, and time_run=$(grep time analysis/peavg.out | awk '{print $5}')

set -euo pipefail





# DIR_to_find="SCALEE_7"
DIR_to_find="SCALEE_6"
tag_DRY_RUN=0 # Set to 1 for dry run, 0 for actual run



HOME_DIR="$PWD"




echo "Processing all '$DIR_to_find' directories (excluding misc, isobar_calc, test, cont, Example)..."
echo

counter=0

find "$HOME_DIR" -type d -name "$DIR_to_find" \
        ! -path "*misc*" \
        ! -path "*isobar_calc*" \
        ! -path "*test*" \
        ! -path "*cont*" \
        ! -path "*Example*" \
    -print0 |
while IFS= read -r -d '' dir; do
    # echo "Directory: $dir"
    (
        cd "$dir" || { echo "  → Cannot cd into $dir"; exit 0; }

        if [[ ! -f analysis/peavg.out ]]; then
            echo "  → Missing analysis/peavg.out, skipping. Directory: $dir"
            exit 0
        fi

        time_run=$(awk '/time/ {print $5; exit}' analysis/peavg.out)
        if [[ -z $time_run || $time_run != ?([0-9])* ]]; then
            echo "  → Invalid time_run ('$time_run'), skipping. Directory: $dir"
            exit 0
        fi

        echo "  → time_run = $time_run"
        if (( time_run < 10000 )); then
            echo "Directory: $dir"
            cd ..
            echo "$PWD"
            echo "  → Restarting run via continue_run__SCALEE.sh..."
            if (( tag_DRY_RUN )); then
                echo "  → [DRY RUN] nohup bash \"$HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh\" \
                    \"$DIR_to_find\" 5 5 \
                    > \"log.continue_run__$DIR_to_find\" 2>&1 &"
            else
                # for SCALEE_6 (or 7)
                nohup bash "$HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh" \
                    "$DIR_to_find" 0 24 \
                    > "log.continue_run__$DIR_to_find" 2>&1 &

                # for SCALEE_7
                # nohup bash "$HELP_SCRIPTS_vasp/RUN_VASP/continue_run__SCALEE.sh" \
                #     "$DIR_to_find" 5 5 \
                #     > "log.continue_run__$DIR_to_find" 2>&1 &
            fi
        else
            echo "  → Already ran long enough, skipping"
        fi
    )
    echo ""
    # echo "test: $PWD"
    # echo ""
done

echo
echo "All directories processed."