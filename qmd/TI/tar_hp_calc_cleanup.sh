#!/usr/bin/env bash
set -euo pipefail


# tar_hp_calc_cleanup.sh
# Usage: source $HELP_SCRIPTS_TI/tar_hp_calc_cleanup.sh > log.tar_hp_calc_cleanup 2>&1 &
#        nohup $HELP_SCRIPTS_TI/tar_hp_calc_cleanup.sh > log.tar_hp_calc_cleanup 2>&1 &
# This script finds all hp_calculations.tar* files, verifies their integrity,
# and removes any hp_calculations/ directories that are still present in their respective locations.

echo ""
echo "Starting hp_calculations cleanup at $(date)"
echo "Current directory: $(pwd)"
echo ""

counter=0

# 1) For every hp_calculations.tar* found under this tree:
find . -name "hp_calculations.tar*" -print0 \
| while IFS= read -r -d '' tar_file; do
    SCALEE_1_dir=$(dirname "$tar_file")
    echo "Checking archive: $tar_file"

    # 2) Verify archive integrity (list contents without extracting)
    if tar -tf "$tar_file" &>/dev/null; then
        echo "  ✔ Archive is valid."

        # 3) If a hp_calculations/ subdirectory still exists, remove it
        if [ -d "$SCALEE_1_dir/hp_calculations" ]; then
            echo "  Removing directory: $SCALEE_1_dir/hp_calculations"
            counter=$((counter + 1))
            echo "counter: $counter"
            rm -rf "$SCALEE_1_dir/hp_calculations"
        else
            echo "  No hp_calculations/ directory to remove in $SCALEE_1_dir"
        fi

    else
        echo "  ✖ Invalid archive: $tar_file"
        echo "    → Keeping both $tar_file and any existing hp_calculations/ for manual inspection."
    fi
    echo ""
done

echo ""
echo "Cleanup complete at $(date)"
# echo "Total hp_calculations/ directories removed: $counter"