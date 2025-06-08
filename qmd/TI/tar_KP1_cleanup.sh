#!/usr/bin/env bash
set -euo pipefail


# tar_KP1_cleanup.sh
# Usage: source $HELP_SCRIPTS_TI/tar_KP1_cleanup.sh > log.tar_KP1_cleanup 2>&1 &
#        nohup $HELP_SCRIPTS_TI/tar_KP1_cleanup.sh > log.tar_KP1_cleanup 2>&1 &
# This script finds all KP1.tar* files, verifies their integrity,
# and removes any KP1/ directories that are still present in their respective locations.

echo ""
echo "Starting KP1 cleanup at $(date)"
echo "Current directory: $(pwd)"
echo ""

counter=0

# 1) For every KP1.tar* found under this tree:
find . -name "KP1.tar*" -print0 \
| while IFS= read -r -d '' tar_file; do
    V_est_dir=$(dirname "$tar_file")
    echo "Checking archive: $tar_file"

    # 2) Verify archive integrity (list contents without extracting)
    if tar -tf "$tar_file" &>/dev/null; then
        echo "  ✔ Archive is valid."

        # 3) If a KP1/ subdirectory still exists, remove it
        if [ -d "$V_est_dir/KP1" ]; then
            echo "  Removing directory: $V_est_dir/KP1"
            counter=$((counter + 1))
            echo "counter: $counter"
            rm -rf "$V_est_dir/KP1"
        else
            echo "  No KP1/ directory to remove in $V_est_dir"
        fi

    else
        echo "  ✖ Invalid archive: $tar_file"
        echo "    → Keeping both $tar_file and any existing KP1/ for manual inspection."
    fi
    echo ""
done

echo ""
echo "Cleanup complete at $(date)"
# echo "Total KP1/ directories removed: $counter"