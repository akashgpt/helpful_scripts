#!/bin/bash

# See comments at the bottom of the script for how to run this in parallel across all subdirectories

set -euo pipefail

d="${1:?pass subfolder path}"
cd "$d"

PROJECT_MLDP="/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev/TRAIN_MLMD_scripts/ANALYSIS/mldp"
MLDP_DIR="${LOCAL__ALCHEMY__main__MLDP:-${ALCHEMY__main__MLDP:-$PROJECT_MLDP}}"

if [ ! -d "$MLDP_DIR" ]; then
    MLDP_DIR="$PROJECT_MLDP"
fi

# repeat in each folder
#inside recal to create new deepmd files
rm -rf deepmd OUTCAR
python "${MLDP_DIR}/merge_out.py" -o OUTCAR
python "${MLDP_DIR}/extract_deepmd.py" -d deepmd -ttr 1000000

# ===========================================
# ===========================================
# ===========================================
# Run the following command to execute the above script in parallel across all subdirectories:
# find . -mindepth 1 -maxdepth 1 -type d -print0 | \
# xargs -0 -n 1 -P 112 bash -c '
#   d="$1"
#   ./create_deepmd_from_OUTCARs.sh "$d" > "$d/log.create_deepmd" 2>&1
# ' _
# ===========================================
# ===========================================
# ===========================================
