#!/usr/bin/env bash
set -euo pipefail

# Usage: nohup bash correct_SCALEE_value_inconsistencies.sh > log.correct_SCALEE_value_inconsistencies 2>&1 &

###############################################################################
# CONFIG
###############################################################################
# DIR_to_eval="SCALEE_6"
# TARGET_SCALEE=0.00035461   # the value every SCALEE_7* run should have
DIR_to_eval="SCALEE_7"
TARGET_SCALEE=0.00000108469   # the value every SCALEE_7* run should have
EPS=1e-9                      # tolerance for floating-point compare
DRY_RUN=1                     # 1 → just echo actions, 0 → actually rm -rf

###############################################################################
# MAIN
###############################################################################
echo "Scanning for $DIR_to_eval* directories …"
counter_bad=0
# find every dir whose basename starts with 'SCALEE_7'
find . -type d -name "${DIR_to_eval}*" \
        ! -path "*misc*" \
        ! -path "*isobar_calc*" \
        ! -path "*test*" \
        ! -path "*cont*" \
        ! -path "*Example*" \
        -print0 |
while IFS= read -r -d '' dir; do
    incar="$dir/INCAR"
    if [[ ! -f $incar ]]; then
        echo "WARNING: $incar not found → skipping."
        continue
    fi

    # Pull the first line containing 'SCALEE' and grab the 3rd field
    scalee=$(grep -m 1 -i 'SCALEE' "$incar" | awk '{print $3}')

    # Validate: must look like a float or int
    if [[ ! $scalee =~ ^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
        echo "ERROR: '$scalee' in $incar is not a valid number – skipping."
        continue
    fi

    # Compare numerically |scalee − TARGET| ≤ EPS
    same=$(awk -v s="$scalee" -v t="$TARGET_SCALEE" -v eps="$EPS" \
                'BEGIN{print (s-t<eps && t-s<eps)?1:0}')

    if (( same )); then
        # echo "OK   : $dir has correct SCALEE ($scalee)"
        continue
    else
        counter_bad=$((counter_bad + 1))
        echo "BAD  : $dir has SCALEE=$scalee ≠ $TARGET_SCALEE"
        if (( DRY_RUN )); then
            echo "DRY : would remove   $dir"
        else
            echo "RM  : removing $dir"
            rm -rf "$dir"
        fi
    fi
done

echo
if (( DRY_RUN )); then
    echo "Dry-run complete – nothing deleted.  Set DRY_RUN=0 to remove bad runs."
    echo "Summary: $counter_bad bad directories found."
else
    echo "Finished – inconsistent SCALEE_7* directories were deleted."
    echo "Summary: $counter_bad bad directories found and removed."
fi