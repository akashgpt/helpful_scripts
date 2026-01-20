#!/usr/bin/env bash
# set -euo pipefail
shopt -s nullglob

# Script to gather information from OUTCAR files in recal directories -- info relevant to plumed.
# It collects energy, volume, temperature, and pressure data, computes min/max values,
# converts energy to kJ/mol, and formats pressure in bars and GPa.
# Run this script in the parent ZONE directory. This goes into each configuration directory on its own.
#
# Usage: source $HELP_SCRIPTS_plmd/ZONE_recal_plumed_info.sh > log.ZONE_recal_plumed_info 2>&1 &
# or     nohup bash $HELP_SCRIPTS_plmd/ZONE_recal_plumed_info.sh > log.ZONE_recal_plumed_info 2>&1 &
#
# Author: Akash Gupta


ZONE_dir=$(pwd)

echo "Gathering plumed info from recal directories in $ZONE_dir @ $(date)"

# Prepare arrays
energy_vals=()
volume_vals=()
temp_vals=()
pressure_vals=()

MISSING_DATA_FLAG=false
MISSING_DATA_DIRS=()

# 1) Gather one value per directory from */pre/recal/*
for d in */pre/recal/*/; do
    dir=${d%/}
    dir_name=$(basename "$dir")
    # skip non-numeric folder names
    [[ $dir_name =~ ^[0-9]+$ ]] || continue
    echo "Processing directory: $dir"

    # last TOTEN in OUTCAR → energy in eV
    ev=$(grep "free  energy   TOTEN" "$dir/OUTCAR" \
            | tail -n1 \
            | awk '{print $5}')
    
    # first "volume" line → volume in Å³
    vv=$(grep -i "volume of cell" "$dir/OUTCAR" \
            | head -n1 \
            | awk '{print $5}')
    
    # first TEBEG line → temperature in K
    tempK=$(grep -i "TEBEG" "$dir/OUTCAR" | head -n1 | awk '{print $3}')

    # external pressure line → pressure in kBar
    pressurekBar=$(grep -i "external pressure" "$dir/OUTCAR" | head -n1 | awk '{print $4}')

    # --- Check for empty values ---
    if [[ -z "$ev" ]] || [[ -z "$vv" ]] || [[ -z "$tempK" ]] || [[ -z "$pressurekBar" ]]; then
        echo "⚠️  WARNING: One or more value extractions failed in: $dir/OUTCAR"
        MISSING_DATA_FLAG=true
        MISSING_DATA_DIRS+=("$dir")
        continue  # Skip adding this empty value to the arrays
    fi
    # ----------------------------------

    energy_vals+=("$ev")
    volume_vals+=("$vv")
    temp_vals+=("$tempK")
    pressure_vals+=("$pressurekBar")
done

# print warning if any directories had missing data
if [ "$MISSING_DATA_FLAG" = true ]; then
    echo "⚠️  WARNING: Some directories had missing data and were skipped:"
    for missing_dir in "${MISSING_DATA_DIRS[@]}"; do
        echo "    - $missing_dir"
    done
    echo ""
fi

# print total number of vals collected + min and max for a quick check -- for energy, volume, temp, pressure
echo "Collected ${#energy_vals[@]} energy values: min=$(printf '%s\n' "${energy_vals[@]}" | sort -n | head -n1), max=$(printf '%s\n' "${energy_vals[@]}" | sort -n | tail -n1)"
# print all energy values collected for debugging
# echo "Energy values collected: ${energy_vals[@]}"
echo "Collected ${#volume_vals[@]} volume values: min=$(printf '%s\n' "${volume_vals[@]}" | sort -n | head -n1), max=$(printf '%s\n' "${volume_vals[@]}" | sort -n | tail -n1)"
echo "Collected ${#temp_vals[@]} temperature values: min=$(printf '%s\n' "${temp_vals[@]}" | sort -n | head -n1), max=$(printf '%s\n' "${temp_vals[@]}" | sort -n | tail -n1)"
echo "Collected ${#pressure_vals[@]} pressure values: min=$(printf '%s\n' "${pressure_vals[@]}" | sort -n | head -n1), max=$(printf '%s\n' "${pressure_vals[@]}" | sort -n | tail -n1)"


# 2) Compute global min/max
energy_eV_min=$(printf '%s\n' "${energy_vals[@]}" | sort -n | head -n1)
energy_eV_max=$(printf '%s\n' "${energy_vals[@]}" | sort -n | tail -n1)
volume_min=$(printf '%s\n' "${volume_vals[@]}" | sort -n | head -n1)
volume_max=$(printf '%s\n' "${volume_vals[@]}" | sort -n | tail -n1)
temp_min=$(printf '%s\n' "${temp_vals[@]}" | sort -n | head -n1)
temp_max=$(printf '%s\n' "${temp_vals[@]}" | sort -n | tail -n1)
pressure_min=$(printf '%s\n' "${pressure_vals[@]}" | sort -n | head -n1)
pressure_max=$(printf '%s\n' "${pressure_vals[@]}" | sort -n | tail -n1)

# 3) Convert energy to kJ/mol (1 eV = 96.485 kJ/mol)
energy_kJ_min=$(awk -v e="$energy_eV_min" 'BEGIN {printf "%.6f", e * 96.485}')
energy_kJ_max=$(awk -v e="$energy_eV_max" 'BEGIN {printf "%.6f", e * 96.485}')

# 3.5) Change pressure from kBar to bars, and GPa
pressure_bars_min=$(awk -v p="$pressure_min" 'BEGIN {printf "%.6f", p * 1000}')
pressure_bars_max=$(awk -v p="$pressure_max" 'BEGIN {printf "%.6f", p * 1000}')
pressure_GPa_min=$(awk -v p="$pressure_min" 'BEGIN {printf "%.6f", p * 0.1}')
pressure_GPa_max=$(awk -v p="$pressure_max" 'BEGIN {printf "%.6f", p * 0.1}')

# # 3.9) Round off values to the nearest integer -- min to lowest, max to highest nearest integer
# # helper functions
# floor() { awk -v v="$1" 'BEGIN{printf "%d", int(v)}'; }
# ceil () { awk -v v="$1" 'BEGIN{printf "%d", (v==int(v)?int(v):int(v)+1)}'; }

# energy_eV_min=$(floor  "$energy_eV_min")
# energy_eV_max=$(ceil   "$energy_eV_max")
# energy_kJ_min=$(floor  "$energy_kJ_min")
# energy_kJ_max=$(ceil   "$energy_kJ_max")
# volume_min=$(floor     "$volume_min")
# volume_max=$(ceil      "$volume_max")
# temp_min=$(floor       "$temp_min")
# temp_max=$(ceil        "$temp_max")
# pressure_bars_min=$(floor  "$pressure_bars_min")
# pressure_bars_max=$(ceil   "$pressure_bars_max")
# pressure_min=$(floor      "$pressure_min")
# pressure_max=$(ceil       "$pressure_max")
# pressure_GPa_min=$(floor  "$pressure_GPa_min")
# pressure_GPa_max=$(ceil   "$pressure_GPa_max")


# 4) Print or write out
cat > "plumed.info" <<EOF
Range of properties across all the current ZONE's recal (single-point calculation) directories:

    ZONE directory:             ${ZONE_dir}
    
    Number of dirs processed:   ${#energy_vals[@]}
    ... with missing data:      ${#MISSING_DATA_DIRS[@]}

    Energy/TOTEN (eV):          ${energy_eV_min} to ${energy_eV_max}
    Energy/TOTEN (kJ/mol):      ${energy_kJ_min} to ${energy_kJ_max}

    Volume (Å³):                ${volume_min} to ${volume_max} 

    Temperature (K):            ${temp_min} to ${temp_max}

    External pressure (bars):   ${pressure_bars_min} to ${pressure_bars_max}
    External pressure (kbar):   ${pressure_min} to ${pressure_max}
    External pressure (GPa):    ${pressure_GPa_min} to ${pressure_GPa_max}

EOF

echo "plumed.info file created with recalculation stats in $ZONE_dir at $(date)!"
