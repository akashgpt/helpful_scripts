#!/usr/bin/env bash
# set -euo pipefail

# Script to gather information from OUTCAR files in recal directories -- info relevant to plumed.
# It collects energy, volume, temperature, and pressure data, computes min/max values,
# converts energy to kJ/mol, and formats pressure in bars and GPa.
# Run this script in the parent directory containing recal directories named with numeric values.
#
# Usage: source $HELP_SCRIPTS_plmd/recal_plumed_info.sh > log.recal_plumed_info 2>&1 &
# or     nohup bash $HELP_SCRIPTS_plmd/recal_plumed_info.sh > log.recal_plumed_info 2>&1 &

# Prepare arrays
energy_vals=()
volume_vals=()
temp_vals=()
pressure_vals=()

# 1) Gather one value per directory
for d in */; do
    dir=${d%/}
    # skip non-numeric folder names
    [[ $dir =~ ^[0-9]+$ ]] || continue

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

    energy_vals+=("$ev")
    volume_vals+=("$vv")
    temp_vals+=("$tempK")
    pressure_vals+=("$pressurekBar")
done

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


# 4) Print or write out
cat > "plumed.info" <<EOF
Across all directories:

    Energy range: 
    ${energy_eV_min} to ${energy_eV_max} eV
    ${energy_kJ_min} to ${energy_kJ_max} kJ/mol

    Volume range:
    ${volume_min} to ${volume_max} Å³

    Temperature range:
    ${temp_min} to ${temp_max} K

    External pressure range:
    ${pressure_bars_min} to ${pressure_bars_max} bars
    ${pressure_min} to ${pressure_max} kbar
    ${pressure_GPa_min} to ${pressure_GPa_max} GPa

EOF