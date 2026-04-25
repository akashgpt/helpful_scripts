#!/bin/bash

#############################################################
# Summary: This script is used to analyze the output of VASP simulations.
# It extracts relevant data from the OUTCAR file, performs calculations, 
# and generates plots for analysis.
#
# Usage: source data_4_analysis.sh
#
# Author: Akash Gupta
#############################################################

parent_dir=$(pwd)
parent_dir_name=$(basename "$parent_dir")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
outcar_path="${1:-OUTCAR}"
data_4_analysis_python="${DATA_4_ANALYSIS_PYTHON:-${HELPFUL_SCRIPTS_PYTHON:-python}}"
band_summary_script="${script_dir}/extract_band_occupations.py"
snapshot_plot_script="${script_dir}/plot_vasp_current_snapshot.py"
standard_outcar_is_non_md=0
last_snapshot_path=""

export DATA_4_ANALYSIS_PYTHON="$data_4_analysis_python"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

resolve_helper_script() {
    local script_name="$1"
    local candidate_path resolved_path

    candidate_path="${script_dir}/${script_name}"
    if [[ -f "$candidate_path" ]]; then
        printf '%s\n' "$candidate_path"
        return 0
    fi

    resolved_path=$(command -v "$script_name" 2>/dev/null || true)
    if [[ -n "$resolved_path" && -f "$resolved_path" ]]; then
        printf '%s\n' "$resolved_path"
        return 0
    fi

    return 1
}

validate_standard_md_outcar() {
	local pressure_count toten_count internal_energy_count temperature_count mlff_count

	standard_outcar_is_non_md=0

	mlff_count=$(grep -c "free  energy ML TOTEN\|MLFF:" "$outcar_path" || true)

	if [[ "$mlff_count" -gt 0 ]]; then
		echo "Error: $outcar_path looks like an MLFF OUTCAR, not a standard MD OUTCAR."
		echo "Use the wrapper data_4_analysis.sh or the MLFF-specific analysis script instead of the standard script."
		echo "Found MLFF markers: mlff_count=$mlff_count"
		return 1
	fi

	pressure_count=$(grep -c "total pressure" "$outcar_path" || true)
	toten_count=$(grep -c "free  energy   TOTEN" "$outcar_path" || true)
	internal_energy_count=$(grep -c "energy  without entropy" "$outcar_path" || true)
	temperature_count=$(grep -c "(temperature" "$outcar_path" || true)

	if [[ "$pressure_count" -eq 0 || "$toten_count" -eq 0 || "$internal_energy_count" -eq 0 || "$temperature_count" -eq 0 ]]; then
		standard_outcar_is_non_md=1
		echo "Notice: $outcar_path does not look like a standard MD OUTCAR."
		echo "The MD workflow expects non-zero counts for 'total pressure', 'free  energy   TOTEN', 'energy  without entropy', and '(temperature)'."
		echo "Found: total_pressure=$pressure_count, TOTEN=$toten_count, internal_energy=$internal_energy_count, temperature=$temperature_count"
		return 1
	fi

	return 0
}

select_snapshot_structure() {
	local structure_path

	for structure_path in CONTCAR POSCAR; do
		if [[ -f "$structure_path" && -s "$structure_path" ]]; then
			printf '%s\n' "$structure_path"
			return 0
		fi
	done

	return 1
}

plot_current_snapshot() {
	local structure_path

	last_snapshot_path=""
	if [[ ! -f "$snapshot_plot_script" ]]; then
		echo "Warning: snapshot helper missing; skipping analysis/current_snapshot.png"
		return 1
	fi

	if ! structure_path=$(select_snapshot_structure); then
		echo "Warning: neither CONTCAR nor POSCAR is available; skipping analysis/current_snapshot.png"
		return 1
	fi

	echo "Plotting current $structure_path snapshot to analysis/current_snapshot.png."
	if "$data_4_analysis_python" "$snapshot_plot_script" \
		--structure "$structure_path" \
		--outcar "$outcar_path" \
		--output analysis/current_snapshot.png \
		--title "Current $structure_path snapshot: $parent_dir_name"; then
		last_snapshot_path="analysis/current_snapshot.png"
		return 0
	fi

	echo "Warning: failed to create analysis/current_snapshot.png"
	return 1
}

extract_band_summary() {
	if [[ -f "$band_summary_script" ]]; then
		if ! grep -aq "band No.  band energies     occupation" "$outcar_path"; then
			echo "Warning: no band occupation table found in OUTCAR; skipping analysis/band_occupations_summary.out"
			return 1
		fi

		echo "Extracting occupied-band summary from OUTCAR."
		if "$data_4_analysis_python" "$band_summary_script" \
			--outcar "$outcar_path" \
			--output analysis/band_occupations_summary.out \
			--selection second_last; then
			if grep -q '^flag_no_nonzero_occupied_bands=yes$' analysis/band_occupations_summary.out; then
				echo "Warning: no non-zero occupied bands were found in the selected band table."
			fi
			return 0
		fi
		echo "Warning: failed to create analysis/band_occupations_summary.out"
		return 1
	fi

	echo "Warning: band summary helper not found: $band_summary_script"
	return 1
}

get_last_outcar_field() {
	local pattern="$1"
	local field_number="$2"

	grep -a "$pattern" "$outcar_path" | tail -n 1 | awk -v field_number="$field_number" '{print $field_number}'
}

get_incar_tag_value() {
	local key="$1"

	if [[ ! -f INCAR ]]; then
		return 0
	fi

	awk -v key="$key" '
		BEGIN {
			key = toupper(key)
		}
		{
			line = $0
			sub(/[!#].*$/, "", line)
			gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
			if (line == "") {
				next
			}

			if (index(line, "=") > 0) {
				tag = substr(line, 1, index(line, "=") - 1)
				value = substr(line, index(line, "=") + 1)
			} else {
				tag = $1
				value = line
				sub(/^[^[:space:]]+[[:space:]]*/, "", value)
			}

			gsub(/[[:space:]]+/, "", tag)
			tag = toupper(tag)
			if (tag != key) {
				next
			}

			gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
			print value
			exit
		}
	' INCAR
}

get_potcar_lexch_value() {
	if [[ ! -f POTCAR ]]; then
		return 0
	fi

	awk -F '=' '
		/LEXCH/ {
			value = $2
			gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
			print value
			exit
		}
	' POTCAR
}

get_exchange_correlation_label() {
	local gga metagga potcar_lexch

	metagga=$(get_incar_tag_value "METAGGA")
	if [[ -n "$metagga" ]]; then
		printf 'METAGGA=%s\n' "$metagga"
		return 0
	fi

	gga=$(get_incar_tag_value "GGA")
	if [[ -n "$gga" ]]; then
		printf 'GGA=%s\n' "$gga"
		return 0
	fi

	potcar_lexch=$(get_potcar_lexch_value)
	if [[ -n "$potcar_lexch" ]]; then
		printf 'POTCAR_LEXCH=%s\n' "$potcar_lexch"
		return 0
	fi

	printf 'default_from_VASP_or_POTCAR\n'
}

get_potcar_species_summary() {
	if [[ ! -f POTCAR ]]; then
		return 0
	fi

	grep -a "TITEL" POTCAR | awk '{print $4}' | paste -sd "," -
}

get_nions_from_outcar() {
	grep -a -m 1 "NIONS" "$outcar_path" | awk '
		{
			for (i = 1; i <= NF; i++) {
				if ($i == "NIONS") {
					for (j = i + 1; j <= NF; j++) {
						if ($j ~ /^[0-9]+$/) {
							print $j
							exit
						}
					}
				}
			}
		}
	'
}

write_summary_key_value() {
	local key="$1"
	local value="${2:-}"

	if [[ -z "$value" ]]; then
		value="NA"
	fi
	printf '%s=%s\n' "$key" "$value" >> analysis/single_point_summary.out
}

print_value_or_na() {
	local value="${1:-}"

	if [[ -n "$value" ]]; then
		printf '%s\n' "$value"
	else
		printf 'NA\n'
	fi
}

write_single_point_evo_files() {
	local value

	value=$(get_last_outcar_field "free  energy   TOTEN" 5)
	if [[ -n "$value" ]]; then
		printf '%s\n' "$value" > analysis/evo_TOTEN.dat
		cp analysis/evo_TOTEN.dat analysis/evo_free_energy.dat
	fi

	value=$(get_last_outcar_field "energy  without entropy" 4)
	if [[ -n "$value" ]]; then
		printf '%s\n' "$value" > analysis/evo_internal_energy.dat
	fi

	value=$(get_last_outcar_field "ETOTAL" 5)
	if [[ -n "$value" ]]; then
		printf '%s\n' "$value" > analysis/evo_total_energy.dat
	fi

	value=$(get_last_outcar_field "volume of cell :" 5)
	if [[ -n "$value" ]]; then
		printf '%s\n' "$value" > analysis/evo_cell_volume.dat
	fi

	value=$(get_last_outcar_field "total pressure" 4)
	if [[ -n "$value" ]]; then
		printf '%s\n' "$value" > analysis/evo_total_pressure.dat
	fi

	value=$(get_last_outcar_field "external" 4)
	if [[ -n "$value" ]]; then
		printf '%s\n' "$value" > analysis/evo_external_pressure.dat
	fi
}

write_single_point_summary() {
	local electronic_converged elapsed_time_seconds potcar_species vasp_completed

	if grep -aq "aborting loop because EDIFF is reached\|reached required accuracy" "$outcar_path"; then
		electronic_converged="yes"
	else
		electronic_converged="unknown"
	fi

	if grep -aq "General timing and accounting" "$outcar_path"; then
		vasp_completed="yes"
	else
		vasp_completed="no"
	fi

	elapsed_time_seconds=$(grep -a "Elapsed time" "$outcar_path" | tail -n 1 | awk '{print $NF}')
	potcar_species=$(get_potcar_species_summary)

	: > analysis/single_point_summary.out
	write_summary_key_value "run_directory" "$parent_dir"
	write_summary_key_value "outcar_path" "$outcar_path"
	write_summary_key_value "analysis_mode" "single_point_or_non_md"
	write_summary_key_value "snapshot_png" "$last_snapshot_path"
	write_summary_key_value "TOTEN_eV" "$(get_last_outcar_field "free  energy   TOTEN" 5)"
	write_summary_key_value "energy_without_entropy_eV" "$(get_last_outcar_field "energy  without entropy" 4)"
	write_summary_key_value "ETOTAL_eV" "$(get_last_outcar_field "ETOTAL" 5)"
	write_summary_key_value "cell_volume_A3" "$(get_last_outcar_field "volume of cell :" 5)"
	write_summary_key_value "total_pressure_kB" "$(get_last_outcar_field "total pressure" 4)"
	write_summary_key_value "external_pressure_kB" "$(get_last_outcar_field "external" 4)"
	write_summary_key_value "NIONS" "$(get_nions_from_outcar)"
	write_summary_key_value "ENCUT_eV" "$(get_incar_tag_value "ENCUT")"
	write_summary_key_value "exchange_correlation" "$(get_exchange_correlation_label)"
	write_summary_key_value "POTCAR_species" "$potcar_species"
	write_summary_key_value "electronic_converged" "$electronic_converged"
	write_summary_key_value "vasp_completed" "$vasp_completed"
	write_summary_key_value "elapsed_time_seconds" "$elapsed_time_seconds"
}

run_non_md_analysis_fallback() {
	echo "Running lightweight single-point/non-MD analysis for $parent_dir_name."
	echo "Skipping MD-only averaging and evolution plotting."
	mkdir -p analysis

	plot_current_snapshot || true
	write_single_point_evo_files
	write_single_point_summary
	extract_band_summary || true

	if [[ -f analysis/band_occupations_summary.out ]]; then
		printf '\n# band_occupations_summary.out\n' >> analysis/single_point_summary.out
		cat analysis/band_occupations_summary.out >> analysis/single_point_summary.out
	fi

	echo "Wrote analysis/single_point_summary.out"
	echo "Done with lightweight single-point/non-MD analysis for $parent_dir_name."
	return 0
}

resolved_band_summary_script=$(command -v extract_band_occupations.py 2>/dev/null || true)
if [[ -n "$resolved_band_summary_script" && -f "$resolved_band_summary_script" ]]; then
    band_summary_script="$resolved_band_summary_script"
fi
if [[ ! -f "$band_summary_script" ]]; then
	band_summary_script="/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/qmd/vasp/extract_band_occupations.py"
fi

resolved_snapshot_plot_script=$(command -v plot_vasp_current_snapshot.py 2>/dev/null || true)
if [[ -n "$resolved_snapshot_plot_script" && -f "$resolved_snapshot_plot_script" ]]; then
    snapshot_plot_script="$resolved_snapshot_plot_script"
fi
if [[ ! -f "$snapshot_plot_script" ]]; then
	snapshot_plot_script="/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/qmd/vasp/plot_vasp_current_snapshot.py"
fi

if [[ ! -f "$outcar_path" ]]; then
    echo "Error: OUTCAR not found: $outcar_path"
    return 1 2>/dev/null || exit 1
fi

if ! validate_standard_md_outcar; then
	if [[ "$standard_outcar_is_non_md" -eq 1 ]]; then
		run_non_md_analysis_fallback
		return_code=$?
		return "$return_code" 2>/dev/null || exit "$return_code"
	fi
	return 1 2>/dev/null || exit 1
fi

if ! peavg_script=$(resolve_helper_script "peavg.sh"); then
	echo "Error: peavg.sh not found beside this script or on PATH."
	return 1 2>/dev/null || exit 1
fi


# module purge
MSD_python_file="${LOCAL_HELP_SCRIPTS_vasp}/msd_calc_v3.py"
ENV_for_MSD="module load anaconda3/2024.6; conda activate mda_env"
echo $ENV_for_MSD > setting_env.sh #for mda analysis and the "bc" command required for peavg.sh




echo "################################"
echo "Running data_4_analysis.sh for $parent_dir_name"
# echo "Runtime: $runtime seconds"
echo "################################"

# figure out how long the scripts takes to run
# start=$(date +%s)  # Start time in seconds





echo "Updating data for 'analysis/' ..."

mkdir -p analysis

plot_current_snapshot || true


# Count the number of lines matching the two patterns
scaled_count=$(grep "SCALED FREE ENERGIE" "$outcar_path" | wc -l)
free_count=$(grep "free  energy" "$outcar_path" | wc -l)
half_free=$(echo "0.5 * $free_count" | bc)
# make half_free an integer
half_free=${half_free%.*}

# Define TI_mode: 1 if scaled_count equals half_free, 0 otherwise.
if [ "$scaled_count" -eq "$half_free" ]; then
    TI_mode=1
    # echo "TI_mode switched on."
else
    TI_mode=0
fi

echo "TI_mode is: $TI_mode" #; scaled_count is: $scaled_count, free_count is: $free_count, half_free is: $half_free"




grep "total pressure" "$outcar_path" | awk '{print $4}' > analysis/evo_total_pressure.dat
grep external "$outcar_path" | awk '{print $4}' > analysis/evo_external_pressure.dat
grep "kinetic pressure" "$outcar_path" | awk '{print $7}' > analysis/evo_kinetic_pressure.dat
grep "Pullay stress" "$outcar_path" | awk '{print $9}' > analysis/evo_pullay_stress.dat
grep -a "volume of cell :" "$outcar_path" | awk '{print $5}' > analysis/evo_cell_volume.dat
sed -i '1,2d' analysis/evo_cell_volume.dat

# grep "free  energy" OUTCAR | awk '{print $5}' > analysis/evo_free_energy.dat
grep ETOTAL "$outcar_path" | awk '{print $5}' > analysis/evo_total_energy.dat
grep "free  energy   TOTEN" "$outcar_path" | awk '{print $5}' > analysis/evo_TOTEN.dat
grep "energy  without entropy" "$outcar_path" | awk '{print $4}' > analysis/evo_internal_energy.dat

# grep "mean temperature" OUTCAR | awk '{print $5}' > analysis/evo_mean_temp.dat
grep "(temperature" "$outcar_path" | sed -E 's/.*temperature[[:space:]]*([0-9]+\.[0-9]+).*/\1/' > analysis/evo_mean_temp.dat

# if TI_mode is 1, then
if [ "$TI_mode" -eq 1 ]; then
    echo "Given the TI_mode, only keeping the non-scaled energy values from OUTCAR (i.e., not the <SCALED FREE ENERGIE ...> values)."
    awk 'NR%2==1' analysis/evo_internal_energy.dat > analysis/temp
    mv analysis/temp analysis/evo_internal_energy.dat
    # awk 'NR%2==0' analysis/evo_free_energy.dat > analysis/temp
    # mv analysis/temp analysis/evo_free_energy.dat
    awk 'NR%2==1' analysis/evo_TOTEN.dat > analysis/temp
    mv analysis/temp analysis/evo_TOTEN.dat
fi

cp analysis/evo_TOTEN.dat analysis/evo_free_energy.dat # for backward compatibility

echo "Running peavg.sh using: $peavg_script"
bash "$peavg_script" "$outcar_path"

extract_band_summary || true




# append a line to analysis/peavg_numbers.out with $parent_dir
echo "$parent_dir" > analysis/peavg_summary.out
# second and fourth line from peavg_numbers.out to analysis/peavg_summary.out
sed -n '1p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #TEMP
sed -n '4p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #NIONS
sed -n '24p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #Pressure
sed -n '25p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #Pressure error
sed -n '10p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #Internal energy
sed -n '11p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #Internal energy error
print_value_or_na "$(get_incar_tag_value "ENCUT")" >> analysis/peavg_summary.out #ENCUT
get_exchange_correlation_label >> analysis/peavg_summary.out #XC
grep "TITEL" $parent_dir/POTCAR | awk '{print $4}' >> analysis/peavg_summary.out #POTCAR 
grep "free  energy   TOTEN" "$outcar_path" | tail -n 1 | awk '{print $5}' >> analysis/peavg_summary.out #last free energy TOTEN value -- the only value for single point calculations
sed -n '16p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #E-TS_el
sed -n '17p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #E-TS_el error
sed -n '18p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #S_el
sed -n '19p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #S_el error
sed -n '7p' $parent_dir/analysis/peavg_numbers.out >> analysis/peavg_summary.out #Volume in cm^3/mol-atom
if [[ -f analysis/band_occupations_summary.out ]]; then
	cat analysis/band_occupations_summary.out >> analysis/peavg_summary.out
fi





######################################
echo "Plotting some relevant data."
# call python to create a plot of the following in 1 figure, 4 X 1 panels
# 1. data in evo_total_pressure vs time-step
# 2. data in evo_total_energy vs time-step
# 3. data in evo_cell_volume vs time-step
# 4. data in evo_mean_temp vs time-step
"$data_4_analysis_python" << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
import os


def load_species_metadata(run_dir: str) -> str:
    """Build a compact composition summary for the plot title."""
    for structure_name in ("CONTCAR", "POSCAR"):
        structure_path = os.path.join(run_dir, structure_name)
        if not os.path.exists(structure_path):
            continue
        with open(structure_path, encoding="utf-8", errors="ignore") as handle:
            lines = [line.strip() for line in handle.readlines()]
        if len(lines) < 7:
            continue
        candidate_species = lines[5].split()
        candidate_counts = lines[6].split()
        if not candidate_species or not candidate_counts:
            continue
        try:
            counts = [int(value) for value in candidate_counts]
        except ValueError:
            continue
        if len(candidate_species) != len(counts):
            continue
        total_atoms = sum(counts)
        species_summary = " ".join(f"{count}{species}" for species, count in zip(candidate_species, counts))
        return f"{species_summary} ({total_atoms} atoms; ratio: {{ratio:g}})"
    return "({ratio:g})"

current_dir = os.getcwd()
analysis_dir = os.path.join(current_dir, "analysis")
composition_template = load_species_metadata(current_dir)

axis_low_limit = 0.90
axis_high_limit = 1.10

# Load data from each file.
total_pressure = np.loadtxt("analysis/evo_total_pressure.dat")
external_pressure = np.loadtxt("analysis/evo_external_pressure.dat")
total_energy   = np.loadtxt("analysis/evo_total_energy.dat")
TOTEN        = np.loadtxt("analysis/evo_TOTEN.dat")
internal_energy = np.loadtxt("analysis/evo_internal_energy.dat")
free_energy   = np.loadtxt("analysis/evo_free_energy.dat")
cell_volume    = np.loadtxt("analysis/evo_cell_volume.dat")
mean_temp      = np.loadtxt("analysis/evo_mean_temp.dat")

# check if a ratio file exists
if os.path.exists("ratio"):
    ratio = np.loadtxt("ratio")
    print(f"Ratio file found: {ratio}")
else:
    print("No ratio file found. Using default value of 4")
    ratio = 4

# choose the last (1 - (1/ratio)) of the data
stat_total_pressure = total_pressure[int(len(total_pressure) * (1 - (1 / ratio))):]
stat_external_pressure = external_pressure[int(len(external_pressure) * (1 - (1 / ratio))):]
stat_total_energy = total_energy[int(len(total_energy) * (1 - (1 / ratio))):]
stat_TOTEN = TOTEN[int(len(TOTEN) * (1 - (1 / ratio))):]
stat_internal_energy = internal_energy[int(len(internal_energy) * (1 - (1 / ratio))):]
stat_free_energy = free_energy[int(len(free_energy) * (1 - (1 / ratio))):]
stat_cell_volume = cell_volume[int(len(cell_volume) * (1 - (1 / ratio))):]
stat_mean_temp = mean_temp[int(len(mean_temp) * (1 - (1 / ratio))):]


# make internal_energy and total_energy the same length as each other and get ride of the extra lines whichever has it
if len(internal_energy) > len(total_energy):
    internal_energy = internal_energy[:len(total_energy)]
elif len(total_energy) > len(internal_energy):
    total_energy = total_energy[:len(internal_energy)]

# same with free_energy and total_energy
if len(free_energy) > len(total_energy):
    free_energy = free_energy[:len(total_energy)]
elif len(total_energy) > len(free_energy):
    total_energy = total_energy[:len(free_energy)]

# TOTEN and total_energy
if len(TOTEN) > len(total_energy):
    TOTEN = TOTEN[:len(total_energy)]
elif len(total_energy) > len(TOTEN):
    total_energy = total_energy[:len(TOTEN)]

# pressure kBar to GPa
total_pressure = total_pressure * 0.1
external_pressure = external_pressure * 0.1
stat_total_pressure = stat_total_pressure * 0.1
stat_external_pressure = stat_external_pressure * 0.1

# Create a time-step array based on the number of data points.
time_steps_pressure = np.arange(1, len(total_pressure) + 1)
time_steps_external_pressure = np.arange(1, len(external_pressure) + 1)
time_steps_energy = np.arange(1, len(total_energy) + 1)
time_steps_volume = np.arange(1, len(cell_volume) + 1)
time_steps_temp = np.arange(1, len(mean_temp) + 1)

# Create a figure with 4 vertical subplots.
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 12))
fig.subplots_adjust(hspace=0.5)

# Panel 1: evo_total_pressure vs time-step
axs[0].plot(time_steps_pressure, total_pressure, 'b-', alpha=0.5)
axs[0].axhline(np.mean(stat_total_pressure), color='b', linestyle='--', label=f'Mean: {np.mean(stat_total_pressure):.2f} +/- {np.std(stat_total_pressure):.2f} GPa')
axs[0].set_ylabel('Total Pressure (GPa)')
leg = axs[0].legend(loc='upper left')
for text in leg.get_texts():
    text.set_color('b')
axs[0].grid()
axs[0].set_ylim(np.min(total_pressure)*axis_low_limit, np.max(total_pressure)*axis_high_limit)
# twinx axis for external pressure
ax1 = axs[0].twinx()
ax1.plot(time_steps_external_pressure, external_pressure, 'r-', alpha=0.5)
ax1.axhline(np.mean(stat_external_pressure), color='r', linestyle='--', label=f'Mean: {np.mean(stat_external_pressure):.2f} +/- {np.std(stat_external_pressure):.2f} GPa')
ax1.set_ylabel('External Pressure (GPa)')
# color the axis red
# ax1.tick_params(axis='y', labelcolor='r')
leg = ax1.legend(loc='upper right')
for text in leg.get_texts():
    text.set_color('r')
ax1.set_ylim(np.min(external_pressure)*axis_low_limit, np.max(external_pressure)*axis_high_limit)


# Panel 2: evo_total_energy vs time-step
axs[1].plot(time_steps_energy, total_energy, 'g-', alpha=0.5)
axs[1].axhline(np.mean(stat_total_energy), color='g', linestyle='--', label=f'Mean: {np.mean(stat_total_energy):.2f} +/- {np.std(stat_total_energy):.2f} eV')
# axs[1].plot(time_steps_energy, TOTEN, 'b-', alpha=0.5)
# axs[1].axhline(np.mean(stat_TOTEN), color='b', linestyle='--', label=f'Mean: {np.mean(stat_TOTEN):.2f} +/- {np.std(stat_TOTEN):.2f} eV')
# axs[1].plot(time_steps_energy, free_energy, 'm:', label='Free Energy')
axs[1].set_ylabel('Total Energy (ETOTAL; eV)')
axs[1].grid();
leg = axs[1].legend(loc='upper left');
for text in leg.get_texts():
    text.set_color('g')
# axs[1].set_ylim(np.min(total_energy)*axis_low_limit, np.max(total_energy)*axis_high_limit)
# twinx axis for TOTEN
ax2 = axs[1].twinx()
ax2.plot(time_steps_energy, TOTEN, 'r-',alpha=0.5)
ax2.axhline(np.mean(stat_TOTEN), color='r', linestyle='--', label=f'Mean: {np.mean(stat_TOTEN):.2f} +/- {np.std(stat_TOTEN):.2f} eV')
ax2.set_ylabel('TOTEN (El. Helmholtz free energy; eV)')
# color the axis red
# ax2.tick_params(axis='y', labelcolor='r')
leg = ax2.legend(loc='upper right')
for text in leg.get_texts():
    text.set_color('r')
# if np.max(TOTEN) > 0 and np.min(TOTEN) < 0:
#     ax2.set_ylim(np.min(TOTEN)*axis_high_limit, np.max(TOTEN)*axis_high_limit)
# elif np.max(TOTEN) > 0 and np.min(TOTEN) > 0:
#     ax2.set_ylim(np.min(TOTEN)*axis_low_limit, np.max(TOTEN)*axis_high_limit)
# elif np.max(TOTEN) < 0 and np.min(TOTEN) < 0:
#     ax2.set_ylim(np.min(TOTEN)*axis_high_limit, np.max(TOTEN)*axis_low_limit)
# else:
#     ax2.set_ylim(np.min(TOTEN)*axis_low_limit, np.max(TOTEN)*axis_high_limit)

# Panel 3: evo_cell_volume vs time-step
axs[2].plot(time_steps_volume, cell_volume, 'r-', alpha=0.5)
axs[2].axhline(np.mean(stat_cell_volume), color='r', linestyle='--', label=f'Mean: {np.mean(stat_cell_volume):.2f} +/- {np.std(stat_cell_volume):.2f} Å³')
axs[2].set_ylabel('Cell Volume (Å³)')
axs[2].grid()
axs[2].legend()
# axs[2].set_ylim(np.min(cell_volume)*axis_low_limit, np.max(cell_volume)*axis_high_limit)

# Panel 4: evo_mean_temp vs time-step
axs[3].plot(time_steps_temp, mean_temp, 'm-', alpha=0.5)
axs[3].axhline(np.mean(stat_mean_temp), color='m', linestyle='--', label=f'Mean: {np.mean(stat_mean_temp):.2f} +/- {np.std(stat_mean_temp):.2f} K')
axs[3].set_xlabel('Time-step')
axs[3].set_ylabel('Temperature (K)')
axs[3].legend()
axs[3].grid()
# axs[3].set_ylim(np.min(mean_temp)*axis_low_limit, np.max(mean_temp)*axis_high_limit)

# plot title
plt.suptitle(composition_template.format(ratio=ratio), fontsize=12)

# Improve layout to prevent overlapping labels
plt.tight_layout()
# plt.show()
# Save the figure to a file
plt.savefig("analysis/plot_evo_data.png", dpi=300)


# create a log file with all means
with open("analysis/log.plot_evo_data", "w") as log_file:
    log_file.write(f"Mean Total Pressure: {np.mean(stat_total_pressure):.2f} +/- {np.std(stat_total_pressure):.2f} GPa\n")
    log_file.write(f"Mean External Pressure: {np.mean(stat_external_pressure):.2f} +/- {np.std(stat_external_pressure):.2f} GPa\n")
    log_file.write(f"Mean Total Energy: {np.mean(stat_total_energy):.2f} +/- {np.std(stat_total_energy):.2f} eV\n")
    log_file.write(f"Mean Internal Energy: {np.mean(stat_internal_energy):.2f} +/- {np.std(stat_internal_energy):.2f} eV\n")
    log_file.write(f"Mean Free Energy: {np.mean(stat_free_energy):.2f} +/- {np.std(stat_free_energy):.2f} eV\n")
    log_file.write(f"Mean Cell Volume: {np.mean(stat_cell_volume):.2f} +/- {np.std(stat_cell_volume):.2f} Å³\n")
    log_file.write(f"Mean Temperature: {np.mean(stat_mean_temp):.2f} +/- {np.std(stat_mean_temp):.2f} K\n")

EOF









echo ""
######################################
# # if TI_mode=0
# if [ "$TI_mode" -eq 0 ]; then
#     echo "Running MSD calculation ..."
#     # echo "Diffusion calculation deactivated."
#     # Diffusion calculcation
#     module purge
#     source setting_env.sh
#     cp $MSD_python_file .
#     python msd_calc_v3.py
#     module purge
# else
#     echo "Diffusion calculation deactivated."
# fi
######################################
echo ""
rm -f msd_calc_v3.py setting_env.sh
echo "Diffusion calculation deactivated."
echo ""




# end=$(date +%s)  # End time in seconds
# runtime=$((end - start))
# runtime in proper format
# run_mins=$((runtime / 60))

echo "################################"
echo "Done with data_4_analysis.sh for $parent_dir_name"
# echo "Runtime: $runtime seconds"
echo "################################"
echo

module purge

#exit
