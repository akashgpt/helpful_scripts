#!/bin/bash

#############################################################
# Summary:
#   Analyze VASP MLFF OUTCAR files using block-based parsing.
#   Writes evo_*.dat files, peavg summaries, and a quick-look plot.
#
# Usage: source data_4_analysis__MLFF.sh
#############################################################

parent_dir=$(pwd)
parent_dir_name=$(basename "$parent_dir")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
outcar_path="${1:-OUTCAR}"
peavg_script="${script_dir}/peavg_mlff.sh"
band_summary_script="${script_dir}/extract_band_occupations.py"
mlff_training_summary_script="${script_dir}/extract_mlff_training_summary.py"
snapshot_plot_script="${script_dir}/plot_vasp_current_snapshot.py"
mlff_summary_path="analysis/.mlff_training_summary.tmp"

resolve_helper_script() {
	local script_name="$1"
	local candidate_path resolved_path

	candidate_path="${script_dir}/${script_name}"
	if [[ -f "$candidate_path" ]]; then
		printf '%s\n' "$candidate_path"
		return 0
	fi

	candidate_path="${script_dir}/Box_Lars/${script_name}"
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

validate_mlff_outcar() {
	local ionic_step_count ml_toten_count temperature_count total_pressure_count external_pressure_count kinetic_pressure_count volume_count

	ionic_step_count=$(grep -c "Ionic step" "$outcar_path" || true)
	ml_toten_count=$(grep -c "free  energy ML TOTEN" "$outcar_path" || true)
	temperature_count=$(grep -c "(temperature" "$outcar_path" || true)
	total_pressure_count=$(grep -c "total pressure  =" "$outcar_path" || true)
	external_pressure_count=$(grep -c "external pressure =" "$outcar_path" || true)
	kinetic_pressure_count=$(grep -c "kinetic pressure (ideal gas correction)" "$outcar_path" || true)
	volume_count=$(grep -c "volume of cell :" "$outcar_path" || true)

	if [[ "$ionic_step_count" -eq 0 || "$ml_toten_count" -eq 0 || "$temperature_count" -eq 0 || "$total_pressure_count" -eq 0 || "$external_pressure_count" -eq 0 || "$kinetic_pressure_count" -eq 0 || "$volume_count" -eq 0 ]]; then
		echo "Error: $outcar_path does not look like a parseable MLFF MD OUTCAR."
		echo "Expected non-zero counts for 'Ionic step', 'free  energy ML TOTEN', '(temperature)', 'total pressure  =', 'external pressure =', 'kinetic pressure (ideal gas correction)', and 'volume of cell :'."
		echo "Found: ionic_step=$ionic_step_count, ml_toten=$ml_toten_count, temperature=$temperature_count, total_pressure=$total_pressure_count, external_pressure=$external_pressure_count, kinetic_pressure=$kinetic_pressure_count, volume=$volume_count"
		echo "If this OUTCAR is not an MLFF MD run, do not run the MLFF data_4_analysis workflow on it."
		return 1
	fi
}

if ! peavg_script=$(resolve_helper_script "peavg_mlff.sh"); then
	echo "Error: peavg_mlff.sh not found beside this script or on PATH."
	return 1 2>/dev/null || exit 1
fi

resolved_band_summary_script=$(command -v extract_band_occupations.py 2>/dev/null || true)
if [[ -n "$resolved_band_summary_script" && -f "$resolved_band_summary_script" ]]; then
	band_summary_script="$resolved_band_summary_script"
fi
if [[ ! -f "$band_summary_script" ]]; then
	band_summary_script="/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/qmd/vasp/extract_band_occupations.py"
fi

resolved_mlff_training_summary_script=$(command -v extract_mlff_training_summary.py 2>/dev/null || true)
if [[ -n "$resolved_mlff_training_summary_script" && -f "$resolved_mlff_training_summary_script" ]]; then
	mlff_training_summary_script="$resolved_mlff_training_summary_script"
fi
if [[ ! -f "$mlff_training_summary_script" ]]; then
	mlff_training_summary_script="/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/qmd/vasp/extract_mlff_training_summary.py"
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

if ! validate_mlff_outcar; then
	return 1 2>/dev/null || exit 1
fi

echo "################################"
echo "Running data_4_analysis__MLFF.sh for $parent_dir_name"
echo "################################"

mkdir -p analysis

if [[ -f CONTCAR && -s CONTCAR && -f "$snapshot_plot_script" ]]; then
	echo "Plotting current CONTCAR snapshot to analysis/current_snapshot.png."
	python "$snapshot_plot_script" \
		--structure CONTCAR \
		--outcar "$outcar_path" \
		--output analysis/current_snapshot.png \
		--title "Current CONTCAR snapshot: $parent_dir_name" || \
		echo "Warning: failed to create analysis/current_snapshot.png"
else
	echo "Warning: CONTCAR or snapshot helper missing; skipping current_snapshot.png"
fi

bash "$peavg_script" "$outcar_path" || {
	echo "Error: peavg_mlff.sh failed"
	return 1 2>/dev/null || exit 1
}

if [[ -f "$band_summary_script" ]]; then
	echo "Extracting occupied-band summary from OUTCAR."
	python "$band_summary_script" \
		--outcar "$outcar_path" \
		--output analysis/band_occupations_summary.out \
		--selection second_last
	if grep -q '^flag_no_nonzero_occupied_bands=yes$' analysis/band_occupations_summary.out; then
		echo "Warning: no non-zero occupied bands were found in the selected band table."
	fi
else
	echo "Warning: band summary helper not found: $band_summary_script"
fi

if [[ -f "$mlff_training_summary_script" ]]; then
	echo "Extracting MLFF training/storage summary from ML_LOGFILE."
	rm -f analysis/mlff_training_summary.out "$mlff_summary_path"
	python "$mlff_training_summary_script" \
		--ml-logfile ML_LOGFILE \
		--outcar "$outcar_path" \
		--incar INCAR \
		--output "$mlff_summary_path"
else
	echo "Warning: MLFF training summary helper not found: $mlff_training_summary_script"
fi

python <<'PY'
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_mlff_table(file_path: Path) -> dict[str, np.ndarray]:
	"""Load the MLFF step table into numpy arrays."""
	data = np.genfromtxt(file_path, names=True, delimiter="\t", dtype=float, encoding="utf-8")
	if data.shape == ():
		data = np.array([data], dtype=data.dtype)
	return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def write_series(file_path: Path, values: np.ndarray) -> None:
	"""Write one value per line, preserving NaNs."""
	with file_path.open("w", encoding="utf-8") as handle:
		for value in values:
			handle.write(("nan" if not np.isfinite(value) else f"{value:.16g}") + "\n")


def finite_tail(values: np.ndarray, ratio_value: float) -> np.ndarray:
	"""Select the final 1 - 1/ratio fraction of finite values."""
	finite_values = values[np.isfinite(values)]
	if finite_values.size == 0:
		return finite_values
	if ratio_value <= 1:
		return finite_values
	start = int(len(finite_values) * (1.0 - (1.0 / ratio_value)))
	return finite_values[start:]


def finite_bounds(values: np.ndarray, low_scale: float, high_scale: float):
	"""Return plot limits for a finite series."""
	finite_values = values[np.isfinite(values)]
	if finite_values.size == 0:
		return None
	minimum = float(np.min(finite_values))
	maximum = float(np.max(finite_values))
	if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-12):
		padding = max(abs(minimum) * 0.05, 1e-6)
		return minimum - padding, maximum + padding
	return minimum * low_scale, maximum * high_scale


def plot_series_with_mean(
	axis,
	values: np.ndarray,
	color: str,
	mean_values: np.ndarray,
	ylabel: str,
	legend_template: str,
	legend_loc: str,
	low_scale: float,
	high_scale: float,
) -> None:
	"""Plot a finite series and annotate it with a standard-style mean line."""
	mask = np.isfinite(values)
	axis.set_ylabel(ylabel)
	axis.grid()
	if not np.any(mask):
		axis.text(0.5, 0.5, "No finite data", transform=axis.transAxes, ha="center", va="center")
		return
	time_steps = np.arange(1, len(values) + 1)[mask]
	finite_values = values[mask]
	axis.plot(time_steps, finite_values, color=color, linestyle='-', alpha=0.5)
	if mean_values.size:
		mean_value = float(np.nanmean(mean_values))
		std_value = float(np.nanstd(mean_values))
		axis.axhline(mean_value, color=color, linestyle='--', label=legend_template.format(mean=mean_value, std=std_value))
		legend = axis.legend(loc=legend_loc)
		for text in legend.get_texts():
			text.set_color(color)
	bounds = finite_bounds(finite_values, low_scale, high_scale)
	if bounds is not None:
		axis.set_ylim(*bounds)


def load_species_metadata(run_dir: Path) -> str:
	"""Build a compact composition summary for the plot title."""
	for structure_name in ("CONTCAR", "POSCAR"):
		structure_path = run_dir / structure_name
		if not structure_path.exists():
			continue
		lines = [line.strip() for line in structure_path.read_text(errors="ignore").splitlines()]
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


current_dir = Path.cwd()
analysis_dir = current_dir / "analysis"
table = load_mlff_table(analysis_dir / "mlff_step_data.tsv")
composition_template = load_species_metadata(current_dir)

axis_low_limit = 0.90
axis_high_limit = 1.10

ratio_value = 4.0
ratio_file = current_dir / "ratio"
if ratio_file.is_file():
	try:
		ratio_value = float(ratio_file.read_text().split()[0])
	except (IndexError, ValueError):
		ratio_value = 4.0

write_series(analysis_dir / "evo_total_pressure.dat", table["total_pressure_kbar"])
write_series(analysis_dir / "evo_external_pressure.dat", table["external_pressure_kbar"])
write_series(analysis_dir / "evo_kinetic_pressure.dat", table["kinetic_pressure_kbar"])
write_series(analysis_dir / "evo_pullay_stress.dat", table["pullay_stress_kbar"])
write_series(analysis_dir / "evo_cell_volume.dat", table["cell_volume"])
write_series(analysis_dir / "evo_total_energy.dat", table["total_energy"])
write_series(analysis_dir / "evo_TOTEN.dat", table["free_energy"])
write_series(analysis_dir / "evo_free_energy.dat", table["free_energy"])
write_series(analysis_dir / "evo_free_energy_ML.dat", table["free_energy_ml"])
write_series(analysis_dir / "evo_free_energy_DFT.dat", table["free_energy_dft"])
write_series(analysis_dir / "evo_internal_energy.dat", table["internal_energy"])
write_series(analysis_dir / "evo_mean_temp.dat", table["temperature"])

pressure_gpa = table["total_pressure_kbar"] * 0.1
external_pressure_gpa = table["external_pressure_kbar"] * 0.1
total_energy = table["total_energy"]
free_energy = table["free_energy"]
internal_energy = table["internal_energy"]
cell_volume = table["cell_volume"]
mean_temp = table["temperature"]

stat_total_pressure = finite_tail(pressure_gpa, ratio_value)
stat_external_pressure = finite_tail(external_pressure_gpa, ratio_value)
stat_total_energy = finite_tail(total_energy, ratio_value)
stat_free_energy = finite_tail(free_energy, ratio_value)
stat_internal_energy = finite_tail(internal_energy, ratio_value)
stat_cell_volume = finite_tail(cell_volume, ratio_value)
stat_mean_temp = finite_tail(mean_temp, ratio_value)

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 12))
fig.subplots_adjust(hspace=0.5)

plot_series_with_mean(
	axs[0],
	pressure_gpa,
	"tab:blue",
	stat_total_pressure,
	"Total Pressure (GPa)",
	"Mean: {mean:.2f} +/- {std:.2f} GPa",
	"upper left",
	axis_low_limit,
	axis_high_limit,
)
ax1 = axs[0].twinx()
plot_series_with_mean(
	ax1,
	external_pressure_gpa,
	"tab:red",
	stat_external_pressure,
	"External Pressure (GPa)",
	"Mean: {mean:.2f} +/- {std:.2f} GPa",
	"upper right",
	axis_low_limit,
	axis_high_limit,
)

plot_series_with_mean(
	axs[1],
	total_energy,
	"tab:green",
	stat_total_energy,
	"Total Energy (ETOTAL; eV)",
	"Mean: {mean:.2f} +/- {std:.2f} eV",
	"upper left",
	axis_low_limit,
	axis_high_limit,
)
ax2 = axs[1].twinx()
plot_series_with_mean(
	ax2,
	free_energy,
	"tab:red",
	stat_free_energy,
	"Free Energy (TOTEN/ML TOTEN; eV)",
	"Mean: {mean:.2f} +/- {std:.2f} eV",
	"upper right",
	axis_low_limit,
	axis_high_limit,
)

plot_series_with_mean(
	axs[2],
	cell_volume,
	"tab:purple",
	stat_cell_volume,
	"Cell Volume (A^3)",
	"Mean: {mean:.2f} +/- {std:.2f} A^3",
	"upper left",
	axis_low_limit,
	axis_high_limit,
)

plot_series_with_mean(
	axs[3],
	mean_temp,
	"tab:orange",
	stat_mean_temp,
	"Temperature (K)",
	"Mean: {mean:.2f} +/- {std:.2f} K",
	"upper left",
	axis_low_limit,
	axis_high_limit,
)
axs[3].set_xlabel("Ionic Step")

plt.suptitle(composition_template.format(ratio=ratio_value), fontsize=12)
plt.tight_layout()
plt.savefig(analysis_dir / "plot_evo_data.png", dpi=300)

with (analysis_dir / "log.plot_evo_data").open("w", encoding="utf-8") as handle:
	handle.write(f"Mean Total Pressure: {np.nanmean(stat_total_pressure):.2f} +/- {np.nanstd(stat_total_pressure):.2f} GPa\n" if stat_total_pressure.size else "Mean Total Pressure: nan +/- nan GPa\n")
	handle.write(f"Mean External Pressure: {np.nanmean(stat_external_pressure):.2f} +/- {np.nanstd(stat_external_pressure):.2f} GPa\n" if stat_external_pressure.size else "Mean External Pressure: nan +/- nan GPa\n")
	handle.write(f"Mean Total Energy: {np.nanmean(stat_total_energy):.2f} +/- {np.nanstd(stat_total_energy):.2f} eV\n" if stat_total_energy.size else "Mean Total Energy: nan +/- nan eV\n")
	handle.write(f"Mean Internal Energy: {np.nanmean(stat_internal_energy):.2f} +/- {np.nanstd(stat_internal_energy):.2f} eV\n" if stat_internal_energy.size else "Mean Internal Energy: nan +/- nan eV\n")
	handle.write(f"Mean Free Energy: {np.nanmean(stat_free_energy):.2f} +/- {np.nanstd(stat_free_energy):.2f} eV\n" if stat_free_energy.size else "Mean Free Energy: nan +/- nan eV\n")
	handle.write(f"Mean Cell Volume: {np.nanmean(stat_cell_volume):.2f} +/- {np.nanstd(stat_cell_volume):.2f} A^3\n" if stat_cell_volume.size else "Mean Cell Volume: nan +/- nan A^3\n")
	handle.write(f"Mean Temperature: {np.nanmean(stat_mean_temp):.2f} +/- {np.nanstd(stat_mean_temp):.2f} K\n" if stat_mean_temp.size else "Mean Temperature: nan +/- nan K\n")
PY

echo "$parent_dir" > analysis/peavg_summary.out
sed -n '1p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
sed -n '4p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
sed -n '24p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
sed -n '25p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
sed -n '10p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
sed -n '11p' analysis/peavg_numbers.out >> analysis/peavg_summary.out

if [[ -f INCAR ]]; then
	grep -m 1 ENCUT INCAR | awk '{print $3}' >> analysis/peavg_summary.out
else
	echo "NA" >> analysis/peavg_summary.out
fi

if [[ -f INCAR ]]; then
	grep -m 1 GGA INCAR | awk '{print $3}' >> analysis/peavg_summary.out
else
	echo "NA" >> analysis/peavg_summary.out
fi

if [[ -f POTCAR ]]; then
	grep -m 1 "TITEL" POTCAR | awk '{print $4}' >> analysis/peavg_summary.out
else
	echo "NA" >> analysis/peavg_summary.out
fi

awk 'END{print $1}' analysis/evo_free_energy.dat >> analysis/peavg_summary.out
sed -n '16p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
sed -n '17p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
sed -n '18p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
sed -n '19p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
sed -n '7p' analysis/peavg_numbers.out >> analysis/peavg_summary.out
if [[ -f analysis/band_occupations_summary.out ]]; then
	cat analysis/band_occupations_summary.out >> analysis/peavg_summary.out
fi

echo ""
echo "--- MLFF Health Check (see analysis/MLFF_HEALTH_check_up.dat) ---"

# ---- Parse run-recorded ML parameters ----
ml_logfile="ML_LOGFILE"

parse_outcar_ml_value() {
	local tag_name="$1"
	grep -m1 "^[[:space:]]*${tag_name}[[:space:]]*=" "$outcar_path" 2>/dev/null \
		| awk -F= '{print $2}' \
		| awk '{print $1}'
}

parse_ml_log_value() {
	local tag_name="$1"
	awk -v tag_name="$tag_name" '
		$NF == tag_name {
			for (i = 1; i <= NF; i++) {
				if ($i == ":") {
					print $(i + 1)
					exit
				}
			}
		}
	' "$ml_logfile" 2>/dev/null
}

ml_mb_declared="NA"; ml_mconf_declared="NA"; ml_ctifor_declared="NA"; ml_lbasis_discard="NA"; ml_istart="NA"
ml_mb_declared=$(parse_outcar_ml_value "ML_MB")
[[ -z "$ml_mb_declared" ]] && ml_mb_declared=$(parse_ml_log_value "ML_MB")
[[ -z "$ml_mb_declared" ]] && ml_mb_declared="NA"
ml_mconf_declared=$(parse_outcar_ml_value "ML_MCONF")
[[ -z "$ml_mconf_declared" ]] && ml_mconf_declared=$(parse_ml_log_value "ML_MCONF")
[[ -z "$ml_mconf_declared" ]] && ml_mconf_declared="NA"
ml_ctifor_declared=$(parse_outcar_ml_value "ML_CTIFOR")
[[ -z "$ml_ctifor_declared" ]] && ml_ctifor_declared=$(parse_ml_log_value "ML_CTIFOR")
[[ -z "$ml_ctifor_declared" ]] && ml_ctifor_declared="NA"
ml_lbasis_discard=$(parse_outcar_ml_value "ML_LBASIS_DISCARD")
[[ -z "$ml_lbasis_discard" ]] && ml_lbasis_discard=$(parse_ml_log_value "ML_LBASIS_DISCARD")
[[ -z "$ml_lbasis_discard" ]] && ml_lbasis_discard="NA"
ml_istart=$(parse_outcar_ml_value "ML_ISTART")
[[ -z "$ml_istart" ]] && ml_istart=$(parse_ml_log_value "ML_ISTART")
[[ -z "$ml_istart" ]] && ml_istart="NA"

# ---- Parse ML_LOGFILE ----
dft_call_index="NA"; n_species_triplets=0; last_lconf=""; total_lconf_lines=0
total_steps=0; last_max_f="NA"; last_rms_f="NA"; last_thresh="NA"; dft_rate="NA"
ml_abn_size="NA"

if [[ -f "$ml_logfile" ]]; then
	last_lconf=$(grep "^LCONF" "$ml_logfile" 2>/dev/null | tail -1)
	total_lconf_lines=$(grep -c "^LCONF" "$ml_logfile" 2>/dev/null || echo 0)
	if [[ -n "$last_lconf" ]]; then
		dft_call_index=$(echo "$last_lconf" | awk '{print $2}')
		n_fields=$(echo "$last_lconf" | awk '{print NF}')
		n_species_triplets=$(( (n_fields - 2) / 3 ))
	fi
	last_beef=$(grep "^BEEF" "$ml_logfile" 2>/dev/null | tail -1)
	first_beef=$(grep "^BEEF" "$ml_logfile" 2>/dev/null | awk 'NF>=6 && $4+0 > 0 {print; exit}')
	if [[ -n "$last_beef" ]]; then
		total_steps=$(echo "$last_beef" | awk '{print $2 + 1}')
		last_max_f=$(echo "$last_beef" | awk '{printf "%.4f", $4}')
		last_rms_f=$(echo "$last_beef" | awk '{printf "%.4f", $5}')
		last_thresh=$(echo "$last_beef" | awk '{print $6}')
		[[ "$total_steps" -gt 0 ]] && \
			dft_rate=$(awk "BEGIN {printf \"%.2f\", 100.0 * $total_lconf_lines / $total_steps}")
	fi
fi
[[ -f ML_ABN ]] && ml_abn_size=$(du -sh -- ML_ABN 2>/dev/null | awk 'NR==1 {print $1; exit}' | tr -d '\n')

# ---- Read the compact MLFF training summary from SPRSC/LCONF data ----
get_mlff_summary_value() {
	local key="$1"
	awk -F= -v key="$key" '$1 == key {print $2; exit}' "$mlff_summary_path"
}

mlff_memory_total_gb="NA"
stored_structures="NA"
stored_structures_pct="NA"
ml_mb_summary="$ml_mb_declared"
ml_mconf_summary="$ml_mconf_declared"
if [[ -f "$mlff_summary_path" ]]; then
	mlff_memory_total_gb=$(get_mlff_summary_value "mlff_memory_total_gb")
	stored_structures=$(get_mlff_summary_value "mlff_stored_structures_after_sparsification")
	stored_structures_pct=$(get_mlff_summary_value "mlff_stored_structures_percent_of_ml_mconf")
	ml_mb_summary=$(get_mlff_summary_value "mlff_ml_mb")
	ml_mconf_summary=$(get_mlff_summary_value "mlff_ml_mconf")
	[[ -z "$mlff_memory_total_gb" ]] && mlff_memory_total_gb="NA"
	[[ -z "$stored_structures" ]] && stored_structures="NA"
	[[ -z "$stored_structures_pct" ]] && stored_structures_pct="NA"
	[[ -z "$ml_mb_summary" ]] && ml_mb_summary="$ml_mb_declared"
	[[ -z "$ml_mconf_summary" ]] && ml_mconf_summary="$ml_mconf_declared"
fi

# ---- Write combined simple report to analysis/MLFF_HEALTH_check_up.dat ----
{
	echo "########################################"
	echo "# MLFF Health Check"
	echo "# Directory: $parent_dir"
	echo "# Generated: $(date)"
	echo "########################################"
	echo ""

	echo "Memory estimate: ${mlff_memory_total_gb} GB"
	echo "Stored structures: ${stored_structures} / ${ml_mconf_summary} (${stored_structures_pct}%)"
	if [[ -f "$mlff_summary_path" ]]; then
		grep -E '^mlff_basis_.*_after_sparsification=' "$mlff_summary_path" | while IFS='=' read -r basis_key basis_value; do
			species_name="${basis_key#mlff_basis_}"
			species_name="${species_name%_after_sparsification}"
			basis_percent=$(get_mlff_summary_value "mlff_basis_${species_name}_percent_of_ml_mb")
			echo "Basis ${species_name}: ${basis_value:-NA} / ${ml_mb_summary:-NA} (${basis_percent:-NA}%)"
		done
	fi
	if [[ "$last_max_f" != "NA" ]]; then
		echo "BEEF last: max_F=$last_max_f rms_F=$last_rms_f eV/A  |  DFT call rate: ${dft_rate}%  ($total_lconf_lines / $total_steps steps)"
		echo "Restart ML_CTIFOR candidate: $last_thresh"
	else
		echo "BEEF last: NA"
		echo "Restart ML_CTIFOR candidate: NA"
	fi
	echo ""
	echo "Meaning:"
	echo "  Stored structures = SPRSC nstr_spar, compared against ML_MCONF."
	echo "  Basis per species = SPRSC nlrc_spar after sparsification, compared against ML_MB."
	echo "  LCONF old/new counts are pre-sparsification growth diagnostics; they are not the final retained basis size."
	echo ""
	echo "Restart readiness:"
	if [[ -f ML_ABN ]]; then
		echo "  ML_ABN: present (${ml_abn_size}; copy to ML_AB before restart)"
	else
		echo "  ML_ABN: NOT found (no new ab initio training database to copy to ML_AB)"
	fi
	[[ -f CONTCAR ]] \
		&& echo "  CONTCAR: present" \
		|| echo "  CONTCAR: NOT found"
	echo "  Restart tip: use CONTCAR as POSCAR, copy ML_ABN to ML_AB, then continue training."

	# Saturation + discard warning
	if [[ "$ml_lbasis_discard" == "F" || "$ml_lbasis_discard" == ".FALSE." ]] \
		&& [[ -n "$last_lconf" && "$ml_mb_declared" != "NA" && "$ml_mb_declared" -gt 0 ]]; then
		_saturated=0
		for (( i=0; i<n_species_triplets; i++ )); do
			field_nafter=$(( 5 + i * 3 ))
			n_after_chk=$(echo "$last_lconf" | awk -v f="$field_nafter" '{print $f}')
			[[ "$n_after_chk" -ge "$ml_mb_declared" ]] && _saturated=1
		done
		if [[ "$_saturated" -eq 1 ]]; then
			_new_mb=$(( ml_mb_declared * 3 / 2 ))
			echo ""
			echo "  *** WARNING: basis saturated AND ML_LBASIS_DISCARD=F ***"
			echo "  *** Increase ML_MB before restart (e.g. ML_MB = $_new_mb). Increase ML_MCONF only if SPRSC nstr_spar approaches ML_MCONF. ***"
		fi
	fi
	echo ""
	echo "########################################"
} > analysis/MLFF_HEALTH_check_up.dat

if [[ -f analysis/MLFF_HEALTH_check_up.dat ]]; then
	cat analysis/MLFF_HEALTH_check_up.dat >> analysis/peavg_summary.out
fi

# ---- Brief stdout summary ----
if [[ -f "$mlff_summary_path" ]]; then
	echo "  Stored structures: ${stored_structures:-NA} / ${ml_mconf_summary:-NA} (${stored_structures_pct:-NA}%)"

	grep -E '^mlff_basis_.*_after_sparsification=' "$mlff_summary_path" | while IFS='=' read -r basis_key basis_value; do
		species_name="${basis_key#mlff_basis_}"
		species_name="${species_name%_after_sparsification}"
		basis_percent=$(get_mlff_summary_value "mlff_basis_${species_name}_percent_of_ml_mb")
		echo "  Basis ${species_name}: ${basis_value:-NA} / ${ml_mb_summary:-NA} (${basis_percent:-NA}%)"
	done
else
	echo "  Stored structures: NA / $ml_mconf_declared (NA%)"
	if [[ -n "$last_lconf" ]]; then
		for (( i=0; i<n_species_triplets; i++ )); do
			sp=$(echo "$last_lconf" | awk -v f=$(( 3 + i*3 )) '{print $f}')
			n_after=$(echo "$last_lconf" | awk -v f=$(( 5 + i*3 )) '{print $f}')
			[[ "$ml_mb_declared" != "NA" && "$ml_mb_declared" -gt 0 ]] && \
				pct=$(awk "BEGIN {printf \"%.1f\", 100.0 * $n_after / $ml_mb_declared}") || pct="NA"
			echo "  Basis $sp: $n_after / $ml_mb_declared (${pct}%)"
		done
	fi
fi
[[ "$last_max_f" != "NA" ]] && \
	echo "  BEEF last: max_F=$last_max_f rms_F=$last_rms_f eV/A  |  DFT call rate: ${dft_rate}%  ($total_lconf_lines / $total_steps steps)"
[[ "$last_thresh" != "NA" ]] && \
	echo "  Restart ML_CTIFOR candidate: $last_thresh"
echo "  Full report: analysis/MLFF_HEALTH_check_up.dat"
echo "--- End MLFF Health Check ---"

echo ""
echo "Diffusion calculation deactivated."
echo ""
echo "################################"
echo "Done with data_4_analysis__MLFF.sh for $parent_dir_name"
echo "################################"
echo

module purge
