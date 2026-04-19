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

# ---- Parse INCAR ML parameters ----
ml_mb_declared="NA"; ml_mconf_declared="NA"; ml_ctifor_declared="NA"; ml_lbasis_discard="NA"; ml_istart="NA"
if [[ -f INCAR ]]; then
	ml_mb_declared=$(grep -m1 "^[[:space:]]*ML_MB[[:space:]=]" INCAR \
		| sed 's/[^0-9]/ /g' | awk '{print $1}')
	[[ -z "$ml_mb_declared" ]] && ml_mb_declared="NA"
	ml_mconf_declared=$(grep -m1 "^[[:space:]]*ML_MCONF[[:space:]=]" INCAR \
		| sed 's/[^0-9]/ /g' | awk '{print $1}')
	[[ -z "$ml_mconf_declared" ]] && ml_mconf_declared="NA"
	ml_ctifor_declared=$(grep -m1 "^[[:space:]]*ML_CTIFOR[[:space:]=]" INCAR \
		| awk -F'[=[:space:]]+' '{for(i=1;i<=NF;i++) if($i~/^[0-9]/) {print $i+0; exit}}')
	[[ -z "$ml_ctifor_declared" ]] && ml_ctifor_declared="NA"
	ml_lbasis_discard=$(grep -m1 "^[[:space:]]*ML_LBASIS_DISCARD[[:space:]=]" INCAR \
		| awk '{print $NF}' | tr -d '[:space:]')
	[[ -z "$ml_lbasis_discard" ]] && ml_lbasis_discard="NA"
	ml_istart=$(grep -m1 "^[[:space:]]*ML_ISTART[[:space:]=]" INCAR \
		| sed 's/[^0-9]/ /g' | awk '{print $1}')
	[[ -z "$ml_istart" ]] && ml_istart="NA"
fi

# ---- Parse ML_LOGFILE ----
ml_logfile="ML_LOGFILE"
dft_call_index="NA"; n_species_triplets=0; last_lconf=""; total_lconf_lines=0
total_steps=0; last_max_f="NA"; last_rms_f="NA"; last_thresh="NA"; dft_rate="NA"
ml_ff_size="NA"

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
		last_thresh=$(echo "$last_beef" | awk '{printf "%.4f", $6}')
		[[ "$total_steps" -gt 0 ]] && \
			dft_rate=$(awk "BEGIN {printf \"%.2f\", 100.0 * $total_lconf_lines / $total_steps}")
	fi
fi
[[ -f ML_FF ]] && ml_ff_size=$(du -sh ML_FF 2>/dev/null | awk '{print $1}')

# ---- Write detailed report to analysis/MLFF_HEALTH_check_up.dat ----
{
	echo "########################################"
	echo "# MLFF Health Check"
	echo "# Directory: $parent_dir"
	echo "# Generated: $(date)"
	echo "########################################"
	echo ""

	echo "=== [1] INCAR ML Parameters ==="
	echo "  ML_MB             = $ml_mb_declared"
	echo "    (guidance: increase 1.5-2x before restart if any species basis reaches ~90% full)"
	echo "  ML_MCONF          = $ml_mconf_declared"
	echo "    (max new LRC candidates extracted per DFT call; default=1500)"
	echo "    (independent of ML_MB; no direct saturation metric available from LCONF lines)"
	echo "  ML_CTIFOR         = $ml_ctifor_declared"
	echo "    (guidance: 0.05-0.15 eV/A for production; NA = VASP adaptive default ~0.2 eV/A)"
	echo "    (lower -> more DFT calls -> better model but slower; higher -> fewer calls but overconfident)"
	echo "  ML_LBASIS_DISCARD = $ml_lbasis_discard"
	echo "    (guidance: F = safe, no data lost; T = run continues past ceiling but forgets old configs)"
	echo "  ML_ISTART         = $ml_istart"
	echo "    (guidance: always use ML_ISTART=2 when restarting; 0 discards all prior training)"
	echo ""

	echo "=== [2] Basis Growth (LCONF) ==="
	echo "  LCONF format: LCONF <DFT_call#> <species> <nlrc_new> <nlrc_old>"
	echo "  nlrc_old (N_before): LRC count for this species before this DFT call's update"
	echo "  nlrc_new (N_after):  LRC count after adding and sparsifying new environments"
	echo "  Key diagnostics:"
	echo "    N_after > N_before -> new environments added (model still learning)"
	echo "    N_after == N_before -> nothing new survived sparsification (plateaued or ML_MB limiting)"
	echo "    N_after near ML_MB -> basis ceiling too tight; increase ML_MB"
	echo "    N_after far below ML_MB -> ML_MB has comfortable headroom (may be safely reduced)"
	if [[ -z "$last_lconf" ]]; then
		echo "  No LCONF lines found in ML_LOGFILE — training may not have started."
	else
		echo "  Total DFT calls: $total_lconf_lines  (last at DFT call #$dft_call_index)"
		echo ""
		# Also grab first LCONF to show growth trajectory
		first_lconf=$(grep "^LCONF" "$ml_logfile" 2>/dev/null | head -1)
		for (( i=0; i<n_species_triplets; i++ )); do
			field_sp=$(( 3 + i * 3 )); field_nafter=$(( 4 + i * 3 )); field_nbefore=$(( 5 + i * 3 ))
			sp=$(echo "$last_lconf" | awk -v f="$field_sp" '{print $f}')
			n_after=$(echo "$last_lconf" | awk -v f="$field_nafter" '{print $f}')
			n_before=$(echo "$last_lconf" | awk -v f="$field_nbefore" '{print $f}')
			first_n_after=$(echo "$first_lconf" | awk -v f="$field_nafter" '{print $f}')
			delta=$(( n_after - n_before ))

			if [[ "$ml_mb_declared" != "NA" && "$ml_mb_declared" -gt 0 ]]; then
				pct=$(awk "BEGIN {printf \"%.1f\", 100.0 * $n_after / $ml_mb_declared}")
				headroom=$(( ml_mb_declared - n_after ))
				if (( n_after >= ml_mb_declared )); then
					sat_flag="  *** SATURATED: increase ML_MB before restart ***"
				elif awk "BEGIN {exit ($pct < 90)}"; then
					sat_flag="  ** >90% full: plan to increase ML_MB"
				else
					sat_flag=""
				fi
				echo "  $sp (last update): N_before=$n_before -> N_after=$n_after  (delta=$delta)"
				echo "    N_after vs ML_MB: $n_after / $ml_mb_declared ($pct% full, headroom=$headroom)$sat_flag"
				echo "    Growth: first LCONF N_after=$first_n_after -> last N_after=$n_after  (+$(( n_after - first_n_after )) over $total_lconf_lines DFT calls)"
			else
				echo "  $sp (last update): N_before=$n_before -> N_after=$n_after  (delta=$delta)"
				echo "    Growth: first LCONF N_after=$first_n_after -> last N_after=$n_after  (ML_MB not set)"
			fi
			echo ""
		done
	fi

	echo "=== [3] Bayesian Force Error (BEEF) ==="
	echo "  BEEF format: BEEF <step> <E_err_eV/atom> <max_F_err_eV/A> <rms_F_err_eV/A> <adaptive_threshold_eV/A> ..."
	if [[ "$last_max_f" == "NA" ]]; then
		echo "  No BEEF lines found in ML_LOGFILE."
	else
		echo "  Total MD steps: $total_steps"
		echo ""
		echo "  Last Bayesian force error:"
		if awk "BEGIN {exit ($last_max_f < 0.50)}"; then max_flag="  ** >0.50: very uncertain in some configs"
		else max_flag=""; fi
		echo "    max_F = ${last_max_f} eV/A${max_flag}"
		if awk "BEGIN {exit ($last_rms_f < 0.05)}"; then
			rms_flag="  ** aim for <0.05 for production accuracy"
		elif awk "BEGIN {exit ($last_rms_f < 0.10)}"; then
			rms_flag="  * 0.05-0.10: acceptable for exploratory runs"
		else
			rms_flag=""
		fi
		echo "    rms_F = ${last_rms_f} eV/A${rms_flag}"
		echo "    (guidance: rms_F < 0.05 eV/A for production; < 0.10 for exploratory)"
		echo ""
		echo "  Adaptive DFT trigger threshold: ${last_thresh} eV/A  (declared ML_CTIFOR: $ml_ctifor_declared)"
		echo "    (if adaptive threshold >> ML_CTIFOR, VASP has auto-relaxed the trigger)"
		echo ""
		echo "  DFT call rate: $total_lconf_lines / $total_steps steps = ${dft_rate}%"
		if awk "BEGIN {exit ($dft_rate < 10.0)}"; then
			echo "    ** >10%: model still learning; normal early on, worrying after 5k+ steps"
			echo "       -> consider increasing ML_MB or running longer"
		elif awk "BEGIN {exit ($dft_rate < 1.0)}"; then
			echo "    * 1-10%: healthy training range"
		else
			echo "    * <1%: model mature OR ML_CTIFOR is too high (overconfident)"
			echo "       -> if rms_F is still large, lower ML_CTIFOR to force more DFT calls"
		fi
		echo ""
		if [[ -n "$first_beef" ]]; then
			first_step=$(echo "$first_beef" | awk '{print $2}')
			first_max_f=$(echo "$first_beef" | awk '{printf "%.4f", $4}')
			last_step=$(echo "$last_beef" | awk '{print $2}')
			echo "  Force error trend: step $first_step max=$first_max_f -> step $last_step max=$last_max_f eV/A"
			if awk "BEGIN {exit ($last_max_f < $first_max_f)}"; then
				echo "    ** flat or increasing: model may be stuck"
				echo "       -> try lowering ML_CTIFOR to add more diverse DFT calls"
				echo "       -> or increase ML_MB so sparsification retains more configs"
			else
				echo "    (decreasing: model is improving)"
			fi
		fi
	fi
	echo ""

	echo "=== [4] Restart Readiness ==="
	if [[ -f ML_FF ]]; then
		echo "  ML_FF: present (${ml_ff_size})  <- copy to restart directory alongside INCAR/POTCAR"
	else
		echo "  ML_FF: NOT found  <- cannot restart with ML_ISTART=1/2; must retrain from scratch"
	fi
	[[ -f CONTCAR ]] \
		&& echo "  CONTCAR: present  <- use as POSCAR for restart" \
		|| echo "  CONTCAR: NOT found  <- extract last ionic step manually before restart"
	echo "  ML_ISTART (current run): $ml_istart"
	echo "    -> set ML_ISTART=2 in restart INCAR (reads ML_FF and continues training)"
	echo "    -> do NOT use ML_ISTART=0 (discards all learned data)"

	# Saturation + discard warning
	if [[ "$ml_lbasis_discard" == "F" || "$ml_lbasis_discard" == ".FALSE." ]] \
		&& [[ -n "$last_lconf" && "$ml_mb_declared" != "NA" && "$ml_mb_declared" -gt 0 ]]; then
		_saturated=0
		for (( i=0; i<n_species_triplets; i++ )); do
			field_nafter=$(( 4 + i * 3 ))
			n_after_chk=$(echo "$last_lconf" | awk -v f="$field_nafter" '{print $f}')
			[[ "$n_after_chk" -ge "$ml_mb_declared" ]] && _saturated=1
		done
		if [[ "$_saturated" -eq 1 ]]; then
			_new_mb=$(( ml_mb_declared * 3 / 2 ))
			echo ""
			echo "  *** WARNING: basis saturated AND ML_LBASIS_DISCARD=F ***"
			echo "  *** Increase ML_MB before restart (e.g. ML_MB = $_new_mb); scale ML_MCONF proportionally ***"
		fi
	fi
	echo ""
	echo "########################################"
} > analysis/MLFF_HEALTH_check_up.dat

# ---- Brief stdout summary ----
echo "  ML_MB=$ml_mb_declared  ML_MCONF=$ml_mconf_declared  ML_CTIFOR=$ml_ctifor_declared  ML_LBASIS_DISCARD=$ml_lbasis_discard  ML_ISTART=$ml_istart"
if [[ -n "$last_lconf" ]]; then
	for (( i=0; i<n_species_triplets; i++ )); do
		sp=$(echo "$last_lconf" | awk -v f=$(( 3 + i*3 )) '{print $f}')
		n_after=$(echo "$last_lconf" | awk -v f=$(( 4 + i*3 )) '{print $f}')
		[[ "$ml_mb_declared" != "NA" && "$ml_mb_declared" -gt 0 ]] && \
			pct=$(awk "BEGIN {printf \"%.1f\", 100.0 * $n_after / $ml_mb_declared}") || pct="NA"
		echo "  Basis $sp: $n_after / $ml_mb_declared (${pct}%)"
	done
fi
[[ "$last_max_f" != "NA" ]] && \
	echo "  BEEF last: max_F=$last_max_f rms_F=$last_rms_f eV/A  |  DFT call rate: ${dft_rate}%  ($total_lconf_lines / $total_steps steps)"
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
