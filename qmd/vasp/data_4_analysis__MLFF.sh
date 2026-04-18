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
peavg_script="${script_dir}/peavg_mlff.sh"
band_summary_script="${script_dir}/extract_band_occupations.py"
if [[ ! -f "$band_summary_script" ]]; then
	band_summary_script="/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/qmd/vasp/extract_band_occupations.py"
fi

if [[ ! -f OUTCAR ]]; then
	echo "Error: OUTCAR not found in $parent_dir"
	return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$peavg_script" ]]; then
	echo "Error: peavg_mlff.sh not found beside this script: $peavg_script"
	return 1 2>/dev/null || exit 1
fi

echo "################################"
echo "Running data_4_analysis__MLFF.sh for $parent_dir_name"
echo "################################"

mkdir -p analysis

bash "$peavg_script" OUTCAR || {
	echo "Error: peavg_mlff.sh failed"
	return 1 2>/dev/null || exit 1
}

if [[ -f "$band_summary_script" ]]; then
	echo "Extracting occupied-band summary from OUTCAR."
	python "$band_summary_script" \
		--outcar OUTCAR \
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
echo "Diffusion calculation deactivated."
echo ""
echo "################################"
echo "Done with data_4_analysis__MLFF.sh for $parent_dir_name"
echo "################################"
echo

module purge
