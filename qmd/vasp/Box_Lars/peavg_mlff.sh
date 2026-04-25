#!/bin/bash

#############################################################
# Summary:
#   Compute peavg-style thermodynamic summaries for VASP MLFF
#   OUTCAR files by parsing data block-by-block per ionic step.
#
# Usage: source peavg_mlff.sh [OUTCAR name]
#############################################################

outcar_path="${1:-OUTCAR}"
data_4_analysis_python="${DATA_4_ANALYSIS_PYTHON:-${HELPFUL_SCRIPTS_PYTHON:-python}}"

export DATA_4_ANALYSIS_PYTHON="$data_4_analysis_python"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

if [[ ! -f "$outcar_path" ]]; then
	echo "Error: OUTCAR file not found: $outcar_path"
	return 1 2>/dev/null || exit 1
fi

mkdir -p analysis

"$data_4_analysis_python" - "$outcar_path" <<'PY'
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


EV_TO_K = 11605.0
R_GAS = 8.3143
KBAR_PER_GPA = 10.0
AVOGADRO_VOLUME = 0.6022
EV_TO_KJMOL = 96.4869
EVG = 160.21863
MPA_PER_GPA = 1000.0
FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"


def parse_float(token: str) -> float:
	"""Convert a numeric token to float."""
	return float(token)


def last_float_from_line(line: str) -> float:
	"""Return the last float-like token from a line."""
	match = re.findall(FLOAT_RE, line)
	if not match:
		return math.nan
	return parse_float(match[-1])


def mean_and_block_error(values: list[float]) -> tuple[float, float]:
	"""Estimate mean and a conservative blocking error."""
	array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
	if array.size == 0:
		return math.nan, math.nan
	mean_value = float(np.mean(array))
	if array.size == 1:
		return mean_value, 0.0
	error_value = float(np.std(array, ddof=1) / math.sqrt(array.size))
	blocked = array.copy()
	while blocked.size >= 4:
		if blocked.size % 2 == 1:
			blocked = blocked[:-1]
		blocked = 0.5 * (blocked[0::2] + blocked[1::2])
		if blocked.size > 1:
			blocked_error = float(np.std(blocked, ddof=1) / math.sqrt(blocked.size))
			error_value = max(error_value, blocked_error)
	return mean_value, error_value


def safe_bc_div(numerator: float, denominator: float) -> float:
	"""Return numerator / denominator, guarding against zero and NaN."""
	if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
		return math.nan
	return numerator / denominator


def safe_sqrt(value: float) -> float:
	"""Return sqrt for non-negative finite values."""
	if not np.isfinite(value) or value < 0:
		return math.nan
	return math.sqrt(value)


def format_value(value: float) -> str:
	"""Format values consistently for peavg outputs."""
	if value is None or not np.isfinite(value):
		return "nan"
	return f"{value:.16g}"


outcar_path = Path(sys.argv[1]).resolve()
run_dir = outcar_path.parent
analysis_dir = run_dir / "analysis"
analysis_dir.mkdir(exist_ok=True)

text = outcar_path.read_text(errors="ignore")
lines = text.splitlines()

ratio_value = 4.0
ratio_file = run_dir / "ratio"
if ratio_file.is_file():
	try:
		ratio_value = float(ratio_file.read_text().split()[0])
	except (IndexError, ValueError):
		ratio_value = 4.0

temperature_match = re.search(r"TEBEG\s*=\s*(%s)" % FLOAT_RE, text)
temperature_value = parse_float(temperature_match.group(1)) if temperature_match else math.nan
temperature_file = run_dir / "temperature"
if temperature_file.is_file():
	try:
		temperature_value = float(temperature_file.read_text().split()[0])
	except (IndexError, ValueError):
		pass

nions_match = re.search(r"NIONS\s*=\s*(\d+)", text)
nions_value = int(nions_match.group(1)) if nions_match else 0

volume_match = re.search(r"volume of cell\s*:\s*(%s)" % FLOAT_RE, text)
initial_volume = parse_float(volume_match.group(1)) if volume_match else math.nan

ions_per_type = []
for line in lines:
	if "ions per type" in line:
		ions_per_type = [int(value) for value in re.findall(r"\d+", line.split("=", 1)[1])]
		break

masses = []
for index, line in enumerate(lines):
	if "Mass of Ions in am" in line:
		for offset in range(index + 1, min(index + 5, len(lines))):
			candidate = lines[offset]
			if "POMASS" in candidate:
				parts = candidate.split("=", 1)[1]
				masses = [float(value) for value in re.findall(FLOAT_RE, parts)]
				break
		break

mass_density = math.nan
if masses and ions_per_type and initial_volume and np.isfinite(initial_volume):
	total_mass = 0.0
	for mass_value, count_value in zip(masses, ions_per_type):
		total_mass += mass_value * count_value
	mass_density = safe_bc_div(total_mass, initial_volume * AVOGADRO_VOLUME)

step_indices = [index for index, line in enumerate(lines) if "Ionic step" in line]
rows = []

for index, start in enumerate(step_indices):
	end = step_indices[index + 1] if index + 1 < len(step_indices) else len(lines)
	block_lines = lines[start:end]
	block_text = "\n".join(block_lines)
	step_match = re.search(r"Ionic step\s+(\d+)", block_lines[0])
	if not step_match:
		continue
	step_number = int(step_match.group(1))
	free_energy_ml = math.nan
	free_energy_dft = math.nan
	internal_energy = math.nan
	total_energy = math.nan
	temperature_step = math.nan
	total_pressure = math.nan
	external_pressure = math.nan
	kinetic_pressure = math.nan
	pullay_stress = math.nan
	cell_volume = math.nan

	for line in block_lines:
		if "free  energy ML TOTEN" in line:
			match = re.search(r"=\s*(%s)\s+eV" % FLOAT_RE, line)
			if match:
				free_energy_ml = parse_float(match.group(1))
		elif "free  energy   TOTEN" in line:
			match = re.search(r"=\s*(%s)\s+eV" % FLOAT_RE, line)
			if match:
				free_energy_dft = parse_float(match.group(1))
		elif "energy  without entropy=" in line:
			match = re.search(r"entropy=\s*(%s)" % FLOAT_RE, line)
			if match:
				internal_energy = parse_float(match.group(1))
		elif "total energy   ETOTAL" in line:
			match = re.search(r"=\s*(%s)\s+eV" % FLOAT_RE, line)
			if match:
				total_energy = parse_float(match.group(1))
		elif "(temperature" in line:
			match = re.search(r"temperature\s+(%s)\s+K" % FLOAT_RE, line)
			if match:
				temperature_step = parse_float(match.group(1))
		elif "external pressure =" in line and "Pullay stress =" in line:
			match = re.search(r"external pressure =\s*(%s)\s+kB\s+Pullay stress =\s*(%s)\s+kB" % (FLOAT_RE, FLOAT_RE), line)
			if match:
				external_pressure = parse_float(match.group(1))
				pullay_stress = parse_float(match.group(2))
		elif "kinetic pressure (ideal gas correction)" in line:
			match = re.search(r"=\s*(%s)\s+kB" % FLOAT_RE, line)
			if match:
				kinetic_pressure = parse_float(match.group(1))
		elif "total pressure  =" in line:
			match = re.search(r"=\s*(%s)\s+kB" % FLOAT_RE, line)
			if match:
				total_pressure = parse_float(match.group(1))
		elif "Total+kin." in line and not np.isfinite(total_pressure):
			components = [float(value) for value in re.findall(FLOAT_RE, line)]
			if len(components) >= 3:
				total_pressure = float(np.mean(components[:3]))
		elif "volume of cell :" in line:
			match = re.search(r":\s*(%s)" % FLOAT_RE, line)
			if match:
				cell_volume = parse_float(match.group(1))

	selected_free_energy = free_energy_ml
	if (not np.isfinite(selected_free_energy)) or abs(selected_free_energy) < 1e-12:
		selected_free_energy = free_energy_dft
	if not np.isfinite(internal_energy):
		internal_energy = selected_free_energy
	if not np.isfinite(total_energy):
		total_energy = selected_free_energy
	if not np.isfinite(temperature_step):
		temperature_step = temperature_value

	rows.append(
		{
			"step": step_number,
			"internal_energy": internal_energy,
			"total_pressure_kbar": total_pressure,
			"free_energy": selected_free_energy,
			"temperature": temperature_step,
			"total_energy": total_energy,
			"cell_volume": cell_volume,
			"external_pressure_kbar": external_pressure,
			"kinetic_pressure_kbar": kinetic_pressure,
			"pullay_stress_kbar": pullay_stress,
			"free_energy_ml": free_energy_ml,
			"free_energy_dft": free_energy_dft,
		}
	)

# The final MLFF block can be partially written or otherwise inconsistent while
# a run is active, so all downstream MLFF analysis intentionally ignores it.
if len(rows) > 1:
	rows = rows[:-1]

data_path = analysis_dir / "mlff_step_data.tsv"
with data_path.open("w", encoding="utf-8") as handle:
	handle.write(
		"step\tinternal_energy\ttotal_pressure_kbar\tfree_energy\ttemperature\ttotal_energy\tcell_volume\texternal_pressure_kbar\tkinetic_pressure_kbar\tpullay_stress_kbar\tfree_energy_ml\tfree_energy_dft\n"
	)
	for row in rows:
		handle.write(
			"\t".join(
				[
					str(row["step"]),
					format_value(row["internal_energy"]),
					format_value(row["total_pressure_kbar"]),
					format_value(row["free_energy"]),
					format_value(row["temperature"]),
					format_value(row["total_energy"]),
					format_value(row["cell_volume"]),
					format_value(row["external_pressure_kbar"]),
					format_value(row["kinetic_pressure_kbar"]),
					format_value(row["pullay_stress_kbar"]),
					format_value(row["free_energy_ml"]),
					format_value(row["free_energy_dft"]),
				]
			)
			+ "\n"
		)

complete_rows = [
	row
	for row in rows
	if np.isfinite(row["internal_energy"])
	and np.isfinite(row["total_pressure_kbar"])
	and np.isfinite(row["free_energy"])
	and np.isfinite(row["temperature"])
	and np.isfinite(row["total_energy"])
	and np.isfinite(row["cell_volume"])
	and np.isfinite(row["external_pressure_kbar"])
	and np.isfinite(row["kinetic_pressure_kbar"])
	and np.isfinite(row["pullay_stress_kbar"])
]

nlines = len(complete_rows)
start_index = int(nlines / ratio_value) + 1 if nlines else 1
end_index = nlines if nlines else 0

chunk_file = run_dir / "chunkdata.txt"
if chunk_file.is_file():
	chunk_tokens = chunk_file.read_text().split()
	if len(chunk_tokens) >= 2:
		start_index = max(1, int(float(chunk_tokens[0])))
		end_index = min(nlines, int(float(chunk_tokens[1])))

selected_rows = complete_rows[start_index - 1:end_index] if end_index >= start_index else []
selected_count = len(selected_rows)

internal_series = [row["internal_energy"] for row in selected_rows]
pressure_series_raw = [row["total_pressure_kbar"] for row in selected_rows]
pressure_squared_series_raw = [value * value for value in pressure_series_raw]
energy_pressure_series_raw = [row["internal_energy"] * row["total_pressure_kbar"] for row in selected_rows]
free_series = [row["free_energy"] for row in selected_rows]
entropy_term_series = [row["internal_energy"] - row["free_energy"] for row in selected_rows]
temperature_series = [row["temperature"] for row in selected_rows]
total_energy_series = [row["total_energy"] for row in selected_rows]
volume_series = [row["cell_volume"] for row in selected_rows]
external_pressure_series_raw = [row["external_pressure_kbar"] for row in selected_rows]
kinetic_pressure_series_raw = [row["kinetic_pressure_kbar"] for row in selected_rows]
pullay_pressure_series_raw = [row["pullay_stress_kbar"] for row in selected_rows]

E, sE = mean_and_block_error(internal_series)
E2, sE2 = mean_and_block_error([value * value for value in internal_series])
pressure_mean_raw, pressure_error_raw = mean_and_block_error(pressure_series_raw)
P = safe_bc_div(pressure_mean_raw, KBAR_PER_GPA)
sP = safe_bc_div(pressure_error_raw, KBAR_PER_GPA)
P2, sP2 = mean_and_block_error([safe_bc_div(value, KBAR_PER_GPA * KBAR_PER_GPA) for value in pressure_squared_series_raw])
EP, sEP = mean_and_block_error([safe_bc_div(value, KBAR_PER_GPA) for value in energy_pressure_series_raw])
F, sF = mean_and_block_error(free_series)
entropy_mean_raw, entropy_error_raw = mean_and_block_error(entropy_term_series)
S = safe_bc_div(entropy_mean_raw * EV_TO_K, temperature_value * nions_value)
sS = safe_bc_div(entropy_error_raw * EV_TO_K, temperature_value * nions_value)
Tk, sTk = mean_and_block_error(temperature_series)
Q, sQ = mean_and_block_error(total_energy_series)
Vcell, sVcell = mean_and_block_error(volume_series)
external_pressure_mean_raw, external_pressure_error_raw = mean_and_block_error(external_pressure_series_raw)
ExP = safe_bc_div(external_pressure_mean_raw, KBAR_PER_GPA)
sExP = safe_bc_div(external_pressure_error_raw, KBAR_PER_GPA)
kinetic_pressure_mean_raw, kinetic_pressure_error_raw = mean_and_block_error(kinetic_pressure_series_raw)
KP = safe_bc_div(kinetic_pressure_mean_raw, KBAR_PER_GPA)
sKP = safe_bc_div(kinetic_pressure_error_raw, KBAR_PER_GPA)
pullay_pressure_mean_raw, pullay_pressure_error_raw = mean_and_block_error(pullay_pressure_series_raw)
PS = safe_bc_div(pullay_pressure_mean_raw, KBAR_PER_GPA)
sPS = safe_bc_div(pullay_pressure_error_raw, KBAR_PER_GPA)

Vcell_initial = initial_volume
V = Vcell
rho = safe_bc_div(nions_value, V)
Vm = safe_bc_div(V * AVOGADRO_VOLUME, nions_value)

drift = 0.0
if selected_count > 1 and nions_value > 0:
	drift = (selected_rows[-1]["total_energy"] - selected_rows[0]["total_energy"]) / nions_value / selected_count * 1000.0 * 1000.0

kinetic_term = 1.5 * nions_value * temperature_value / EV_TO_K if np.isfinite(temperature_value) else math.nan
Epk = E + kinetic_term if np.isfinite(E) and np.isfinite(kinetic_term) else math.nan
H = E + safe_bc_div(P * V, EVG) + kinetic_term if np.isfinite(E) and np.isfinite(P) and np.isfinite(V) and np.isfinite(kinetic_term) else math.nan
sH = safe_sqrt((sE * sE) + ((safe_bc_div(sP * V, EVG)) ** 2 if np.isfinite(sP) and np.isfinite(V) else math.nan))
EJ = safe_bc_div(Epk * EV_TO_KJMOL, nions_value)
HJ = safe_bc_div(H * EV_TO_KJMOL, nions_value)
sEJ = safe_bc_div(sE * EV_TO_KJMOL, nions_value)
sHJ = safe_bc_div(sH * EV_TO_KJMOL, nions_value)

variance_energy = E2 - (E * E) if np.isfinite(E2) and np.isfinite(E) else math.nan
variance_pressure = P2 - (P * P) if np.isfinite(P2) and np.isfinite(P) else math.nan
CV = safe_bc_div(variance_energy * EV_TO_K * EV_TO_K, nions_value * temperature_value * temperature_value) + 1.5 if np.isfinite(variance_energy) and np.isfinite(temperature_value) and temperature_value != 0 else math.nan
sCV = safe_bc_div(sE * safe_sqrt(variance_energy) * EV_TO_K * EV_TO_K, nions_value * temperature_value * temperature_value) if np.isfinite(sE) and np.isfinite(variance_energy) and np.isfinite(temperature_value) and temperature_value != 0 else math.nan

aKT = math.nan
if all(np.isfinite(value) for value in (EP, E, P, temperature_value, V)) and temperature_value != 0 and nions_value > 0:
	aKT = MPA_PER_GPA * (EP - E * P) / (temperature_value * temperature_value) * EV_TO_K + R_GAS * nions_value / (V * AVOGADRO_VOLUME)

saKT = math.nan
if all(np.isfinite(value) for value in (sE, variance_energy, sP, variance_pressure, EP, E, P, temperature_value)) and variance_energy > 0 and variance_pressure > 0 and temperature_value != 0:
	saKT = math.sqrt(math.sqrt((sE * sE) / variance_energy * (sP * sP) / variance_pressure)) * MPA_PER_GPA * (EP - E * P) / (temperature_value * temperature_value) * EV_TO_K

gam = math.nan
if all(np.isfinite(value) for value in (aKT, CV, V)) and CV != 0 and nions_value > 0:
	gam = aKT / (CV * R_GAS * nions_value) * (V * AVOGADRO_VOLUME)

sgam = math.nan
if all(np.isfinite(value) for value in (gam, saKT, aKT, sCV, CV)) and aKT != 0 and CV != 0:
	sgam = abs(gam) * math.sqrt((saKT / aKT) ** 2 + (sCV / CV) ** 2)

ecor = safe_bc_div((sE * sE) * selected_count, variance_energy) if np.isfinite(variance_energy) and variance_energy != 0 else math.nan
pcor = safe_bc_div((sP * sP) * selected_count, variance_pressure) if np.isfinite(variance_pressure) and variance_pressure != 0 else math.nan
cubic_cell_size = math.exp(math.log(Vcell) / 3.0) if np.isfinite(Vcell) and Vcell > 0 else math.nan

peavg_lines = [
	("Temperature", temperature_value, "K"),
	("Computed temperature", Tk, f"+- {format_value(sTk)} K"),
	("Number", float(nions_value), ""),
	("Number density", rho, "atom/A^3"),
	("Mass density", mass_density, "g/cm^3"),
	("Volume", Vm, "cm^3/mol-atom"),
	("Conserved Quantity", Q, f"eV, drift {format_value(drift)} meV/atom/ps"),
	("Internal Energy", E, f"+- {format_value(sE)} eV"),
	("Internal Energy (incl. kinetic)", Epk, f"+- {format_value(sE)} eV"),
	("Enthalpy (incl. kinetic)", H, f"+- {format_value(sH)} eV"),
	("E-TS_el", F, f"+- {format_value(sF)} eV"),
	("Electronic entropy", S, f"+- {format_value(sS)} Nk_B"),
	("Internal Energy (incl. kinetic)", EJ, f"+- {format_value(sEJ)} kJ/mol-atom"),
	("Enthalpy (incl. kinetic)", HJ, f"+- {format_value(sHJ)} kJ/mol-atom"),
	("Pressure", P, f"+- {format_value(sP)} GPa"),
	("CV/Nk", CV, f"+- {format_value(sCV)}"),
	("alpha*K_T", aKT, f"+- {format_value(saKT)} MPa/K"),
	("Gruneisen parameter", gam, f"+- {format_value(sgam)}"),
	("Time Steps", float(selected_count), f"Ratio = {format_value(ratio_value)} E Correlation Length = {format_value(ecor)} P Correlation Length = {format_value(pcor)}"),
	("Free energy (TOTEN)", F, f"+- {format_value(sF)} eV"),
	("Volume of cell", Vcell, f"+- {format_value(sVcell)} A^3 ( initial: {format_value(Vcell_initial)} )"),
	("Cubic cell size", cubic_cell_size, "A"),
	("External pressure", ExP, f"+- {format_value(sExP)} GPa"),
	("Kinetic pressure", KP, f"+- {format_value(sKP)} GPa"),
	("Pullay stress", PS, f"+- {format_value(sPS)} GPa"),
	("Total time steps", float(nlines), ""),
]

with (analysis_dir / "peavg.out").open("w", encoding="utf-8") as handle:
	for label, value, suffix in peavg_lines:
		handle.write(f"{label} = {format_value(value)} {suffix}".rstrip() + "\n")

numbers = [
	temperature_value,
	Tk,
	sTk,
	float(nions_value),
	rho,
	mass_density,
	Vm,
	Q,
	drift,
	E,
	sE,
	Epk,
	sE,
	H,
	sH,
	F,
	sF,
	S,
	sS,
	EJ,
	sEJ,
	HJ,
	sHJ,
	P,
	sP,
	CV,
	sCV,
	aKT,
	saKT,
	gam,
	sgam,
	float(nlines),
	ratio_value,
	ecor,
	pcor,
	F,
	sF,
	Vcell,
	sVcell,
	cubic_cell_size,
	ExP,
	sExP,
	KP,
	sKP,
	PS,
	sPS,
]

with (analysis_dir / "peavg_numbers.out").open("w", encoding="utf-8") as handle:
	for value in numbers:
		handle.write(format_value(value) + "\n")
PY
