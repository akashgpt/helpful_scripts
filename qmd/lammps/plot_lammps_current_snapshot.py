#!/usr/bin/env python3
"""Plot a current structure snapshot from a LAMMPS dump or data file.

Examples:
	python plot_lammps_current_snapshot.py
	python plot_lammps_current_snapshot.py -f npt.dump -t -1 --elements H N
	python plot_lammps_current_snapshot.py -f npt.dump -t 4000 --dt-ps 0.0005
	python plot_lammps_current_snapshot.py -f conf.lmp --format lammps-data --elements H N
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.getuid()}"))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


@dataclass(frozen=True)
class LammpsSnapshot:
	"""Store structural data needed for a quick LAMMPS snapshot plot."""

	title: str
	lattice: np.ndarray
	species: list[str]
	counts: list[int]
	positions: np.ndarray
	symbols: list[str]
	frame_index: int | None
	total_steps: int | None
	timestep: str | None
	dt_ps: float | None
	time_ps: float | None
	temperature_k: float | None
	pressure_bar: float | None
	pressure_label: str | None
	source_format: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments.

	Args:
		argv: Optional argument vector. If None, argparse reads from sys.argv.

	Returns:
		Parsed command-line arguments.
	"""
	parser = argparse.ArgumentParser(
		description=(
			"Create a species-colored snapshot PNG from a LAMMPS trajectory dump "
			"or LAMMPS data file."
		),
	)
	parser.add_argument(
		"-f",
		"--file",
		"--structure",
		dest="structure",
		type=Path,
		default=Path("npt.dump"),
		help="LAMMPS dump or data file to read. Default: npt.dump.",
	)
	parser.add_argument(
		"--format",
		choices=("auto", "lammps-dump-text", "lammps-data"),
		default="auto",
		help="ASE input format. Default: auto.",
	)
	parser.add_argument(
		"-t",
		"--timestep",
		"--index",
		dest="frame_selector",
		default="-1",
		help=(
			"LAMMPS timestep value or frame index to plot. Default: -1, meaning "
			"the last frame/time step in the dump."
		),
	)
	parser.add_argument(
		"--elements",
		nargs="+",
		help=(
			"Element symbols ordered by LAMMPS type id, e.g. '--elements H N'. "
			"Use this when the dump stores numeric atom types rather than element labels."
		),
	)
	parser.add_argument(
		"--lammps-data-style",
		default="atomic",
		help="LAMMPS data style passed to ASE for lammps-data files. Default: atomic.",
	)
	parser.add_argument(
		"--dt-ps",
		type=float,
		help=(
			"LAMMPS timestep size in ps. If omitted, the script reads variable TIMESTEP "
			"from --lammps-input when available."
		),
	)
	parser.add_argument(
		"--lammps-input",
		type=Path,
		help=(
			"LAMMPS input file used to read variable TIMESTEP. Default: "
			"in.lammps_npt_eq next to the dump, or the first in.lammps* file found."
		),
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("analysis/current_lammps_snapshot.png"),
		help="Output image path. Default: analysis/current_lammps_snapshot.png.",
	)
	parser.add_argument("--title", default="", help="Optional plot title.")
	parser.add_argument(
		"--no-wrap",
		action="store_true",
		help="Do not wrap positions back into the simulation cell before plotting.",
	)
	return parser.parse_args(argv)


def infer_lammps_format(structure_path: Path, requested_format: str) -> str:
	"""Infer the ASE LAMMPS format for a structure file.

	Args:
		structure_path: Input structure path.
		requested_format: Explicit requested format or ``auto``.

	Returns:
		ASE format string.

	Raises:
		ValueError: If the format cannot be inferred.
	"""
	if requested_format != "auto":
		return requested_format
	with structure_path.open("r", encoding="utf-8", errors="replace") as structure_file:
		first_lines = [structure_file.readline() for _ in range(8)]
	if any(line.strip() == "ITEM: TIMESTEP" for line in first_lines):
		return "lammps-dump-text"
	if structure_path.name.endswith((".dump", ".lammpstrj")):
		return "lammps-dump-text"
	if structure_path.name.endswith((".lmp", ".data")) or "conf.lmp" in structure_path.name:
		return "lammps-data"
	raise ValueError(
		f"Could not infer LAMMPS format for {structure_path}. "
		"Pass --format lammps-dump-text or --format lammps-data."
	)


def parse_frame_selector(selector_text: str) -> int | str:
	"""Parse a timestep/frame selector.

	Args:
		selector_text: User-provided timestep or frame-index text.

	Returns:
		Integer selector when possible, otherwise raw selector text.
	"""
	try:
		return int(selector_text)
	except ValueError:
		return selector_text


def read_dump_timesteps(dump_path: Path) -> list[str]:
	"""Read all timestep values from a LAMMPS text dump.

	Args:
		dump_path: LAMMPS dump path.

	Returns:
		Timestep values as strings, ordered by frame.
	"""
	timesteps: list[str] = []
	expect_timestep = False
	with dump_path.open("r", encoding="utf-8", errors="replace") as dump_file:
		for line in dump_file:
			if expect_timestep:
				timesteps.append(line.strip())
				expect_timestep = False
				continue
			if line.strip() == "ITEM: TIMESTEP":
				expect_timestep = True
	return timesteps


def calculate_total_elapsed_steps(step_labels: list[str]) -> int | None:
	"""Calculate elapsed MD steps from ordered numeric timestep labels.

	Args:
		step_labels: Ordered LAMMPS timestep labels.

	Returns:
		Elapsed timestep count, or None when labels are unavailable/non-numeric.
	"""
	if not step_labels:
		return None
	try:
		numeric_labels = [int(label) for label in step_labels]
	except ValueError:
		return None
	return numeric_labels[-1] - numeric_labels[0]


def normalize_frame_index(frame_index: int | str, num_frames: int) -> int | None:
	"""Normalize an ASE frame index to a non-negative integer when possible.

	Args:
		frame_index: Requested frame index.
		num_frames: Number of frames in the trajectory.

	Returns:
		Normalized frame index, or None for non-integer/slice inputs.
	"""
	if not isinstance(frame_index, int):
		return None
	if frame_index < 0:
		return num_frames + frame_index
	return frame_index


def resolve_dump_frame_selector(
	frame_selector: int | str,
	timesteps: list[str],
) -> tuple[int | str, int | None, str | None]:
	"""Resolve a timestep/frame selector for a LAMMPS dump.

	Selection order for integer selectors is:

	1. ``-1`` and other negative values are treated as Python frame indices.
	2. Non-negative values that match a LAMMPS ``ITEM: TIMESTEP`` value select
		that physical timestep.
	3. Non-negative values that do not match a timestep are treated as frame
		indices, matching ASE's indexing convention.

	Args:
		frame_selector: User-provided selector.
		timesteps: Timestep values from the dump.

	Returns:
		ASE frame index, normalized non-negative frame index when known, and the
		selected LAMMPS timestep value when known.

	Raises:
		ValueError: If the selected timestep or frame index is out of range.
	"""
	num_frames = len(timesteps)
	if num_frames == 0:
		raise ValueError("No ITEM: TIMESTEP frames found in the LAMMPS dump")
	if isinstance(frame_selector, int):
		if frame_selector < 0:
			normalized_index = num_frames + frame_selector
			if normalized_index < 0 or normalized_index >= num_frames:
				raise ValueError(f"Frame selector {frame_selector} is out of range for {num_frames} frames")
			return frame_selector, normalized_index, timesteps[normalized_index]
		selector_text = str(frame_selector)
		if selector_text in timesteps:
			normalized_index = timesteps.index(selector_text)
			return normalized_index, normalized_index, timesteps[normalized_index]
		if frame_selector < num_frames:
			return frame_selector, frame_selector, timesteps[frame_selector]
		raise ValueError(
			f"Selector {frame_selector} is neither a timestep in the dump nor a valid frame index "
			f"for {num_frames} frames"
		)

	selector_text = str(frame_selector)
	if selector_text in timesteps:
		normalized_index = timesteps.index(selector_text)
		return normalized_index, normalized_index, timesteps[normalized_index]
	return frame_selector, None, None


def apply_element_map(symbols_or_types: list[int], elements: list[str]) -> list[str]:
	"""Map one-based LAMMPS type ids to element symbols.

	Args:
		symbols_or_types: Atomic numbers or type ids from ASE.
		elements: Element symbols ordered by LAMMPS type id.

	Returns:
		Element symbol for each atom.

	Raises:
		ValueError: If a type id cannot be mapped.
	"""
	mapped_symbols: list[str] = []
	for type_id in symbols_or_types:
		if type_id < 1 or type_id > len(elements):
			raise ValueError(
				f"LAMMPS type id {type_id} cannot be mapped by --elements {elements}"
			)
		mapped_symbols.append(elements[type_id - 1])
	return mapped_symbols


def choose_symbols(atoms: object, elements: list[str] | None) -> list[str]:
	"""Choose chemical symbols for a LAMMPS snapshot.

	Args:
		atoms: ASE Atoms object.
		elements: Optional type-id-to-element mapping.

	Returns:
		Chemical symbols for each atom.
	"""
	if elements:
		type_array = None
		for key in ("type", "types", "atom_type"):
			if key in atoms.arrays:
				type_array = atoms.arrays[key]
				break
		if type_array is None:
			type_array = atoms.get_atomic_numbers()
		return apply_element_map([int(value) for value in type_array], elements)
	return list(atoms.get_chemical_symbols())


def ordered_species_counts(symbols: list[str]) -> tuple[list[str], list[int]]:
	"""Count species while preserving first-seen order.

	Args:
		symbols: Chemical symbol per atom.

	Returns:
		Species order and counts in that order.
	"""
	counter = Counter(symbols)
	species: list[str] = []
	for symbol in symbols:
		if symbol not in species:
			species.append(symbol)
	return species, [counter[symbol] for symbol in species]


def find_lammps_input_path(
	structure_path: Path,
	explicit_lammps_input: Path | None,
) -> Path | None:
	"""Find a LAMMPS input file near the structure file.

	Args:
		structure_path: LAMMPS dump or data path.
		explicit_lammps_input: User-provided LAMMPS input file, if any.

	Returns:
		LAMMPS input path when one is available, otherwise None.
	"""
	if explicit_lammps_input is not None:
		return explicit_lammps_input
	preferred_path = structure_path.resolve().parent / "in.lammps_npt_eq"
	if preferred_path.is_file():
		return preferred_path
	candidates = sorted(structure_path.resolve().parent.glob("in.lammps*"))
	if candidates:
		return candidates[0]
	return None


def read_lammps_dt_ps(lammps_input_path: Path | None) -> float | None:
	"""Read ``variable TIMESTEP equal ...`` from a LAMMPS input file.

	Args:
		lammps_input_path: LAMMPS input path, if available.

	Returns:
		Timestep size in ps, or None when unavailable.
	"""
	if lammps_input_path is None or not lammps_input_path.is_file():
		return None
	pattern = re.compile(r"^\s*variable\s+TIMESTEP\s+equal\s+([-+0-9.eE]+)", re.IGNORECASE)
	with lammps_input_path.open("r", encoding="utf-8", errors="replace") as lammps_input:
		for line in lammps_input:
			match = pattern.search(line)
			if match:
				return float(match.group(1))
	return None


def read_lammps_conditions(lammps_input_path: Path | None) -> tuple[float | None, float | None, str | None]:
	"""Read target temperature and pressure from a LAMMPS input file.

	Args:
		lammps_input_path: LAMMPS input path, if available.

	Returns:
		Tuple of temperature in K, pressure in bar, and pressure label. Missing
		values are returned as None.
	"""
	if lammps_input_path is None or not lammps_input_path.is_file():
		return None, None, None
	pattern = re.compile(
		r"^\s*variable\s+(\S+)\s+equal\s+([-+0-9.eE]+)",
		re.IGNORECASE,
	)
	variables: dict[str, float] = {}
	with lammps_input_path.open("r", encoding="utf-8", errors="replace") as lammps_input:
		for line in lammps_input:
			match = pattern.search(line)
			if match:
				variables[match.group(1).upper()] = float(match.group(2))
	temperature_k = variables.get("TEMP")
	for variable_name, pressure_label in (
		("PZ", "P"),
		("PRES", "P"),
		("PRESSURE", "P"),
		("P", "P"),
	):
		if variable_name in variables:
			return temperature_k, variables[variable_name], pressure_label
	return temperature_k, None, None


def calculate_time_ps(timestep: str | None, dt_ps: float | None) -> float | None:
	"""Calculate physical time from a LAMMPS timestep value and dt.

	Args:
		timestep: LAMMPS timestep value from the dump.
		dt_ps: LAMMPS timestep size in ps.

	Returns:
		Physical time in ps, or None when unavailable.
	"""
	if timestep is None or dt_ps is None:
		return None
	try:
		return float(timestep) * dt_ps
	except ValueError:
		return None


def read_lammps_snapshot(
	structure_path: Path,
	input_format: str,
	frame_selector: int | str,
	elements: list[str] | None,
	lammps_data_style: str,
	wrap_positions: bool,
	dt_ps: float | None,
	temperature_k: float | None,
	pressure_bar: float | None,
	pressure_label: str | None,
) -> LammpsSnapshot:
	"""Read one LAMMPS snapshot through ASE.

	Args:
		structure_path: LAMMPS dump or data path.
		input_format: ASE input format.
		frame_selector: LAMMPS timestep or frame index for dump trajectories.
		elements: Optional type-id-to-element mapping.
		lammps_data_style: LAMMPS data style for ``lammps-data`` reads.
		wrap_positions: Whether to wrap positions into the simulation cell.
		dt_ps: LAMMPS timestep size in ps, if available.
		temperature_k: Target temperature in K, if available.
		pressure_bar: Target pressure in bar, if available.
		pressure_label: Label for the pressure variable, if available.

	Returns:
		Parsed LAMMPS snapshot.

	Raises:
		FileNotFoundError: If the input file is missing.
		ValueError: If ASE cannot read a single frame.
	"""
	if not structure_path.is_file():
		raise FileNotFoundError(f"LAMMPS structure file not found: {structure_path}")

	from ase.io import read

	if input_format == "lammps-data":
		atoms = read(
			str(structure_path),
			format="lammps-data",
			style=lammps_data_style,
		)
		normalized_index: int | None = 0
		total_steps: int | None = None
		timestep = None
	else:
		timesteps = read_dump_timesteps(structure_path)
		total_steps = calculate_total_elapsed_steps(timesteps)
		ase_frame_index, normalized_index, timestep = resolve_dump_frame_selector(
			frame_selector,
			timesteps,
		)
		atoms = read(str(structure_path), format="lammps-dump-text", index=ase_frame_index)
		if isinstance(atoms, list):
			if len(atoms) != 1:
				raise ValueError("Snapshot plotting requires one frame; pass a single -t/--timestep.")
			atoms = atoms[0]

	atoms = atoms.copy()
	cell_disp = np.asarray(atoms.get_celldisp(), dtype=float)
	if cell_disp.shape == (3,) and np.any(np.abs(cell_disp) > 0.0):
		atoms.positions -= cell_disp
	if wrap_positions:
		atoms.wrap()

	symbols = choose_symbols(atoms, elements)
	atoms.set_chemical_symbols(symbols)
	species, counts = ordered_species_counts(symbols)
	lattice = np.asarray(atoms.get_cell().array, dtype=float)
	if lattice.shape != (3, 3) or not np.any(np.abs(lattice) > 0.0):
		raise ValueError(f"No usable simulation cell found in {structure_path}")

	return LammpsSnapshot(
		title=structure_path.name,
		lattice=lattice,
		species=species,
		counts=counts,
		positions=np.asarray(atoms.get_positions(), dtype=float),
		symbols=symbols,
		frame_index=normalized_index,
		total_steps=total_steps,
		timestep=timestep,
		dt_ps=dt_ps,
		time_ps=calculate_time_ps(timestep, dt_ps),
		temperature_k=temperature_k,
		pressure_bar=pressure_bar,
		pressure_label=pressure_label,
		source_format=input_format,
	)


def build_species_color_map(species_order: list[str]) -> dict[str, str]:
	"""Assign chemistry-aware colors to species labels.

	Args:
		species_order: Species labels in plotting order.

	Returns:
		Color per species.
	"""
	element_colors: dict[str, str] = {
		"H": "#f8fafc",
		"C": "#4b5563",
		"N": "#2563eb",
		"O": "#dc2626",
		"S": "#facc15",
		"Mg": "#22c55e",
		"Si": "#d6a15f",
		"Fe": "#b45309",
	}
	fallback_palette = [
		"#7c3aed",
		"#0891b2",
		"#db2777",
		"#65a30d",
		"#f97316",
		"#64748b",
	]
	return {
		species: element_colors.get(
			species,
			fallback_palette[index % len(fallback_palette)],
		)
		for index, species in enumerate(species_order)
	}


def build_species_size_map(species_order: list[str], num_atoms: int) -> dict[str, float]:
	"""Assign marker sizes that remain readable for small and large systems.

	Args:
		species_order: Species labels in plotting order.
		num_atoms: Total atom count in the snapshot.

	Returns:
		Marker size per species.
	"""
	element_sizes: dict[str, float] = {
		"H": 18.0,
		"C": 34.0,
		"N": 40.0,
		"O": 42.0,
		"S": 62.0,
		"Mg": 68.0,
		"Si": 64.0,
		"Fe": 70.0,
	}
	scale = min(1.0, max(0.12, 850.0 / max(float(num_atoms), 1.0)))
	return {
		species: element_sizes.get(species, 46.0) * scale
		for species in species_order
	}


def cell_vertices(lattice: np.ndarray) -> np.ndarray:
	"""Return the eight Cartesian vertices of a simulation cell.

	Args:
		lattice: 3x3 lattice matrix.

	Returns:
		Cell vertices.
	"""
	a, b, c = lattice
	return np.array(
		[
			[0.0, 0.0, 0.0],
			a,
			b,
			c,
			a + b,
			a + c,
			b + c,
			a + b + c,
		],
		dtype=float,
	)


def draw_cell_wireframe(axis: Axes3D, lattice: np.ndarray) -> None:
	"""Draw a thin simulation-cell wireframe.

	Args:
		axis: Matplotlib 3D axis.
		lattice: 3x3 lattice matrix.
	"""
	vertices = cell_vertices(lattice)
	edge_pairs = [
		(0, 1),
		(0, 2),
		(0, 3),
		(1, 4),
		(1, 5),
		(2, 4),
		(2, 6),
		(3, 5),
		(3, 6),
		(4, 7),
		(5, 7),
		(6, 7),
	]
	for start, end in edge_pairs:
		points = vertices[[start, end]]
		axis.plot(
			points[:, 0],
			points[:, 1],
			points[:, 2],
			color="#111827",
			linewidth=1.2,
			alpha=0.70,
		)


def draw_side_cell_outline(axis: plt.Axes, lattice: np.ndarray) -> None:
	"""Draw a side-view x-z projection of the simulation-cell wireframe.

	Args:
		axis: Matplotlib axis.
		lattice: 3x3 lattice matrix.
	"""
	vertices = cell_vertices(lattice)
	edge_pairs = [
		(0, 1),
		(0, 2),
		(0, 3),
		(1, 4),
		(1, 5),
		(2, 4),
		(2, 6),
		(3, 5),
		(3, 6),
		(4, 7),
		(5, 7),
		(6, 7),
	]
	for start, end in edge_pairs:
		points = vertices[[start, end]]
		axis.plot(
			points[:, 0],
			points[:, 2],
			color="#111827",
			linewidth=1.0,
			alpha=0.55,
		)


def set_equal_3d_limits(axis: Axes3D, lattice: np.ndarray) -> None:
	"""Set equal-ish 3D limits around the cell vertices.

	Args:
		axis: Matplotlib 3D axis.
		lattice: 3x3 lattice matrix.
	"""
	vertices = cell_vertices(lattice)
	minima = np.min(vertices, axis=0)
	maxima = np.max(vertices, axis=0)
	centers = 0.5 * (minima + maxima)
	spans = maxima - minima
	max_span = float(max(np.max(spans), 1.0))
	padding = 0.04 * max_span
	for setter, center in zip(
		(axis.set_xlim, axis.set_ylim, axis.set_zlim),
		centers,
	):
		setter(center - 0.5 * max_span - padding, center + 0.5 * max_span + padding)
	axis.set_box_aspect((1.0, 1.0, 1.0))


def style_ovito_like_axis(axis: Axes3D) -> None:
	"""Apply a clean OVITO-like 3D viewport style.

	Args:
		axis: Matplotlib 3D axis.
	"""
	axis.view_init(elev=18.0, azim=-52.0)
	axis.set_proj_type("persp")
	axis.set_xlabel("x (A)", labelpad=8)
	axis.set_ylabel("y (A)", labelpad=8)
	axis.set_zlabel("z (A)", labelpad=8)
	axis.grid(False)
	for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
		pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
		pane.set_edgecolor((0.85, 0.85, 0.85, 0.8))
	axis.tick_params(labelsize=8, pad=2)


def set_side_limits(axis: plt.Axes, lattice: np.ndarray) -> None:
	"""Set side-view x-z limits from projected cell vertices.

	Args:
		axis: Matplotlib axis.
		lattice: 3x3 lattice matrix.
	"""
	vertices = cell_vertices(lattice)
	x_min = float(np.min(vertices[:, 0]))
	x_max = float(np.max(vertices[:, 0]))
	z_min = float(np.min(vertices[:, 2]))
	z_max = float(np.max(vertices[:, 2]))
	padding = 0.04 * max(x_max - x_min, z_max - z_min, 1.0)
	axis.set_xlim(x_min - padding, x_max + padding)
	axis.set_ylim(z_min - padding, z_max + padding)
	axis.set_aspect("equal", adjustable="box")


def plot_side_view(
	axis: plt.Axes,
	snapshot: LammpsSnapshot,
	color_map: dict[str, str],
	size_map: dict[str, float],
) -> None:
	"""Plot an x-z side view of the current LAMMPS snapshot.

	Args:
		axis: Matplotlib axis.
		snapshot: Parsed LAMMPS snapshot.
		color_map: Color per species.
		size_map: Marker size per species.
	"""
	draw_side_cell_outline(axis, snapshot.lattice)
	for species in snapshot.species:
		indices = [
			index for index, symbol in enumerate(snapshot.symbols) if symbol == species
		]
		if not indices:
			continue
		axis.scatter(
			snapshot.positions[indices, 0],
			snapshot.positions[indices, 2],
			s=size_map[species] * 0.45,
			c=color_map[species],
			label=species,
			alpha=0.82,
			edgecolors="#0f172a",
			linewidths=0.12,
		)
	axis.set_xlabel("x (A)")
	axis.set_ylabel("z (A)")
	axis.set_title("Side view (x-z)", pad=10)
	axis.grid(alpha=0.18)
	set_side_limits(axis, snapshot.lattice)


def format_metadata(snapshot: LammpsSnapshot) -> str:
	"""Format LAMMPS snapshot metadata for the plot subtitle.

	Args:
		snapshot: Parsed LAMMPS snapshot.

	Returns:
		Human-readable metadata string.
	"""
	if snapshot.timestep is not None:
		step_text = f"step: {snapshot.timestep}"
	elif snapshot.frame_index is not None:
		step_text = f"step: {snapshot.frame_index}"
	else:
		step_text = "step: NA"
	if snapshot.total_steps is not None:
		step_text = f"{step_text} ({snapshot.total_steps} total elapsed)"
	time_text = (
		"time: NA"
		if snapshot.time_ps is None
		else f"time: {snapshot.time_ps:g} ps"
	)
	dt_text = (
		"dt: NA"
		if snapshot.dt_ps is None
		else f"dt: {snapshot.dt_ps * 1000.0:g} fs"
	)
	temperature_text = (
		"T: NA"
		if snapshot.temperature_k is None
		else f"T: {snapshot.temperature_k:g} K"
	)
	pressure_text = (
		"P: NA"
		if snapshot.pressure_bar is None
		else f"{snapshot.pressure_label or 'P'}: {snapshot.pressure_bar / 10000.0:g} GPa"
	)
	return (
		f"{step_text} | {time_text} | {dt_text} | {temperature_text} | "
		f"{pressure_text} | format: {snapshot.source_format}"
	)


def plot_snapshot(snapshot: LammpsSnapshot, output_path: Path, title: str) -> None:
	"""Plot an OVITO-like 3D snapshot of a LAMMPS structure.

	Args:
		snapshot: Parsed LAMMPS snapshot.
		output_path: Output PNG path.
		title: Plot title.
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	color_map = build_species_color_map(snapshot.species)
	size_map = build_species_size_map(snapshot.species, len(snapshot.symbols))

	figure = plt.figure(figsize=(13, 7), constrained_layout=False)
	figure.patch.set_facecolor("#f8fafc")
	grid = figure.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
	figure.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.80, wspace=0.18)
	axis = figure.add_subplot(grid[0, 0], projection="3d")
	side_axis = figure.add_subplot(grid[0, 1])
	axis.set_facecolor("#f8fafc")
	side_axis.set_facecolor("#f8fafc")
	draw_cell_wireframe(axis, snapshot.lattice)

	for species in snapshot.species:
		indices = [
			index for index, symbol in enumerate(snapshot.symbols) if symbol == species
		]
		if not indices:
			continue
		axis.scatter(
			snapshot.positions[indices, 0],
			snapshot.positions[indices, 1],
			snapshot.positions[indices, 2],
			s=size_map[species],
			c=color_map[species],
			label=species,
			alpha=0.92,
			edgecolors="#0f172a",
			linewidths=0.18,
			depthshade=True,
		)

	composition = " ".join(
		f"{count}{species}"
		for species, count in zip(snapshot.species, snapshot.counts)
	)
	set_equal_3d_limits(axis, snapshot.lattice)
	style_ovito_like_axis(axis)
	axis.set_title("Perspective view", pad=4)
	plot_side_view(side_axis, snapshot, color_map, size_map)
	figure.suptitle(
		f"{title or f'Current LAMMPS snapshot: {composition}'}\n{format_metadata(snapshot)}",
		fontsize=13,
		y=0.965,
	)
	axis.legend(
		loc="upper center",
		bbox_to_anchor=(0.5, 1.00),
		ncol=max(1, len(snapshot.species)),
		frameon=True,
		framealpha=0.92,
	)
	figure.savefig(output_path, dpi=240, facecolor=figure.get_facecolor())
	plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
	"""Create the requested LAMMPS snapshot plot.

	Args:
		argv: Optional argument vector. If None, argparse reads from sys.argv.

	Returns:
		Process exit code.
	"""
	args = parse_args(argv)
	input_format = infer_lammps_format(args.structure, args.format)
	lammps_input_path = find_lammps_input_path(args.structure, args.lammps_input)
	dt_ps = args.dt_ps if args.dt_ps is not None else read_lammps_dt_ps(lammps_input_path)
	temperature_k, pressure_bar, pressure_label = read_lammps_conditions(lammps_input_path)
	snapshot = read_lammps_snapshot(
		structure_path=args.structure,
		input_format=input_format,
		frame_selector=parse_frame_selector(args.frame_selector),
		elements=args.elements,
		lammps_data_style=args.lammps_data_style,
		wrap_positions=not args.no_wrap,
		dt_ps=dt_ps,
		temperature_k=temperature_k,
		pressure_bar=pressure_bar,
		pressure_label=pressure_label,
	)
	plot_snapshot(snapshot, args.output, args.title)
	print(f"Wrote {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
