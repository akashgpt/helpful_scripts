#!/usr/bin/env python3
"""Plot VASP structure snapshots from CONTCAR/POSCAR files or XDATCAR trajectories.

Examples:
	python plot_vasp_current_snapshot.py
	python plot_vasp_current_snapshot.py -f POSCAR
	python plot_vasp_current_snapshot.py -f XDATCAR -t -1
	python plot_vasp_current_snapshot.py -f XDATCAR -t 4000 --outcar OUTCAR
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.getuid()}"))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


@dataclass(frozen=True)
class VaspStructure:
	"""Store VASP structural data needed for a quick snapshot plot."""

	title: str
	source_label: str
	lattice: np.ndarray
	species: list[str]
	counts: list[int]
	positions: np.ndarray
	symbols: list[str]


@dataclass(frozen=True)
class SnapshotMetadata:
	"""Store MD time metadata parsed from OUTCAR or trajectory selection."""

	step: int | None
	total_steps: int | None
	potim_fs: float | None

	@property
	def time_ps(self) -> float | None:
		"""Return MD time in picoseconds when both step and POTIM are known."""
		if self.step is None or self.potim_fs is None:
			return None
		return self.step * self.potim_fs / 1000.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments.

	Args:
		argv: Optional argument vector. If None, argparse reads from sys.argv.

	Returns:
		Parsed command-line arguments.
	"""
	parser = argparse.ArgumentParser(
		description=(
			"Create a species-colored snapshot PNG from a VASP CONTCAR/POSCAR "
			"or an XDATCAR trajectory."
		),
	)
	parser.add_argument(
		"-f",
		"--file",
		"--structure",
		dest="structure",
		type=Path,
		default=Path("CONTCAR"),
		help="VASP structure or trajectory file to read. Default: CONTCAR.",
	)
	parser.add_argument("--outcar", type=Path, default=Path("OUTCAR"))
	parser.add_argument(
		"-t",
		"--timestep",
		"--index",
		dest="frame_selector",
		help=(
			"XDATCAR Direct configuration number or frame index to plot. "
			"Negative values use Python-style indexing; -1 selects the last frame. "
			"If -t is supplied without -f, XDATCAR is used when present."
		),
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("analysis/current_snapshot.png"),
		help="Output image path. Default: analysis/current_snapshot.png.",
	)
	parser.add_argument("--title", default="", help="Optional plot title.")
	return parser.parse_args(argv)


def parse_scale(scale_line: str) -> float:
	"""Parse the POSCAR scale line.

	Args:
		scale_line: The second line of a POSCAR-like file.

	Returns:
		Scalar scale value.

	Raises:
		ValueError: If a three-value or negative-volume scale is encountered.
	"""
	values = [float(token) for token in scale_line.split()]
	if len(values) != 1:
		raise ValueError("Only scalar POSCAR scale factors are supported for snapshots.")
	if values[0] <= 0:
		raise ValueError("Negative POSCAR volume scales are not supported for snapshots.")
	return values[0]


def parse_structure(structure_path: Path) -> VaspStructure:
	"""Parse the first structural block from a VASP structure file.

	Args:
		structure_path: POSCAR/CONTCAR-like structure path.

	Returns:
		Parsed structure.

	Raises:
		FileNotFoundError: If the structure file does not exist.
		ValueError: If the file cannot be parsed.
	"""
	if not structure_path.is_file():
		raise FileNotFoundError(f"Structure file not found: {structure_path}")

	lines = structure_path.read_text(errors="ignore").splitlines()
	if len(lines) < 8:
		raise ValueError(f"Structure file is too short: {structure_path}")

	title = lines[0].strip() or structure_path.name
	scale = parse_scale(lines[1])
	lattice = np.array(
		[[float(value) for value in lines[index].split()[:3]] for index in range(2, 5)],
		dtype=float,
	) * scale

	species_line_index = 5
	counts_line_index = 6
	species = lines[species_line_index].split()
	try:
		counts = [int(token) for token in lines[counts_line_index].split()]
	except ValueError as error:
		raise ValueError("Snapshot helper requires VASP 5 POSCAR species labels.") from error
	if len(species) != len(counts):
		raise ValueError("Species and count lines have different lengths.")

	coord_line_index = 7
	if lines[coord_line_index].strip().lower().startswith("selective"):
		coord_line_index += 1
	coordinate_mode = lines[coord_line_index].strip().lower()
	is_direct = coordinate_mode.startswith("d")
	is_cartesian = coordinate_mode.startswith(("c", "k"))
	if not is_direct and not is_cartesian:
		raise ValueError(f"Unknown coordinate mode: {lines[coord_line_index]}")

	total_atoms = sum(counts)
	position_start = coord_line_index + 1
	position_end = position_start + total_atoms
	if len(lines) < position_end:
		raise ValueError("Structure file ended before all atomic positions were read.")

	raw_positions = np.array(
		[
			[float(value) for value in lines[index].split()[:3]]
			for index in range(position_start, position_end)
		],
		dtype=float,
	)
	if is_direct:
		positions = raw_positions @ lattice
	else:
		positions = raw_positions * scale

	symbols: list[str] = []
	for species_label, count in zip(species, counts):
		symbols.extend([species_label] * count)

	return VaspStructure(
		title=title,
		source_label=structure_path.name,
		lattice=lattice,
		species=species,
		counts=counts,
		positions=positions,
		symbols=symbols,
	)


def parse_frame_selector(selector_text: str) -> int | str:
	"""Parse a VASP frame selector.

	Args:
		selector_text: User-provided timestep/configuration or frame-index text.

	Returns:
		Integer selector when possible, otherwise raw selector text.
	"""
	try:
		return int(selector_text)
	except ValueError:
		return selector_text


def is_xdatcar_path(structure_path: Path) -> bool:
	"""Return whether a path looks like a VASP XDATCAR trajectory.

	Args:
		structure_path: Candidate VASP structure path.

	Returns:
		True when the path name or file contents indicate XDATCAR format.
	"""
	if structure_path.name.upper().startswith("XDATCAR"):
		return True
	if not structure_path.is_file():
		return False
	with structure_path.open("r", encoding="utf-8", errors="replace") as structure_file:
		for _ in range(32):
			line = structure_file.readline()
			if not line:
				break
			if line.strip().lower().startswith("direct configuration="):
				return True
	return False


def read_xdatcar_frame_labels(xdatcar_lines: list[str]) -> list[tuple[int, str]]:
	"""Read frame start indices and Direct-configuration labels from XDATCAR lines.

	Args:
		xdatcar_lines: Full XDATCAR contents split into lines.

	Returns:
		Pairs of line index and configuration label, ordered by trajectory frame.
	"""
	frames: list[tuple[int, str]] = []
	for index, line in enumerate(xdatcar_lines):
		stripped = line.strip()
		if not stripped.lower().startswith("direct configuration="):
			continue
		label = stripped.split("=", 1)[1].strip()
		frames.append((index, label))
	return frames


def resolve_xdatcar_frame_selector(
	frame_selector: int | str,
	frame_labels: list[str],
) -> tuple[int, int | None]:
	"""Resolve an XDATCAR configuration/frame selector.

	Integer selection follows the LAMMPS snapshot helper convention: negative
	values are Python frame indices; non-negative values first match the VASP
	``Direct configuration=`` label; otherwise they are treated as zero-based
	frame indices.

	Args:
		frame_selector: User-provided selector.
		frame_labels: XDATCAR ``Direct configuration=`` labels.

	Returns:
		Selected zero-based frame index and integer configuration label when known.

	Raises:
		ValueError: If the requested selector is out of range or absent.
	"""
	num_frames = len(frame_labels)
	if num_frames == 0:
		raise ValueError("No 'Direct configuration=' frames found in XDATCAR")
	if isinstance(frame_selector, int):
		if frame_selector < 0:
			frame_index = num_frames + frame_selector
			if frame_index < 0 or frame_index >= num_frames:
				raise ValueError(f"Frame selector {frame_selector} is out of range for {num_frames} frames")
			return frame_index, int(frame_labels[frame_index]) if frame_labels[frame_index].isdigit() else None
		selector_text = str(frame_selector)
		if selector_text in frame_labels:
			frame_index = frame_labels.index(selector_text)
			return frame_index, int(selector_text)
		if frame_selector < num_frames:
			return frame_selector, int(frame_labels[frame_selector]) if frame_labels[frame_selector].isdigit() else None
		raise ValueError(
			f"Selector {frame_selector} is neither an XDATCAR configuration label nor "
			f"a valid frame index for {num_frames} frames"
		)
	selector_text = str(frame_selector)
	if selector_text in frame_labels:
		frame_index = frame_labels.index(selector_text)
		return frame_index, int(selector_text) if selector_text.isdigit() else None
	raise ValueError(f"XDATCAR configuration label not found: {selector_text}")


def calculate_total_elapsed_steps(step_labels: list[str]) -> int | None:
	"""Calculate elapsed MD steps from ordered numeric step labels.

	Args:
		step_labels: Ordered trajectory step/configuration labels.

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


def parse_xdatcar_structure(
	xdatcar_path: Path,
	frame_selector: int | str,
) -> tuple[VaspStructure, int | None, int]:
	"""Parse one structure frame from a VASP XDATCAR trajectory.

	Args:
		xdatcar_path: XDATCAR path.
		frame_selector: Configuration label or frame index.

	Returns:
		Parsed structure, selected ionic-step/configuration number, and elapsed total steps.

	Raises:
		FileNotFoundError: If XDATCAR does not exist.
		ValueError: If XDATCAR cannot be parsed or the frame is unavailable.
	"""
	if not xdatcar_path.is_file():
		raise FileNotFoundError(f"XDATCAR file not found: {xdatcar_path}")
	lines = xdatcar_path.read_text(errors="ignore").splitlines()
	if len(lines) < 8:
		raise ValueError(f"XDATCAR file is too short: {xdatcar_path}")
	title = lines[0].strip() or xdatcar_path.name
	scale = parse_scale(lines[1])
	lattice = np.array(
		[[float(value) for value in lines[index].split()[:3]] for index in range(2, 5)],
		dtype=float,
	) * scale
	species = lines[5].split()
	try:
		counts = [int(token) for token in lines[6].split()]
	except ValueError as error:
		raise ValueError("Snapshot helper requires VASP 5 XDATCAR species labels.") from error
	if len(species) != len(counts):
		raise ValueError("Species and count lines have different lengths in XDATCAR.")
	total_atoms = sum(counts)
	frames = read_xdatcar_frame_labels(lines)
	frame_labels = [label for _, label in frames]
	frame_index, selected_step = resolve_xdatcar_frame_selector(frame_selector, frame_labels)
	frame_line_index = frames[frame_index][0]
	position_start = frame_line_index + 1
	position_end = position_start + total_atoms
	if len(lines) < position_end:
		raise ValueError("XDATCAR ended before all positions for the selected frame were read.")
	raw_positions = np.array(
		[
			[float(value) for value in lines[index].split()[:3]]
			for index in range(position_start, position_end)
		],
		dtype=float,
	)
	positions = raw_positions @ lattice
	symbols: list[str] = []
	for species_label, count in zip(species, counts):
		symbols.extend([species_label] * count)
	label = frame_labels[frame_index]
	structure = VaspStructure(
		title=title,
		source_label=f"{xdatcar_path.name} configuration {label}",
		lattice=lattice,
		species=species,
		counts=counts,
		positions=positions,
		symbols=symbols,
	)
	return structure, selected_step, calculate_total_elapsed_steps(frame_labels)


def parse_outcar_metadata(outcar_path: Path) -> SnapshotMetadata:
	"""Parse POTIM and current ionic step from an OUTCAR file.

	Args:
		outcar_path: OUTCAR path.

	Returns:
		Parsed snapshot metadata. Missing values are represented as ``None``.
	"""
	if not outcar_path.is_file():
		return SnapshotMetadata(step=None, total_steps=None, potim_fs=None)

	potim_fs: float | None = None
	ionic_step: int | None = None
	temperature_count = 0
	ionic_step_values: list[int] = []
	potim_pattern = re.compile(r"\bPOTIM\s*=\s*([-+0-9.Ee]+)")
	ionic_step_pattern = re.compile(r"Ionic step\s+([0-9]+)")

	for line in outcar_path.read_text(errors="ignore").splitlines():
		potim_match = potim_pattern.search(line)
		if potim_match:
			potim_fs = float(potim_match.group(1))
		ionic_step_match = ionic_step_pattern.search(line)
		if ionic_step_match:
			ionic_step = int(ionic_step_match.group(1))
			ionic_step_values.append(ionic_step)
		if "(temperature" in line:
			temperature_count += 1

	if ionic_step is None and temperature_count > 0:
		ionic_step = temperature_count
	if ionic_step_values:
		total_steps = ionic_step_values[-1] - ionic_step_values[0]
	elif temperature_count > 0:
		total_steps = max(temperature_count - 1, 0)
	else:
		total_steps = None
	return SnapshotMetadata(step=ionic_step, total_steps=total_steps, potim_fs=potim_fs)


def format_metadata(metadata: SnapshotMetadata) -> str:
	"""Format MD step/time metadata for the plot subtitle."""
	if metadata.step is None:
		step_text = "step: NA"
	else:
		step_text = f"step: {metadata.step}"
	if metadata.total_steps is not None:
		step_text = f"{step_text} ({metadata.total_steps} total elapsed)"
	if metadata.potim_fs is None or metadata.time_ps is None:
		return f"{step_text} | time: NA | POTIM: NA"
	return f"{step_text} | time: {metadata.time_ps:.4f} ps | POTIM: {metadata.potim_fs:g} fs"


def build_species_color_map(species_order: list[str]) -> dict[str, str]:
	"""Assign chemistry-aware colors to species labels."""
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
	color_map: dict[str, str] = {}
	for index, species in enumerate(species_order):
		color_map[species] = element_colors.get(
			species,
			fallback_palette[index % len(fallback_palette)],
		)
	return color_map


def build_species_size_map(species_order: list[str]) -> dict[str, float]:
	"""Assign visually useful marker sizes for atom-like 3D dots."""
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
	return {species: element_sizes.get(species, 46.0) for species in species_order}


def cell_vertices(lattice: np.ndarray) -> np.ndarray:
	"""Return the eight Cartesian vertices of a simulation cell."""
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
	"""Draw a thin simulation-cell wireframe."""
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
	"""Draw a side-view x-z projection of the simulation-cell wireframe."""
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
	"""Set equal-ish 3D limits around the cell vertices."""
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
	"""Apply a clean OVITO-like 3D viewport style."""
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
	"""Set side-view x-z limits from projected cell vertices."""
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
	structure: VaspStructure,
	color_map: dict[str, str],
	size_map: dict[str, float],
) -> None:
	"""Plot an x-z side view of the current structure."""
	draw_side_cell_outline(axis, structure.lattice)
	for species in structure.species:
		indices = [
			index for index, symbol in enumerate(structure.symbols) if symbol == species
		]
		if not indices:
			continue
		axis.scatter(
			structure.positions[indices, 0],
			structure.positions[indices, 2],
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
	set_side_limits(axis, structure.lattice)


def format_snapshot_heading(
	structure: VaspStructure,
	metadata: SnapshotMetadata,
	title: str,
) -> str:
	"""Format the figure heading for a VASP snapshot.

	Args:
		structure: Parsed VASP structure.
		metadata: MD time metadata parsed from OUTCAR or XDATCAR selection.
		title: Optional user-supplied title.

	Returns:
		Two-line Matplotlib title text.
	"""
	composition = " ".join(
		f"{count}{species}" for species, count in zip(structure.species, structure.counts)
	)
	main_title = title or f"Current VASP snapshot: {composition}"
	subtitle = f"{format_metadata(metadata)} | source: {structure.source_label}"
	return f"{main_title}\n{subtitle}"


def plot_snapshot(
	structure: VaspStructure,
	metadata: SnapshotMetadata,
	output_path: Path,
	title: str,
) -> None:
	"""Plot an OVITO-like 3D snapshot of the current VASP structure.

	Args:
		structure: Parsed VASP structure.
		metadata: MD time metadata parsed from OUTCAR.
		output_path: Output PNG path.
		title: Plot title.
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	color_map = build_species_color_map(structure.species)
	size_map = build_species_size_map(structure.species)

	figure = plt.figure(figsize=(13, 7), constrained_layout=False)
	figure.patch.set_facecolor("#f8fafc")
	grid = figure.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
	figure.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.80, wspace=0.18)
	axis = figure.add_subplot(grid[0, 0], projection="3d")
	side_axis = figure.add_subplot(grid[0, 1])
	axis.set_facecolor("#f8fafc")
	side_axis.set_facecolor("#f8fafc")
	draw_cell_wireframe(axis, structure.lattice)

	for species in structure.species:
		indices = [
			index for index, symbol in enumerate(structure.symbols) if symbol == species
		]
		if not indices:
			continue
		axis.scatter(
			structure.positions[indices, 0],
			structure.positions[indices, 1],
			structure.positions[indices, 2],
			s=size_map[species],
			c=color_map[species],
			label=species,
			alpha=0.92,
			edgecolors="#0f172a",
			linewidths=0.18,
			depthshade=True,
		)

	set_equal_3d_limits(axis, structure.lattice)
	style_ovito_like_axis(axis)
	axis.set_title("Perspective view", pad=4)
	plot_side_view(side_axis, structure, color_map, size_map)
	figure.suptitle(
		format_snapshot_heading(structure, metadata, title),
		fontsize=13,
		y=0.965,
	)
	axis.legend(
		loc="upper center",
		bbox_to_anchor=(0.5, 1.00),
		ncol=max(1, len(structure.species)),
		frameon=True,
		framealpha=0.92,
	)
	figure.savefig(output_path, dpi=240, facecolor=figure.get_facecolor())
	plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
	"""Create the requested structure snapshot plot.

	Args:
		argv: Optional argument vector. If None, argparse reads from sys.argv.

	Returns:
		Process exit code.
	"""
	args = parse_args(argv)
	structure_path = args.structure
	if (
		args.frame_selector is not None
		and args.structure == Path("CONTCAR")
		and Path("XDATCAR").is_file()
	):
		structure_path = Path("XDATCAR")

	selected_step: int | None = None
	selected_total_steps: int | None = None
	if is_xdatcar_path(structure_path):
		selector = parse_frame_selector(args.frame_selector or "-1")
		structure, selected_step, selected_total_steps = parse_xdatcar_structure(structure_path, selector)
	else:
		if args.frame_selector is not None:
			raise ValueError(
				"-t/--timestep requires an XDATCAR trajectory. "
				"Pass -f XDATCAR, or omit -t for CONTCAR/POSCAR files."
			)
		structure = parse_structure(structure_path)

	metadata = parse_outcar_metadata(args.outcar)
	if selected_step is not None:
		metadata = SnapshotMetadata(
			step=selected_step,
			total_steps=selected_total_steps,
			potim_fs=metadata.potim_fs,
		)
	plot_snapshot(structure, metadata, args.output, args.title)
	print(f"Wrote {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
