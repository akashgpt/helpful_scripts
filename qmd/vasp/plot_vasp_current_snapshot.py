#!/usr/bin/env python3
"""Plot the current VASP structure snapshot from a CONTCAR/POSCAR file."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.getuid()}"))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


@dataclass(frozen=True)
class VaspStructure:
	"""Store the structural data needed for a quick snapshot plot."""

	title: str
	lattice: np.ndarray
	species: list[str]
	counts: list[int]
	positions: np.ndarray
	symbols: list[str]


@dataclass(frozen=True)
class SnapshotMetadata:
	"""Store MD time metadata parsed from OUTCAR."""

	step: int | None
	potim_fs: float | None

	@property
	def time_ps(self) -> float | None:
		"""Return MD time in picoseconds when both step and POTIM are known."""
		if self.step is None or self.potim_fs is None:
			return None
		return self.step * self.potim_fs / 1000.0


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Create a species-colored x-z snapshot from a VASP CONTCAR/POSCAR.",
	)
	parser.add_argument("--structure", type=Path, default=Path("CONTCAR"))
	parser.add_argument("--outcar", type=Path, default=Path("OUTCAR"))
	parser.add_argument("--output", type=Path, default=Path("analysis/current_snapshot.png"))
	parser.add_argument("--title", default="", help="Optional plot title.")
	return parser.parse_args()


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
		structure_path: POSCAR/CONTCAR path.

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
		lattice=lattice,
		species=species,
		counts=counts,
		positions=positions,
		symbols=symbols,
	)


def parse_outcar_metadata(outcar_path: Path) -> SnapshotMetadata:
	"""Parse POTIM and current ionic step from an OUTCAR file.

	Args:
		outcar_path: OUTCAR path.

	Returns:
		Parsed snapshot metadata. Missing values are represented as ``None``.
	"""
	if not outcar_path.is_file():
		return SnapshotMetadata(step=None, potim_fs=None)

	potim_fs: float | None = None
	ionic_step: int | None = None
	temperature_count = 0
	potim_pattern = re.compile(r"\bPOTIM\s*=\s*([-+0-9.Ee]+)")
	ionic_step_pattern = re.compile(r"Ionic step\s+([0-9]+)")

	for line in outcar_path.read_text(errors="ignore").splitlines():
		potim_match = potim_pattern.search(line)
		if potim_match:
			potim_fs = float(potim_match.group(1))
		ionic_step_match = ionic_step_pattern.search(line)
		if ionic_step_match:
			ionic_step = int(ionic_step_match.group(1))
		if "(temperature" in line:
			temperature_count += 1

	if ionic_step is None and temperature_count > 0:
		ionic_step = temperature_count
	return SnapshotMetadata(step=ionic_step, potim_fs=potim_fs)


def format_metadata(metadata: SnapshotMetadata) -> str:
	"""Format MD step/time metadata for the plot subtitle."""
	step_text = "step: NA" if metadata.step is None else f"step: {metadata.step}"
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


def plot_snapshot(
	structure: VaspStructure,
	metadata: SnapshotMetadata,
	output_path: Path,
	title: str,
) -> None:
	"""Plot an OVITO-like 3D snapshot of the current structure.

	Args:
		structure: Parsed VASP structure.
		metadata: MD time metadata parsed from OUTCAR.
		output_path: Output PNG path.
		title: Plot title.
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	color_map = build_species_color_map(structure.species)
	size_map = build_species_size_map(structure.species)

	figure = plt.figure(figsize=(13, 7), constrained_layout=True)
	figure.patch.set_facecolor("#f8fafc")
	grid = figure.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
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

	composition = " ".join(
		f"{count}{species}" for species, count in zip(structure.species, structure.counts)
	)
	set_equal_3d_limits(axis, structure.lattice)
	style_ovito_like_axis(axis)
	axis.set_title("Perspective view", pad=10)
	plot_side_view(side_axis, structure, color_map, size_map)
	figure.suptitle(
		f"{title or f'Current CONTCAR snapshot: {composition}'}\n{format_metadata(metadata)}",
		fontsize=13,
		y=1.02,
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


def main() -> None:
	"""Create the requested structure snapshot plot."""
	args = parse_args()
	structure = parse_structure(args.structure)
	metadata = parse_outcar_metadata(args.outcar)
	plot_snapshot(structure, metadata, args.output, args.title)


if __name__ == "__main__":
	main()
