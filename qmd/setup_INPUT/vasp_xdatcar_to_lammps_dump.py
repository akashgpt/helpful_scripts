#!/usr/bin/env python3
"""Convert a VASP XDATCAR trajectory to a LAMMPS text dump.

The output matches the local ALCHEMY/LAMMPS convention:

	dump myDUMP all custom ${DUMP_FREQ} npt.dump id type element x y z

Examples:
	python vasp_xdatcar_to_lammps_dump.py
	python vasp_xdatcar_to_lammps_dump.py --elements Mg Si O He
	python vasp_xdatcar_to_lammps_dump.py --xdatcar XDATCAR --outcar OUTCAR --output npt.dump
	module load anaconda3/2025.12; conda activate ase_env
	python $HELP_SCRIPTS/qmd/setup_INPUT/vasp_xdatcar_to_lammps_dump.py --elements Mg Si O He
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LammpsBox:
	"""Store one LAMMPS restricted-triclinic box."""

	xlo_bound: float
	xhi_bound: float
	ylo_bound: float
	yhi_bound: float
	zlo_bound: float
	zhi_bound: float
	xy: float
	xz: float
	yz: float
	restricted_cell: np.ndarray

	@property
	def is_triclinic(self) -> bool:
		"""Return whether the box has nonzero tilt factors."""
		return any(abs(value) > 1.0e-12 for value in (self.xy, self.xz, self.yz))


@dataclass(frozen=True)
class CellSelection:
	"""Store selected cells and a short provenance label."""

	cells: list[np.ndarray]
	source_label: str


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments.

	Args:
		argv: Optional command-line argument vector. If None, argparse reads
			from ``sys.argv``.

	Returns:
		Parsed command-line arguments.
	"""
	parser = argparse.ArgumentParser(
		description=(
			"Convert VASP XDATCAR frames to a LAMMPS npt.dump-style text dump. "
			"ASE reads the XDATCAR; OUTCAR is optionally used for per-frame cells."
		)
	)
	parser.add_argument(
		"--xdatcar",
		type=Path,
		default=Path("XDATCAR"),
		help="Input VASP XDATCAR trajectory. Default: XDATCAR",
	)
	parser.add_argument(
		"--outcar",
		type=Path,
		default=Path("OUTCAR"),
		help="Optional OUTCAR used for per-frame lattice vectors. Default: OUTCAR",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("npt.dump"),
		help="Output LAMMPS dump path. Default: npt.dump",
	)
	parser.add_argument(
		"--elements",
		nargs="+",
		help=(
			"Element/type order for LAMMPS type IDs, e.g. --elements Mg Si O He. "
			"Default: first-appearance order in the first XDATCAR frame."
		),
	)
	parser.add_argument(
		"--cell-source",
		choices=("auto", "xdatcar", "outcar"),
		default="auto",
		help=(
			"Cell source. auto uses OUTCAR when its lattice-vector count aligns "
			"with the XDATCAR frames, otherwise XDATCAR. Default: auto"
		),
	)
	parser.add_argument(
		"--timestep-mode",
		choices=("xdatcar", "generated"),
		default="xdatcar",
		help=(
			"Timestep labels. xdatcar uses 'Direct configuration=' labels when "
			"available; generated uses --timestep-start/--timestep-stride. Default: xdatcar"
		),
	)
	parser.add_argument(
		"--timestep-start",
		type=int,
		default=0,
		help="First generated LAMMPS timestep. Default: 0",
	)
	parser.add_argument(
		"--timestep-stride",
		type=int,
		default=1,
		help="Generated timestep spacing. Default: 1",
	)
	parser.add_argument(
		"--skip",
		type=int,
		default=0,
		help="Number of initial XDATCAR frames to skip before writing. Default: 0",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		help="Maximum number of frames to write after --skip. Default: all remaining frames.",
	)
	parser.add_argument(
		"--frame-step",
		type=int,
		default=1,
		help="Write every Nth frame after --skip. Default: 1",
	)
	parser.add_argument(
		"--no-wrap",
		action="store_true",
		help="Do not wrap scaled coordinates into the unit cell before writing.",
	)
	return parser.parse_args(argv)


def read_xdatcar_step_labels(xdatcar_path: Path) -> list[int]:
	"""Read integer ``Direct configuration=`` labels from an XDATCAR.

	Args:
		xdatcar_path: Path to the XDATCAR file.

	Returns:
		Configuration labels in the order they appear. The list is empty if no
		labels can be parsed.
	"""
	pattern = re.compile(r"Direct\s+configuration\s*=\s*([0-9]+)", re.IGNORECASE)
	labels: list[int] = []
	with xdatcar_path.open("r", encoding="utf-8", errors="replace") as xdatcar_file:
		for line in xdatcar_file:
			match = pattern.search(line)
			if match:
				labels.append(int(match.group(1)))
	return labels


def read_xdatcar_frames(xdatcar_path: Path) -> list[object]:
	"""Read all XDATCAR frames through ASE.

	Args:
		xdatcar_path: Path to the XDATCAR file.

	Returns:
		List of ASE ``Atoms`` frames.

	Raises:
		FileNotFoundError: If the XDATCAR is missing.
		ValueError: If ASE reads no frames.
	"""
	if not xdatcar_path.is_file():
		raise FileNotFoundError(f"XDATCAR not found: {xdatcar_path}")

	try:
		from ase.io import read
	except ImportError as error:
		raise ImportError(
			"ASE is required. On Princeton clusters, try: "
			"module load anaconda3/2025.12; conda activate ase_env"
		) from error

	frames = read(xdatcar_path.as_posix(), format="vasp-xdatcar", index=":")
	if not isinstance(frames, list):
		frames = [frames]
	if not frames:
		raise ValueError(f"No frames were read from {xdatcar_path}")
	return frames


def parse_outcar_lattice_vectors(outcar_path: Path) -> list[np.ndarray]:
	"""Parse lattice-vector blocks from a VASP OUTCAR.

	VASP writes blocks headed by ``direct lattice vectors``. Each following
	three lines contains one direct lattice vector plus reciprocal-vector
	columns. This function keeps only the first three numeric columns.

	Args:
		outcar_path: Path to the OUTCAR file.

	Returns:
		List of 3x3 lattice matrices in Angstrom, with vectors as rows. The list
		is empty when the OUTCAR is missing.
	"""
	if not outcar_path.is_file():
		return []

	lines = outcar_path.read_text(encoding="utf-8", errors="replace").splitlines()
	cells: list[np.ndarray] = []
	for line_index, line in enumerate(lines):
		if "direct lattice vectors" not in line.lower():
			continue
		if line_index + 3 >= len(lines):
			continue

		cell_rows: list[list[float]] = []
		for vector_line in lines[line_index + 1: line_index + 4]:
			fields = vector_line.split()
			if len(fields) < 3:
				cell_rows = []
				break
			try:
				cell_rows.append([float(fields[0]), float(fields[1]), float(fields[2])])
			except ValueError:
				cell_rows = []
				break
		if len(cell_rows) == 3:
			cells.append(np.array(cell_rows, dtype=float))
	return cells


def align_outcar_cells(outcar_cells: list[np.ndarray], num_frames: int) -> list[np.ndarray] | None:
	"""Align OUTCAR cell blocks to XDATCAR frames.

	Args:
		outcar_cells: Lattice-vector blocks parsed from OUTCAR.
		num_frames: Number of XDATCAR frames.

	Returns:
		A list with one cell per XDATCAR frame, or None if the OUTCAR blocks
		cannot be aligned.
	"""
	if len(outcar_cells) == num_frames:
		return outcar_cells
	if len(outcar_cells) == num_frames + 1:
		return outcar_cells[1:]
	if len(outcar_cells) > num_frames:
		return outcar_cells[-num_frames:]
	return None


def cells_vary(cells: list[np.ndarray]) -> bool:
	"""Return whether a list of cells contains meaningful cell changes.

	Args:
		cells: Lattice matrices to inspect.

	Returns:
		True if any cell differs from the first cell beyond a small tolerance.
	"""
	if len(cells) < 2:
		return False
	first_cell = cells[0]
	return any(not np.allclose(first_cell, cell, atol=1.0e-8, rtol=1.0e-8) for cell in cells[1:])


def choose_cells(
	frames: list[object],
	outcar_cells: list[np.ndarray],
	cell_source: str,
) -> CellSelection:
	"""Choose one lattice matrix per frame.

	Args:
		frames: ASE ``Atoms`` frames from XDATCAR.
		outcar_cells: Parsed OUTCAR lattice-vector blocks.
		cell_source: One of ``auto``, ``xdatcar``, or ``outcar``.

	Returns:
		Selected cells and a short provenance label.

	Raises:
		ValueError: If ``cell_source=outcar`` is requested but OUTCAR cells do
			not align with the XDATCAR frame count.
	"""
	xdatcar_cells = [np.array(frame.cell.array, dtype=float) for frame in frames]
	aligned_outcar_cells = align_outcar_cells(outcar_cells, len(frames))

	if cell_source == "xdatcar":
		return CellSelection(cells=xdatcar_cells, source_label="XDATCAR")

	if cell_source == "outcar":
		if aligned_outcar_cells is None:
			raise ValueError(
				f"OUTCAR cells do not align with {len(frames)} XDATCAR frames "
				f"(parsed {len(outcar_cells)} OUTCAR cell blocks)."
			)
		return CellSelection(cells=aligned_outcar_cells, source_label="OUTCAR")

	if aligned_outcar_cells is not None:
		if cells_vary(aligned_outcar_cells) and not cells_vary(xdatcar_cells):
			return CellSelection(cells=aligned_outcar_cells, source_label="OUTCAR variable cells")
		return CellSelection(cells=aligned_outcar_cells, source_label="OUTCAR")

	return CellSelection(cells=xdatcar_cells, source_label="XDATCAR fallback")


def infer_element_order(frames: list[object], explicit_elements: Sequence[str] | None) -> list[str]:
	"""Infer or validate LAMMPS type order.

	Args:
		frames: ASE ``Atoms`` frames.
		explicit_elements: Optional user-supplied element order.

	Returns:
		Element symbols ordered by LAMMPS type ID.

	Raises:
		ValueError: If the explicit element list does not cover all atoms.
	"""
	symbols = list(frames[0].get_chemical_symbols())
	if explicit_elements:
		element_order = list(explicit_elements)
		missing = sorted(set(symbols).difference(element_order))
		if missing:
			raise ValueError(
				"--elements is missing symbols present in XDATCAR: " + ", ".join(missing)
			)
		return element_order

	element_order: list[str] = []
	for symbol in symbols:
		if symbol not in element_order:
			element_order.append(symbol)
	return element_order


def select_frames(
	frames: list[object],
	labels: list[int],
	skip: int,
	max_frames: int | None,
	frame_step: int,
) -> tuple[list[object], list[int], list[int]]:
	"""Apply frame selection to frames and labels.

	Args:
		frames: All frames read from XDATCAR.
		labels: One label per frame.
		skip: Number of initial frames to skip.
		max_frames: Optional maximum number of frames to keep.
		frame_step: Keep every Nth frame after skipping.

	Returns:
		A tuple of selected frames, selected labels, and selected original
		zero-based frame indices.

	Raises:
		ValueError: If selection settings are invalid or select no frames.
	"""
	if skip < 0:
		raise ValueError("--skip must be non-negative.")
	if frame_step < 1:
		raise ValueError("--frame-step must be at least 1.")
	if max_frames is not None and max_frames < 1:
		raise ValueError("--max-frames must be at least 1 when supplied.")

	selected_indices = list(range(skip, len(frames), frame_step))
	if max_frames is not None:
		selected_indices = selected_indices[:max_frames]
	if not selected_indices:
		raise ValueError("Frame selection is empty.")

	selected_frames = [frames[index] for index in selected_indices]
	selected_labels = [labels[index] for index in selected_indices]
	return selected_frames, selected_labels, selected_indices


def build_timestep_labels(
	xdatcar_labels: list[int],
	num_frames: int,
	mode: str,
	start: int,
	stride: int,
) -> list[int]:
	"""Build one LAMMPS timestep value per XDATCAR frame.

	Args:
		xdatcar_labels: Labels parsed from ``Direct configuration=`` lines.
		num_frames: Number of frames read from XDATCAR.
		mode: ``xdatcar`` or ``generated``.
		start: First generated timestep.
		stride: Generated timestep spacing.

	Returns:
		One integer timestep label per frame.
	"""
	if mode == "xdatcar" and len(xdatcar_labels) == num_frames:
		return xdatcar_labels
	if mode == "xdatcar" and xdatcar_labels:
		print(
			f"WARNING: parsed {len(xdatcar_labels)} XDATCAR labels for {num_frames} frames; "
			"using generated labels instead.",
			file=sys.stderr,
		)
	return [start + frame_index * stride for frame_index in range(num_frames)]


def cell_to_lammps_box(cell: np.ndarray) -> LammpsBox:
	"""Convert a cell to LAMMPS restricted-triclinic representation.

	The conversion rotates the cell into LAMMPS' restricted orientation while
	preserving fractional coordinates. This is safe for trajectory format
	conversion because the rotated structure is physically equivalent.

	Args:
		cell: 3x3 lattice matrix with vectors as rows.

	Returns:
		LAMMPS box bounds, tilt factors, and the rotated restricted cell.

	Raises:
		ValueError: If the cell is degenerate.
	"""
	a_vector = np.array(cell[0], dtype=float)
	b_vector = np.array(cell[1], dtype=float)
	c_vector = np.array(cell[2], dtype=float)

	lx = float(np.linalg.norm(a_vector))
	if lx <= 0.0:
		raise ValueError("Degenerate cell: a-vector length is zero.")
	a_hat = a_vector / lx

	xy = float(np.dot(b_vector, a_hat))
	xz = float(np.dot(c_vector, a_hat))

	b_perp = b_vector - xy * a_hat
	ly = float(np.linalg.norm(b_perp))
	if ly <= 0.0:
		raise ValueError("Degenerate cell: b-vector is collinear with a-vector.")
	b_hat = b_perp / ly

	yz = float(np.dot(c_vector, b_hat))
	c_perp = c_vector - xz * a_hat - yz * b_hat
	lz = float(np.linalg.norm(c_perp))
	if lz <= 0.0:
		raise ValueError("Degenerate cell: c-vector is coplanar with a/b vectors.")

	restricted_cell = np.array(
		[
			[lx, 0.0, 0.0],
			[xy, ly, 0.0],
			[xz, yz, lz],
		],
		dtype=float,
	)

	xlo = 0.0
	xhi = lx
	ylo = 0.0
	yhi = ly
	zlo = 0.0
	zhi = lz

	xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
	xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
	ylo_bound = ylo + min(0.0, yz)
	yhi_bound = yhi + max(0.0, yz)

	return LammpsBox(
		xlo_bound=xlo_bound,
		xhi_bound=xhi_bound,
		ylo_bound=ylo_bound,
		yhi_bound=yhi_bound,
		zlo_bound=zlo,
		zhi_bound=zhi,
		xy=xy,
		xz=xz,
		yz=yz,
		restricted_cell=restricted_cell,
	)


def format_float(value: float) -> str:
	"""Format a floating-point value for LAMMPS text output.

	Args:
		value: Numeric value to format.

	Returns:
		Formatted value with enough precision for trajectory conversion.
	"""
	return f"{value:.16g}"


def write_lammps_dump(
	output_path: Path,
	frames: list[object],
	cells: list[np.ndarray],
	timestep_labels: list[int],
	element_order: list[str],
	wrap_positions: bool,
) -> None:
	"""Write selected frames as a LAMMPS text dump.

	Args:
		output_path: Output dump path.
		frames: Selected ASE ``Atoms`` frames.
		cells: One cell per selected frame.
		timestep_labels: One LAMMPS timestep per selected frame.
		element_order: Element symbols ordered by LAMMPS type ID.
		wrap_positions: Whether to wrap scaled positions before writing.

	Raises:
		ValueError: If frame/cell/label counts differ.
	"""
	if not (len(frames) == len(cells) == len(timestep_labels)):
		raise ValueError("frames, cells, and timestep_labels must have the same length.")

	type_by_symbol = {symbol: type_id for type_id, symbol in enumerate(element_order, start=1)}
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", encoding="utf-8") as output_file:
		for frame, cell, timestep in zip(frames, cells, timestep_labels):
			symbols = list(frame.get_chemical_symbols())
			scaled_positions = np.array(frame.get_scaled_positions(wrap=wrap_positions), dtype=float)
			lammps_box = cell_to_lammps_box(cell=cell)
			positions = scaled_positions @ lammps_box.restricted_cell

			output_file.write("ITEM: TIMESTEP\n")
			output_file.write(f"{timestep}\n")
			output_file.write("ITEM: NUMBER OF ATOMS\n")
			output_file.write(f"{len(symbols)}\n")

			if lammps_box.is_triclinic:
				output_file.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
				output_file.write(
					f"{format_float(lammps_box.xlo_bound)} "
					f"{format_float(lammps_box.xhi_bound)} "
					f"{format_float(lammps_box.xy)}\n"
				)
				output_file.write(
					f"{format_float(lammps_box.ylo_bound)} "
					f"{format_float(lammps_box.yhi_bound)} "
					f"{format_float(lammps_box.xz)}\n"
				)
				output_file.write(
					f"{format_float(lammps_box.zlo_bound)} "
					f"{format_float(lammps_box.zhi_bound)} "
					f"{format_float(lammps_box.yz)}\n"
				)
			else:
				output_file.write("ITEM: BOX BOUNDS pp pp pp\n")
				output_file.write(
					f"{format_float(lammps_box.xlo_bound)} {format_float(lammps_box.xhi_bound)}\n"
				)
				output_file.write(
					f"{format_float(lammps_box.ylo_bound)} {format_float(lammps_box.yhi_bound)}\n"
				)
				output_file.write(
					f"{format_float(lammps_box.zlo_bound)} {format_float(lammps_box.zhi_bound)}\n"
				)

			output_file.write("ITEM: ATOMS id type element x y z\n")
			for atom_index, (symbol, position) in enumerate(zip(symbols, positions), start=1):
				type_id = type_by_symbol[symbol]
				output_file.write(
					f"{atom_index} {type_id} {symbol} "
					f"{format_float(float(position[0]))} "
					f"{format_float(float(position[1]))} "
					f"{format_float(float(position[2]))}\n"
				)


def main(argv: Sequence[str] | None = None) -> None:
	"""Run the XDATCAR-to-LAMMPS-dump conversion."""
	args = parse_arguments(argv=argv)

	frames = read_xdatcar_frames(xdatcar_path=args.xdatcar)
	xdatcar_labels = read_xdatcar_step_labels(xdatcar_path=args.xdatcar)
	all_timestep_labels = build_timestep_labels(
		xdatcar_labels=xdatcar_labels,
		num_frames=len(frames),
		mode=args.timestep_mode,
		start=args.timestep_start,
		stride=args.timestep_stride,
	)

	outcar_cells = parse_outcar_lattice_vectors(outcar_path=args.outcar)
	cell_selection = choose_cells(
		frames=frames,
		outcar_cells=outcar_cells,
		cell_source=args.cell_source,
	)

	selected_frames, selected_labels, selected_indices = select_frames(
		frames=frames,
		labels=all_timestep_labels,
		skip=args.skip,
		max_frames=args.max_frames,
		frame_step=args.frame_step,
	)
	selected_cells = [cell_selection.cells[index] for index in selected_indices]
	element_order = infer_element_order(frames=selected_frames, explicit_elements=args.elements)

	write_lammps_dump(
		output_path=args.output,
		frames=selected_frames,
		cells=selected_cells,
		timestep_labels=selected_labels,
		element_order=element_order,
		wrap_positions=not args.no_wrap,
	)

	print(f"Wrote {len(selected_frames)} frames to {args.output}")
	print(f"Atoms per frame: {len(selected_frames[0])}")
	print(f"LAMMPS element/type order: {' '.join(element_order)}")
	print(f"Cell source: {cell_selection.source_label}")
	print("Columns: id type element x y z")


if __name__ == "__main__":
	main()
