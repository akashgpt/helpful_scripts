#!/usr/bin/env python3
"""Convert a VASP XDATCAR trajectory to an extended XYZ file.

The output is ASE ``extxyz`` so each frame can keep lattice and PBC metadata.
This is usually more useful than plain XYZ for QMD/MLMD workflows because ASAP,
SOAP, and interface-analysis tools need the simulation cell.

Examples:
	module load anaconda3/2025.12; conda activate ase_env
	python vasp_xdatcar_to_xyz.py
	python vasp_xdatcar_to_xyz.py --xdatcar XDATCAR --outcar OUTCAR --output trajectory.xyz
	python $HELP_SCRIPTS/qmd/setup_INPUT/vasp_xdatcar_to_xyz.py --max-frames 1000 --frame-step 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from vasp_xdatcar_to_lammps_dump import (
	build_timestep_labels,
	choose_cells,
	parse_outcar_lattice_vectors,
	read_xdatcar_frames,
	read_xdatcar_step_labels,
	select_frames,
)


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
			"Convert VASP XDATCAR frames to an ASE extended XYZ trajectory. "
			"OUTCAR can provide per-frame cells when XDATCAR carries only one cell."
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
		default=Path("trajectory.xyz"),
		help="Output extended XYZ path. Default: trajectory.xyz",
	)
	parser.add_argument(
		"--cell-source",
		choices=("auto", "xdatcar", "outcar"),
		default="auto",
		help=(
			"Cell source. auto uses OUTCAR when its lattice-vector count aligns "
			"with XDATCAR frames, otherwise XDATCAR. Default: auto"
		),
	)
	parser.add_argument(
		"--timestep-mode",
		choices=("xdatcar", "generated"),
		default="xdatcar",
		help=(
			"Timestep labels stored in frame metadata. xdatcar uses "
			"'Direct configuration=' labels when available; generated uses "
			"--timestep-start/--timestep-stride. Default: xdatcar"
		),
	)
	parser.add_argument(
		"--timestep-start",
		type=int,
		default=0,
		help="First generated timestep label. Default: 0",
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
		help="Do not wrap atomic positions into the unit cell before writing.",
	)
	parser.add_argument(
		"--plain",
		action="store_true",
		help=(
			"Write plain XYZ instead of extxyz. This drops lattice/PBC metadata "
			"and is usually not recommended for periodic QMD trajectories."
		),
	)
	return parser.parse_args(argv)


def prepare_xyz_frames(
	frames: list[object],
	cells: list[np.ndarray],
	timestep_labels: list[int],
	wrap_positions: bool,
	cell_source_label: str,
) -> list[object]:
	"""Prepare ASE frames for XYZ writing.

	Args:
		frames: Selected ASE ``Atoms`` frames.
		cells: One lattice matrix per selected frame.
		timestep_labels: One timestep/configuration label per selected frame.
		wrap_positions: Whether to wrap atoms into the unit cell.
		cell_source_label: Human-readable label for the selected cell source.

	Returns:
		New ASE ``Atoms`` frames with updated cell/PBC and metadata.

	Raises:
		ValueError: If frame, cell, and label counts differ.
	"""
	if not (len(frames) == len(cells) == len(timestep_labels)):
		raise ValueError("frames, cells, and timestep_labels must have the same length.")

	prepared_frames: list[object] = []
	for frame_index, (frame, cell, timestep_label) in enumerate(
		zip(frames, cells, timestep_labels)
	):
		prepared_frame = frame.copy()
		prepared_frame.set_cell(cell)
		prepared_frame.set_pbc((True, True, True))
		if wrap_positions:
			prepared_frame.wrap()
		prepared_frame.info["vasp_step"] = int(timestep_label)
		prepared_frame.info["frame_index"] = int(frame_index)
		prepared_frame.info["cell_source"] = cell_source_label
		prepared_frames.append(prepared_frame)
	return prepared_frames


def write_xyz(output_path: Path, frames: list[object], plain: bool) -> None:
	"""Write frames to XYZ or extended XYZ.

	Args:
		output_path: Output XYZ path.
		frames: ASE ``Atoms`` frames to write.
		plain: If True, write plain XYZ; otherwise write extended XYZ.
	"""
	try:
		from ase.io import write
	except ImportError as error:
		raise ImportError(
			"ASE is required. On Princeton clusters, try: "
			"module load anaconda3/2025.12; conda activate ase_env"
		) from error

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_format = "xyz" if plain else "extxyz"
	write(output_path.as_posix(), frames, format=output_format)


def main(argv: Sequence[str] | None = None) -> None:
	"""Run the XDATCAR/OUTCAR-to-XYZ conversion."""
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
	xyz_frames = prepare_xyz_frames(
		frames=selected_frames,
		cells=selected_cells,
		timestep_labels=selected_labels,
		wrap_positions=not args.no_wrap,
		cell_source_label=cell_selection.source_label,
	)

	write_xyz(output_path=args.output, frames=xyz_frames, plain=args.plain)

	output_kind = "plain XYZ" if args.plain else "extended XYZ"
	print(f"Wrote {len(xyz_frames)} frames to {args.output}")
	print(f"Output format: {output_kind}")
	print(f"Atoms per frame: {len(xyz_frames[0])}")
	print(f"Cell source: {cell_selection.source_label}")
	if args.plain:
		print("WARNING: plain XYZ does not preserve lattice/PBC metadata.")


if __name__ == "__main__":
	main()
