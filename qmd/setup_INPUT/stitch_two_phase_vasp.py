#!/usr/bin/env python3
"""Build a stitched two-phase VASP setup from two source directories.

This helper reads the first structural block from two POSCAR-like files,
rescales the in-plane cell lengths if needed, stitches the structures along
the z axis, assembles a matching POTCAR, and writes a quick visual-check
plot plus a JSON summary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import write
from ase.io.vasp import read_vasp


LOCAL_POTCAR_ROOT: Path = Path(
	"/projects/BURROWS/akashgpt/run_scripts/helpful_scripts/"
	"qmd/vasp/VASP_POTPAW__POTCARs/potpaw_PBE"
)


@dataclass(frozen=True)
class CellReport:
	"""Store orthorhombic cell metadata.

	Attributes:
		lx: Cell length along x in Angstrom.
		ly: Cell length along y in Angstrom.
		lz: Cell length along z in Angstrom.
	"""

	lx: float
	ly: float
	lz: float


@dataclass(frozen=True)
class ScaleReport:
	"""Store in-plane scaling factors for one input structure.

	Attributes:
		source_dir: Source directory for the structure.
		scale_x: Multiplicative scale factor applied along x.
		scale_y: Multiplicative scale factor applied along y.
	"""

	source_dir: str
	scale_x: float
	scale_y: float


@dataclass(frozen=True)
class SetupSummary:
	"""Capture stitched-setup metadata for later inspection.

	Attributes:
		dir1: First source directory.
		dir2: Second source directory.
		source_filename: Source structure filename used in each directory.
		xy_policy: Policy used to choose the stitched in-plane cell.
		gap: Gap inserted between the two source cells in Angstrom.
		species_order: Final POSCAR/POTCAR species order.
		species_counts: Atom counts in the stitched structure.
		cell_dir1: Original cell report for source 1.
		cell_dir2: Original cell report for source 2.
		cell_final: Final stitched cell report.
		scale_dir1: In-plane scaling applied to source 1.
		scale_dir2: In-plane scaling applied to source 2.
		potcar_paths: POTCAR component paths used to assemble the final POTCAR.
	"""

	dir1: str
	dir2: str
	source_filename: str
	xy_policy: str
	gap: float
	species_order: list[str]
	species_counts: dict[str, int]
	cell_dir1: CellReport
	cell_dir2: CellReport
	cell_final: CellReport
	scale_dir1: ScaleReport
	scale_dir2: ScaleReport
	potcar_paths: list[str]


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments.

	Returns:
		Parsed command-line arguments.
	"""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Stitch two VASP source directories along z."
	)
	parser.add_argument("dir1", type=Path, help="First source directory.")
	parser.add_argument("dir2", type=Path, help="Second source directory.")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("."),
		help="Output directory for the stitched setup.",
	)
	parser.add_argument(
		"--source-filename",
		type=str,
		default="CONTCAR",
		help="Filename to read inside each source directory.",
	)
	parser.add_argument(
		"--gap",
		type=float,
		default=0.50,
		help="Gap to insert between the two source cells in Angstrom.",
	)
	parser.add_argument(
		"--xy-policy",
		type=str,
		choices=("average", "max", "dir1", "dir2"),
		default="average",
		help="How to choose the stitched in-plane cell lengths.",
	)
	parser.add_argument(
		"--species-order",
		type=str,
		default="",
		help="Optional explicit species order, for example 'H O N'.",
	)
	parser.add_argument(
		"--potcar-root",
		type=Path,
		default=LOCAL_POTCAR_ROOT,
		help="Root directory that contains per-species PBE POTCAR folders.",
	)
	parser.add_argument(
		"--plot-file",
		type=str,
		default="stitched_structure_check.png",
		help="Filename for the output plot.",
	)
	parser.add_argument(
		"--summary-file",
		type=str,
		default="setup_summary.json",
		help="Filename for the setup summary JSON.",
	)
	parser.add_argument(
		"--output-poscar",
		type=str,
		default="POSCAR",
		help="Filename for the stitched POSCAR.",
	)
	parser.add_argument(
		"--output-potcar",
		type=str,
		default="POTCAR",
		help="Filename for the stitched POTCAR.",
	)
	return parser.parse_args()


def parse_species_order(species_order_text: str) -> list[str]:
	"""Parse an optional explicit species order string.

	Args:
		species_order_text: User-provided species order text.

	Returns:
		Species labels in the requested order. Returns an empty list if the
		input text is empty.

	Raises:
		ValueError: If duplicate species labels are present.
	"""
	species_order: list[str] = [
		token for token in species_order_text.strip().split() if token
	]
	if not species_order:
		return []
	if len(species_order) != len(set(species_order)):
		raise ValueError("The requested species order contains duplicates.")
	return species_order


def read_trimmed_vasp_structure(structure_path: Path) -> Atoms:
	"""Read only the first structural block from a POSCAR-like file.

	This is robust to VASP MD CONTCAR files that include predictor-corrector
	or lattice-dynamics restart data after the structural block.

	Args:
		structure_path: Path to the POSCAR-like file.

	Returns:
		The parsed ASE atoms object.

	Raises:
		FileNotFoundError: If the structure path does not exist.
		ValueError: If the file is too short to contain a VASP structure.
	"""
	if not structure_path.exists():
		raise FileNotFoundError(f"Missing structure file: {structure_path}")

	lines: list[str] = structure_path.read_text().splitlines()
	if len(lines) < 8:
		raise ValueError(f"File is too short to be a valid POSCAR: {structure_path}")

	atom_count_line: int = 6
	coord_mode_line: int = 7
	has_selective_dynamics: bool = lines[7].strip().lower().startswith("selective")
	if has_selective_dynamics:
		coord_mode_line = 8

	atom_counts: list[int] = [int(token) for token in lines[atom_count_line].split()]
	total_atoms: int = sum(atom_counts)
	last_structure_line: int = coord_mode_line + 1 + total_atoms
	trimmed_text: str = "\n".join(lines[:last_structure_line]) + "\n"

	with NamedTemporaryFile(mode="w", suffix=".vasp", delete=True) as handle:
		handle.write(trimmed_text)
		handle.flush()
		atoms: Atoms = read_vasp(handle.name)

	return atoms


def require_orthorhombic_cell(atoms: Atoms, label: str) -> CellReport:
	"""Validate and summarize a nearly orthorhombic cell.

	Args:
		atoms: Structure to validate.
		label: Human-readable label for error messages.

	Returns:
		Cell lengths for the structure.

	Raises:
		ValueError: If the cell is not close to axis-aligned orthorhombic.
	"""
	cell: np.ndarray = np.array(atoms.cell)
	off_diagonal: np.ndarray = cell.copy()
	np.fill_diagonal(off_diagonal, 0.0)
	if not np.allclose(off_diagonal, 0.0, atol=1.0e-8):
		raise ValueError(
			f"{label} is not axis-aligned orthorhombic, so this simple z-stitcher "
			"would not preserve the intended geometry."
		)

	return CellReport(
		lx=float(cell[0, 0]),
		ly=float(cell[1, 1]),
		lz=float(cell[2, 2]),
	)


def choose_target_xy(
	cell_dir1: CellReport,
	cell_dir2: CellReport,
	xy_policy: str,
) -> tuple[float, float]:
	"""Choose the final x and y cell lengths.

	Args:
		cell_dir1: Cell report for source 1.
		cell_dir2: Cell report for source 2.
		xy_policy: Policy for choosing the final in-plane lengths.

	Returns:
		The target x and y lengths.
	"""
	if xy_policy == "dir1":
		return cell_dir1.lx, cell_dir1.ly
	if xy_policy == "dir2":
		return cell_dir2.lx, cell_dir2.ly
	if xy_policy == "max":
		return max(cell_dir1.lx, cell_dir2.lx), max(cell_dir1.ly, cell_dir2.ly)
	return (
		0.50 * (cell_dir1.lx + cell_dir2.lx),
		0.50 * (cell_dir1.ly + cell_dir2.ly),
	)


def scale_inplane(
	atoms: Atoms,
	target_lx: float,
	target_ly: float,
) -> tuple[Atoms, float, float]:
	"""Scale a structure to the target in-plane cell lengths.

	Args:
		atoms: Source structure.
		target_lx: Desired x length in Angstrom.
		target_ly: Desired y length in Angstrom.

	Returns:
		A tuple containing the scaled structure, x scale factor, and y scale factor.
	"""
	cell_report: CellReport = require_orthorhombic_cell(atoms, "Input structure")
	scale_x: float = target_lx / cell_report.lx
	scale_y: float = target_ly / cell_report.ly
	positions: np.ndarray = np.array(atoms.get_positions())
	positions[:, 0] *= scale_x
	positions[:, 1] *= scale_y

	scaled_atoms: Atoms = atoms.copy()
	scaled_atoms.set_positions(positions)
	scaled_atoms.set_cell(
		[
			[target_lx, 0.0, 0.0],
			[0.0, target_ly, 0.0],
			[0.0, 0.0, cell_report.lz],
		],
		scale_atoms=False,
	)
	scaled_atoms.set_pbc((True, True, True))
	scaled_atoms.wrap()
	return scaled_atoms, scale_x, scale_y


def infer_species_order(
	atoms1: Atoms,
	atoms2: Atoms,
	requested_species_order: list[str],
) -> list[str]:
	"""Choose the final species order.

	Args:
		atoms1: First source structure.
		atoms2: Second source structure.
		requested_species_order: Optional explicit species order.

	Returns:
		The final validated species order.

	Raises:
		ValueError: If the explicit order does not match the stitched species set.
	"""
	encountered: list[str] = []
	for symbol in atoms1.get_chemical_symbols() + atoms2.get_chemical_symbols():
		if symbol not in encountered:
			encountered.append(symbol)

	if not requested_species_order:
		return encountered

	if set(requested_species_order) != set(encountered):
		raise ValueError(
			"The explicit species order does not match the species present in the "
			f"inputs. Expected some ordering of: {' '.join(encountered)}"
		)
	return requested_species_order


def reorder_atoms_by_species(atoms: Atoms, species_order: list[str]) -> Atoms:
	"""Reorder atoms so they are grouped by the final species order.

	Args:
		atoms: Structure to reorder.
		species_order: Desired species grouping order.

	Returns:
		A reordered copy of the input structure.
	"""
	symbols: list[str] = atoms.get_chemical_symbols()
	ordered_indices: list[int] = []
	for symbol in species_order:
		ordered_indices.extend(
			index for index, current_symbol in enumerate(symbols) if current_symbol == symbol
		)
	return atoms[ordered_indices]


def stitch_along_z(
	atoms1: Atoms,
	atoms2: Atoms,
	gap: float,
	species_order: list[str],
) -> Atoms:
	"""Stack two structures along z and reorder by species.

	Args:
		atoms1: First structure placed at the bottom.
		atoms2: Second structure placed above the first.
		gap: Gap between the two source cells in Angstrom.
		species_order: Final species order.

	Returns:
		The stitched structure.
	"""
	cell1: np.ndarray = np.array(atoms1.cell)
	cell2: np.ndarray = np.array(atoms2.cell)
	positions1: np.ndarray = np.array(atoms1.get_positions())
	positions2: np.ndarray = np.array(atoms2.get_positions())
	positions2[:, 2] += float(cell1[2, 2]) + gap

	stitched: Atoms = Atoms(
		symbols=atoms1.get_chemical_symbols() + atoms2.get_chemical_symbols(),
		positions=np.vstack([positions1, positions2]),
		cell=[
			[float(cell1[0, 0]), 0.0, 0.0],
			[0.0, float(cell1[1, 1]), 0.0],
			[0.0, 0.0, float(cell1[2, 2] + gap + cell2[2, 2])],
		],
		pbc=(True, True, True),
	)
	stitched.wrap()
	return reorder_atoms_by_species(stitched, species_order)


def write_potcar(
	species_order: list[str],
	potcar_root: Path,
	output_path: Path,
) -> list[str]:
	"""Assemble a POTCAR from per-species POTCAR files.

	Args:
		species_order: Final species order used in POSCAR.
		potcar_root: Root directory that contains one directory per species.
		output_path: Output POTCAR path.

	Returns:
		List of POTCAR component paths used to assemble the file.

	Raises:
		FileNotFoundError: If any requested POTCAR component is missing.
	"""
	component_paths: list[str] = []
	potcar_chunks: list[str] = []
	for species in species_order:
		component_path: Path = potcar_root / species / "POTCAR"
		if not component_path.exists():
			raise FileNotFoundError(
				f"Missing POTCAR component for species {species}: {component_path}"
			)
		component_paths.append(str(component_path))
		potcar_chunks.append(component_path.read_text())

	output_path.write_text("".join(potcar_chunks))
	return component_paths


def parse_potcar_species(output_potcar: Path) -> list[str]:
	"""Extract the species order from TITEL lines in a POTCAR.

	Args:
		output_potcar: POTCAR path to inspect.

	Returns:
		Species symbols found in TITEL order.
	"""
	species: list[str] = []
	for line in output_potcar.read_text(errors="ignore").splitlines():
		if "TITEL" not in line or "PAW_PBE" not in line:
			continue
		species.append(line.split("PAW_PBE", maxsplit=1)[1].strip().split()[0])
	return species


def build_species_color_map(species_order: list[str]) -> dict[str, str]:
	"""Assign a stable plotting color to each species.

	Args:
		species_order: Final species order.

	Returns:
		Mapping from species label to matplotlib color.
	"""
	palette: list[str] = [
		"tab:blue",
		"tab:red",
		"tab:green",
		"tab:orange",
		"tab:purple",
		"tab:brown",
		"tab:pink",
		"tab:gray",
	]
	return {
		species: palette[index % len(palette)]
		for index, species in enumerate(species_order)
	}


def plot_structure_panel(
	axis: plt.Axes,
	atoms: Atoms,
	title: str,
	color_map: dict[str, str],
) -> None:
	"""Plot an x-z projection for one structure panel.

	Args:
		axis: Matplotlib axis to draw on.
		atoms: Structure to plot.
		title: Panel title.
		color_map: Mapping from species label to plot color.
	"""
	positions: np.ndarray = np.array(atoms.get_positions())
	symbols: list[str] = atoms.get_chemical_symbols()
	for species in color_map:
		indices: list[int] = [
			index for index, symbol in enumerate(symbols) if symbol == species
		]
		if not indices:
			continue
		axis.scatter(
			positions[indices, 0],
			positions[indices, 2],
			s=8,
			c=color_map[species],
			label=species,
			alpha=0.80,
		)

	cell: np.ndarray = np.array(atoms.cell)
	axis.set_xlim(0.0, float(cell[0, 0]))
	axis.set_ylim(0.0, float(cell[2, 2]))
	axis.set_xlabel("x (A)")
	axis.set_ylabel("z (A)")
	axis.set_title(title)
	axis.set_aspect("auto")


def write_structure_plot(
	atoms1: Atoms,
	atoms2: Atoms,
	stitched_atoms: Atoms,
	species_order: list[str],
	output_path: Path,
	title1: str,
	title2: str,
) -> None:
	"""Create the quick-check overview plot.

	Args:
		atoms1: First source structure.
		atoms2: Second source structure.
		stitched_atoms: Stitched output structure.
		species_order: Final species order.
		output_path: Plot output path.
		title1: Human-readable title for source 1.
		title2: Human-readable title for source 2.
	"""
	color_map: dict[str, str] = build_species_color_map(species_order)
	figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
	plot_structure_panel(axes[0], atoms1, title1, color_map)
	plot_structure_panel(axes[1], atoms2, title2, color_map)
	plot_structure_panel(axes[2], stitched_atoms, "Stitched POSCAR", color_map)

	handles, labels = axes[2].get_legend_handles_labels()
	if handles:
		figure.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
	figure.savefig(output_path, dpi=200)
	plt.close(figure)


def write_summary(summary: SetupSummary, summary_path: Path) -> None:
	"""Write the setup summary JSON.

	Args:
		summary: Setup summary dataclass.
		summary_path: Output JSON path.
	"""
	summary_path.write_text(json.dumps(asdict(summary), indent=2))


def main() -> None:
	"""Run the stitched-setup workflow."""
	args: argparse.Namespace = parse_args()
	requested_species_order: list[str] = parse_species_order(args.species_order)
	output_dir: Path = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	structure_path1: Path = args.dir1.resolve() / args.source_filename
	structure_path2: Path = args.dir2.resolve() / args.source_filename

	atoms1_raw: Atoms = read_trimmed_vasp_structure(structure_path1)
	atoms2_raw: Atoms = read_trimmed_vasp_structure(structure_path2)
	cell_dir1: CellReport = require_orthorhombic_cell(atoms1_raw, "dir1 structure")
	cell_dir2: CellReport = require_orthorhombic_cell(atoms2_raw, "dir2 structure")

	target_lx, target_ly = choose_target_xy(cell_dir1, cell_dir2, args.xy_policy)
	atoms1, scale1_x, scale1_y = scale_inplane(atoms1_raw, target_lx, target_ly)
	atoms2, scale2_x, scale2_y = scale_inplane(atoms2_raw, target_lx, target_ly)

	species_order: list[str] = infer_species_order(
		atoms1, atoms2, requested_species_order
	)
	stitched_atoms: Atoms = stitch_along_z(atoms1, atoms2, args.gap, species_order)

	output_poscar: Path = output_dir / args.output_poscar
	output_potcar: Path = output_dir / args.output_potcar
	output_plot: Path = output_dir / args.plot_file
	output_summary: Path = output_dir / args.summary_file

	write(
		str(output_poscar),
		stitched_atoms,
		format="vasp",
		direct=True,
		sort=False,
		vasp5=True,
	)
	potcar_paths: list[str] = write_potcar(species_order, args.potcar_root, output_potcar)

	potcar_species: list[str] = parse_potcar_species(output_potcar)
	if potcar_species[: len(species_order)] != species_order:
		raise ValueError(
			"The assembled POTCAR species order does not match the POSCAR species "
			f"order. POSCAR order: {species_order}; POTCAR order: {potcar_species}"
		)

	write_structure_plot(
		atoms1=atoms1,
		atoms2=atoms2,
		stitched_atoms=stitched_atoms,
		species_order=species_order,
		output_path=output_plot,
		title1=f"{args.dir1.name}/{args.source_filename}",
		title2=f"{args.dir2.name}/{args.source_filename}",
	)

	counts: Counter[str] = Counter(stitched_atoms.get_chemical_symbols())
	cell_final: CellReport = require_orthorhombic_cell(stitched_atoms, "stitched structure")
	summary: SetupSummary = SetupSummary(
		dir1=str(args.dir1.resolve()),
		dir2=str(args.dir2.resolve()),
		source_filename=args.source_filename,
		xy_policy=args.xy_policy,
		gap=float(args.gap),
		species_order=species_order,
		species_counts={species: counts[species] for species in species_order},
		cell_dir1=cell_dir1,
		cell_dir2=cell_dir2,
		cell_final=cell_final,
		scale_dir1=ScaleReport(
			source_dir=str(args.dir1.resolve()),
			scale_x=scale1_x,
			scale_y=scale1_y,
		),
		scale_dir2=ScaleReport(
			source_dir=str(args.dir2.resolve()),
			scale_x=scale2_x,
			scale_y=scale2_y,
		),
		potcar_paths=potcar_paths,
	)
	write_summary(summary, output_summary)

	print(f"Wrote stitched POSCAR: {output_poscar}")
	print(f"Wrote stitched POTCAR: {output_potcar}")
	print(f"Wrote quick-check plot: {output_plot}")
	print(f"Wrote setup summary: {output_summary}")
	print(f"Final species order: {' '.join(species_order)}")
	print(
		"Final species counts: "
		+ " ".join(f"{species}={counts[species]}" for species in species_order)
	)


if __name__ == "__main__":
	main()
