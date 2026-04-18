#!/usr/bin/env python3
"""Generate molecular H2S supercells with ASE.

This script mirrors the molecular-grid workflow used for H2 and NH3 setup
scripts in this directory. It places a requested number of H2S molecules on a
regular grid inside an orthorhombic box and assigns each molecule a random
orientation. The default molecular geometry is an idealized gas-phase H2S
geometry and can be changed with command-line options.

Usage:
	python initialize_structure_ASE_H2S.py
	python initialize_structure_ASE_H2S.py --num-molecules 128 --cell-lengths 16 16 16 --seed 11
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write

REFERENCE_LAMMPS_MASSES: dict[str, float] = {
	"S": 32.065,
	"H": 1.00794,
}

DEFAULT_SPECIES_ORDER: list[str] = ["H", "S"]


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
	"""Return a uniformly random 3D rotation matrix.

	Args:
		rng: Random-number generator used to build a reproducible rotation.

	Returns:
		A ``3 x 3`` rotation matrix.
	"""
	u1, u2, u3 = rng.random(3)

	q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
	q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
	q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
	q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)

	rotation_matrix = np.array(
		[
			[
				1.0 - 2.0 * (q3 * q3 + q4 * q4),
				2.0 * (q2 * q3 - q1 * q4),
				2.0 * (q2 * q4 + q1 * q3),
			],
			[
				2.0 * (q2 * q3 + q1 * q4),
				1.0 - 2.0 * (q2 * q2 + q4 * q4),
				2.0 * (q3 * q4 - q1 * q2),
			],
			[
				2.0 * (q2 * q4 - q1 * q3),
				2.0 * (q3 * q4 + q1 * q2),
				1.0 - 2.0 * (q2 * q2 + q3 * q3),
			],
		],
		dtype=float,
	)
	return rotation_matrix


def h2s_geometry(
	bond_length: float = 1.336,
	angle_deg: float = 92.1,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return an idealized H2S geometry centered on sulfur.

	Args:
		bond_length: S-H bond length in Angstrom.
		angle_deg: H-S-H bond angle in degrees.

	Returns:
		A tuple containing:
			1. The sulfur position relative to the molecular placement center.
			2. A ``2 x 3`` array containing hydrogen positions relative to sulfur.
	"""
	validated_bond_length = validate_positive_float(
		value=bond_length,
		value_name="S-H bond length",
	)
	validated_angle = validate_angle_degrees(angle_deg=angle_deg)
	half_angle = 0.5 * np.deg2rad(validated_angle)
	sulfur_position = np.array([0.0, 0.0, 0.0], dtype=float)
	hydrogen_positions = validated_bond_length * np.array(
		[
			[np.sin(half_angle), 0.0, np.cos(half_angle)],
			[-np.sin(half_angle), 0.0, np.cos(half_angle)],
		],
		dtype=float,
	)
	return sulfur_position, hydrogen_positions


def validate_positive_float(value: float, value_name: str) -> float:
	"""Validate that a floating-point value is positive.

	Args:
		value: Value to validate.
		value_name: Human-readable name used in error messages.

	Returns:
		The validated value.

	Raises:
		ValueError: If ``value`` is not positive.
	"""
	if value <= 0.0:
		raise ValueError(f"{value_name} must be positive.")
	return value


def validate_angle_degrees(angle_deg: float) -> float:
	"""Validate a molecular angle in degrees.

	Args:
		angle_deg: Requested molecular angle in degrees.

	Returns:
		The validated angle.

	Raises:
		ValueError: If the angle is outside the open interval ``(0, 180)``.
	"""
	if angle_deg <= 0.0 or angle_deg >= 180.0:
		raise ValueError("The H-S-H angle must be greater than 0 and less than 180 degrees.")
	return angle_deg


def validate_num_molecules(num_molecules: int) -> int:
	"""Validate the requested H2S molecule count.

	Args:
		num_molecules: Requested number of H2S molecules.

	Returns:
		The validated number of H2S molecules.

	Raises:
		ValueError: If the requested molecule count is not positive.
	"""
	if num_molecules <= 0:
		raise ValueError("The number of H2S molecules must be positive.")
	return num_molecules


def validate_cell_lengths(cell_lengths: tuple[float, float, float]) -> tuple[float, float, float]:
	"""Validate the requested orthorhombic box lengths.

	Args:
		cell_lengths: Requested box lengths along the ``x``, ``y``, and ``z`` axes.

	Returns:
		The validated box lengths.

	Raises:
		ValueError: If any box length is not positive.
	"""
	if len(cell_lengths) != 3:
		raise ValueError("Exactly three cell lengths are required.")
	if any(length <= 0.0 for length in cell_lengths):
		raise ValueError("All cell lengths must be positive.")
	return cell_lengths


def infer_grid_shape(
	num_molecules: int,
	cell_lengths: tuple[float, float, float],
) -> tuple[int, int, int]:
	"""Infer a regular placement grid that can host the requested molecules.

	Args:
		num_molecules: Number of H2S molecular centers that must be placed.
		cell_lengths: Box lengths along the ``x``, ``y``, and ``z`` axes.

	Returns:
		Grid counts along ``x``, ``y``, and ``z`` whose product is at least
		``num_molecules``.
	"""
	cell_lengths_array = np.array(cell_lengths, dtype=float)
	target_spacing = (np.prod(cell_lengths_array) / float(num_molecules)) ** (1.0 / 3.0)
	grid_shape = np.maximum(1, np.ceil(cell_lengths_array / target_spacing).astype(int))

	while int(np.prod(grid_shape)) < num_molecules:
		current_spacings = cell_lengths_array / grid_shape
		axis_index = int(np.argmax(current_spacings))
		grid_shape[axis_index] += 1

	return tuple(int(value) for value in grid_shape)


def generate_molecule_centers(
	num_molecules: int,
	cell_lengths: tuple[float, float, float],
	rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int, int]]:
	"""Generate H2S molecular-center positions inside the box.

	Args:
		num_molecules: Number of H2S molecules to place.
		cell_lengths: Box lengths along the ``x``, ``y``, and ``z`` axes.
		rng: Random-number generator used to shuffle candidate sites.

	Returns:
		A tuple containing:
			1. A ``num_molecules x 3`` array of molecular-center positions.
			2. The regular center-placement grid shape used internally.
	"""
	grid_shape = infer_grid_shape(
		num_molecules=num_molecules,
		cell_lengths=cell_lengths,
	)
	nx, ny, nz = grid_shape
	cell_lengths_array = np.array(cell_lengths, dtype=float)

	centers: list[np.ndarray] = []
	for i_index in range(nx):
		for j_index in range(ny):
			for k_index in range(nz):
				fractional_position = np.array(
					[
						(i_index + 0.5) / nx,
						(j_index + 0.5) / ny,
						(k_index + 0.5) / nz,
					],
					dtype=float,
				)
				centers.append(fractional_position * cell_lengths_array)

	center_array = np.array(centers, dtype=float)
	rng.shuffle(center_array)
	return center_array[:num_molecules], grid_shape


def _min_dist_to_placed(
	new_positions: np.ndarray,
	placed_positions: np.ndarray,
	cell_lengths: np.ndarray,
) -> float:
	"""Return the minimum distance from any new atom to any already-placed atom.

	Uses the orthorhombic minimum-image convention for periodic boundary
	conditions so distances across cell boundaries are handled correctly.

	Args:
		new_positions: Positions of the atoms in the candidate molecule
			(shape ``N_new x 3``).
		placed_positions: Positions of atoms already accepted into the box
			(shape ``N_placed x 3``).
		cell_lengths: Orthorhombic cell side lengths ``[lx, ly, lz]``.

	Returns:
		Minimum distance in Angstrom, or ``float('inf')`` when no atoms have
		been placed yet.
	"""
	if len(placed_positions) == 0:
		return float("inf")
	min_d = float("inf")
	for pos in new_positions:
		diff = placed_positions - pos  # shape (N_placed, 3)
		diff -= cell_lengths * np.round(diff / cell_lengths)
		dists = np.linalg.norm(diff, axis=1)
		min_d = min(min_d, float(dists.min()))
	return min_d


def build_h2s_molecular_box(
	num_molecules: int = 64,
	cell_lengths: tuple[float, float, float] = (12.0, 12.0, 12.0),
	bond_length: float = 1.336,
	angle_deg: float = 92.1,
	seed: int = 7,
	min_interatomic_distance: float = 1.0,
	max_rotation_retries: int = 500,
) -> tuple[Atoms, tuple[int, int, int]]:
	"""Build a representative molecular-H2S structure in an orthorhombic box.

	Each molecule is oriented by rejection-sampling: random rotations are tried
	until all atoms in the new molecule are at least ``min_interatomic_distance``
	Angstrom from every atom already placed.  This prevents the close-contact
	overlaps (< 0.5 Ang) that cause VASP MLFF runs to explode.

	Args:
		num_molecules: Number of H2S molecules to place.
		cell_lengths: Box lengths along the ``x``, ``y``, and ``z`` axes.
		bond_length: S-H bond length in Angstrom.
		angle_deg: H-S-H bond angle in degrees.
		seed: Seed used to make random molecular orientations reproducible.
		min_interatomic_distance: Minimum allowed distance (Angstrom) between
			any atom in the new molecule and any previously placed atom.
			Default is 1.0 Ang, which is safe for VASP MLFF initialisation.
		max_rotation_retries: Maximum number of random rotations tried per
			molecule before raising a ``RuntimeError``.  Increase this or use a
			larger cell if placement fails.  Default is 500.

	Returns:
		A tuple containing:
			1. An ASE ``Atoms`` object containing the molecular H2S structure.
			2. The regular center-placement grid shape used internally.

	Raises:
		RuntimeError: If a valid rotation cannot be found for a molecule within
			``max_rotation_retries`` attempts.
	"""
	validated_num_molecules = validate_num_molecules(num_molecules=num_molecules)
	validated_cell_lengths = validate_cell_lengths(cell_lengths=cell_lengths)
	rng = np.random.default_rng(seed)
	cell = np.diag(validated_cell_lengths).astype(float)
	cell_lengths_array = np.array(validated_cell_lengths, dtype=float)
	sulfur_position, hydrogen_positions = h2s_geometry(
		bond_length=bond_length,
		angle_deg=angle_deg,
	)
	molecular_centers, grid_shape = generate_molecule_centers(
		num_molecules=validated_num_molecules,
		cell_lengths=validated_cell_lengths,
		rng=rng,
	)

	symbols: list[str] = []
	positions: list[np.ndarray] = []
	placed_positions: np.ndarray = np.empty((0, 3), dtype=float)

	for mol_idx, molecular_center in enumerate(molecular_centers):
		placed = False
		for attempt in range(max_rotation_retries):
			rotation_matrix = random_rotation_matrix(rng=rng)
			s_pos = molecular_center + rotation_matrix @ sulfur_position
			h_positions = np.array(
				[molecular_center + rotation_matrix @ h for h in hydrogen_positions],
				dtype=float,
			)
			# Apply periodic wrapping to candidate positions before distance check
			candidate = np.vstack([s_pos, h_positions])
			candidate_wrapped = candidate % cell_lengths_array

			min_d = _min_dist_to_placed(
				new_positions=candidate_wrapped,
				placed_positions=placed_positions,
				cell_lengths=cell_lengths_array,
			)
			if min_d >= min_interatomic_distance:
				symbols.append("S")
				positions.append(s_pos)
				for h_pos in h_positions:
					symbols.append("H")
					positions.append(h_pos)
				placed_positions = np.vstack([placed_positions, candidate_wrapped])
				placed = True
				break

		if not placed:
			raise RuntimeError(
				f"Could not place molecule {mol_idx + 1}/{validated_num_molecules} "
				f"with min interatomic distance >= {min_interatomic_distance:.2f} Ang "
				f"after {max_rotation_retries} rotation attempts. "
				f"Try a larger cell (--cell-lengths) or increase --max-rotation-retries."
			)

	atoms = Atoms(
		symbols=symbols,
		positions=np.array(positions, dtype=float),
		cell=cell,
		pbc=True,
	)
	atoms.wrap()
	return atoms, grid_shape


def reorder_atoms_by_species(atoms: Atoms, species_order: list[str]) -> Atoms:
	"""Return a new structure with atoms grouped by the requested species order.

	Args:
		atoms: Input structure to reorder.
		species_order: Chemical symbols in the desired output order.

	Returns:
		A reordered copy of the input structure.
	"""
	symbols = np.array(atoms.get_chemical_symbols())
	ordered_indices: list[int] = []

	for symbol in species_order:
		ordered_indices.extend(np.where(symbols == symbol)[0].tolist())

	remaining_indices = [
		index for index, symbol in enumerate(symbols) if symbol not in species_order
	]
	ordered_indices.extend(remaining_indices)
	return atoms[ordered_indices]


def infer_species_in_output_order(atoms: Atoms, species_order: list[str]) -> list[str]:
	"""Infer the final species order used in written output files.

	Args:
		atoms: Structure whose species must be written.
		species_order: Preferred species ordering.

	Returns:
		Species present in the structure, ordered first by ``species_order`` and
		then by first appearance for any remaining species.
	"""
	atom_symbols = atoms.get_chemical_symbols()
	unique_symbols_by_appearance: list[str] = []

	for symbol in atom_symbols:
		if symbol not in unique_symbols_by_appearance:
			unique_symbols_by_appearance.append(symbol)

	ordered_symbols = [
		symbol for symbol in species_order if symbol in unique_symbols_by_appearance
	]
	ordered_symbols.extend(
		symbol for symbol in unique_symbols_by_appearance if symbol not in ordered_symbols
	)
	return ordered_symbols


def normalize_poscar_header(
	poscar_path: str | Path,
	comment_line: str,
	species_order: list[str],
) -> None:
	"""Rewrite the POSCAR comment and species line to a clean grouped order.

	Args:
		poscar_path: Path to the POSCAR file to normalize.
		comment_line: Free-form POSCAR title/comment line.
		species_order: Chemical symbols in the desired output order.
	"""
	poscar_file = Path(poscar_path)
	lines = poscar_file.read_text(encoding="utf-8").splitlines()
	lines[0] = comment_line
	lines[5] = " ".join(species_order)
	poscar_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_lammps_mass(symbol: str) -> float:
	"""Return the mass to write for a species in ``conf.lmp``.

	Args:
		symbol: Chemical symbol whose mass should be written.

	Returns:
		The mass value used in the LAMMPS ``Masses`` section.
	"""
	if symbol in REFERENCE_LAMMPS_MASSES:
		return REFERENCE_LAMMPS_MASSES[symbol]
	return float(atomic_masses[atomic_numbers[symbol]])


def validate_orthorhombic_cell(
	atoms: Atoms,
	tolerance: float = 1.0e-10,
) -> tuple[float, float, float]:
	"""Validate that the structure cell is orthorhombic.

	Args:
		atoms: Structure whose cell is about to be written.
		tolerance: Absolute tolerance used to detect nonzero off-diagonal terms.

	Returns:
		The orthorhombic cell lengths ``(lx, ly, lz)`` in Angstrom.

	Raises:
		ValueError: If the structure cell is not orthorhombic.
	"""
	cell_matrix = np.array(atoms.cell.array, dtype=float)
	off_diagonal_terms = cell_matrix.copy()
	np.fill_diagonal(off_diagonal_terms, 0.0)

	if not np.allclose(off_diagonal_terms, 0.0, atol=tolerance):
		raise ValueError("The conf.lmp writer currently expects an orthorhombic cell.")

	return (
		float(cell_matrix[0, 0]),
		float(cell_matrix[1, 1]),
		float(cell_matrix[2, 2]),
	)


def format_lammps_float(value: float) -> str:
	"""Format a floating-point value for the LAMMPS data file.

	Args:
		value: Floating-point value to serialize.

	Returns:
		A compact string representation suitable for ``conf.lmp``.
	"""
	return repr(float(value))


def write_lammps_conf_lmp(
	atoms: Atoms,
	output_path: str | Path,
	species_order: list[str],
	comment_line: str = "# LAMMPS data file written by initialize_structure_ASE_H2S.py",
) -> None:
	"""Write a LAMMPS ``atom_style atomic`` data file in the local MLMD format.

	Args:
		atoms: Structure to write.
		output_path: Destination path for the ``conf.lmp`` file.
		species_order: Preferred output ordering for chemical species.
		comment_line: Comment line written at the top of the file.
	"""
	output_file = Path(output_path)
	atoms_to_write = reorder_atoms_by_species(atoms=atoms, species_order=species_order)
	atoms_to_write = atoms_to_write.copy()
	atoms_to_write.wrap()

	ordered_species = infer_species_in_output_order(
		atoms=atoms_to_write,
		species_order=species_order,
	)
	type_id_by_symbol = {
		symbol: index + 1 for index, symbol in enumerate(ordered_species)
	}
	lx, ly, lz = validate_orthorhombic_cell(atoms=atoms_to_write)
	positions = np.array(atoms_to_write.get_positions(), dtype=float)
	symbols = atoms_to_write.get_chemical_symbols()

	lines: list[str] = [
		comment_line,
		"",
		f"{len(atoms_to_write)} atoms",
		f"{len(ordered_species)} atom types",
		"",
		f"0.0 {format_lammps_float(lx)} xlo xhi",
		f"0.0 {format_lammps_float(ly)} ylo yhi",
		f"0.0 {format_lammps_float(lz)} zlo zhi",
		"",
		"Atom Type Labels",
		"",
	]

	for symbol in ordered_species:
		lines.append(f"{type_id_by_symbol[symbol]} {symbol}")

	lines.extend(["", "Masses", ""])

	for symbol in ordered_species:
		lines.append(
			f"{type_id_by_symbol[symbol]} {format_lammps_float(get_lammps_mass(symbol))}  # {symbol}"
		)

	lines.extend(["", "Atoms  # atomic", ""])

	for atom_index, (symbol, position_vector) in enumerate(
		zip(symbols, positions),
		start=1,
	):
		lines.append(
			" ".join(
				[
					str(atom_index),
					str(type_id_by_symbol[symbol]),
					format_lammps_float(position_vector[0]),
					format_lammps_float(position_vector[1]),
					format_lammps_float(position_vector[2]),
				]
			)
		)

	output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_initialization_log(
	output_path: str | Path,
	atoms: Atoms,
	num_molecules: int,
	cell_lengths: tuple[float, float, float],
	bond_length: float,
	angle_deg: float,
	seed: int,
	grid_shape: tuple[int, int, int],
	species_order: list[str],
	cif_path: Path,
	poscar_path: Path,
	lammps_path: Path,
) -> None:
	"""Write a log file summarizing the H2S generation inputs.

	Args:
		output_path: Destination path for the log file.
		atoms: Generated H2S molecular structure.
		num_molecules: Number of H2S molecules requested.
		cell_lengths: Orthorhombic cell lengths in Angstrom.
		bond_length: S-H bond length in Angstrom.
		angle_deg: H-S-H angle in degrees.
		seed: Random seed used for orientations and grid-site shuffling.
		grid_shape: Grid shape used to place molecular centers.
		species_order: Requested species ordering in the written outputs.
		cif_path: Output CIF path.
		poscar_path: Output POSCAR path.
		lammps_path: Output LAMMPS data-file path.
	"""
	output_file = Path(output_path)
	timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
	lines: list[str] = [
		f"log.initialize written by {Path(__file__).name}",
		f"timestamp_utc = {timestamp_utc}",
		"",
		"[input_structure]",
		"phase = molecular H2S",
		f"num_molecules = {num_molecules}",
		f"cell_lengths_angstrom = {' '.join(f'{length:.10f}' for length in cell_lengths)}",
		f"bond_length_angstrom = {bond_length:.10f}",
		f"h_s_h_angle_degrees = {angle_deg:.10f}",
		f"seed = {seed}",
		f"grid_shape = {grid_shape[0]} {grid_shape[1]} {grid_shape[2]}",
		f"species_order = {' '.join(species_order)}",
		"",
		"[generated_structure]",
		f"natoms = {len(atoms)}",
		f"chemical_formula = {atoms.get_chemical_formula()}",
		f"cell_volume_angstrom3 = {atoms.get_volume():.10f}",
		"",
		"[output_files]",
		f"cif = {cif_path}",
		f"poscar = {poscar_path}",
		f"lammps = {lammps_path}",
	]
	output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_structure_outputs(
	atoms: Atoms,
	output_prefix: str,
	output_dir: Path,
	species_order: list[str],
) -> tuple[Path, Path, Path]:
	"""Write CIF, POSCAR, and LAMMPS data files for H2S.

	Args:
		atoms: Structure to write.
		output_prefix: Prefix used for all output filenames.
		output_dir: Directory where output files are written.
		species_order: Preferred chemical-species order.

	Returns:
		Paths to the generated CIF, POSCAR, and LAMMPS files.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	atoms_to_write = reorder_atoms_by_species(atoms=atoms, species_order=species_order)
	atoms_to_write.wrap()

	cif_path = output_dir / f"{output_prefix}.cif"
	poscar_path = output_dir / f"POSCAR_{output_prefix}"
	lammps_path = output_dir / f"conf.lmp_{output_prefix}"

	write(cif_path.as_posix(), atoms_to_write)
	write(poscar_path.as_posix(), atoms_to_write, direct=True, vasp5=True)
	normalize_poscar_header(
		poscar_path=poscar_path,
		comment_line="H2S_molecular",
		species_order=species_order,
	)
	write_lammps_conf_lmp(
		atoms=atoms_to_write,
		output_path=lammps_path,
		species_order=species_order,
	)
	return cif_path, poscar_path, lammps_path


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for the H2S structure generator.

	Returns:
		Parsed command-line arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Generate molecular H2S supercells and write CIF, POSCAR, and LAMMPS files."
	)
	parser.add_argument(
		"--num-molecules",
		type=int,
		default=64,
		help="Number of H2S molecules to generate. Default: %(default)s.",
	)
	parser.add_argument(
		"--cell-lengths",
		type=float,
		nargs=3,
		metavar=("LX", "LY", "LZ"),
		default=(12.0, 12.0, 12.0),
		help="Orthorhombic cell lengths along x, y, and z in Angstrom.",
	)
	parser.add_argument(
		"--bond-length",
		type=float,
		default=1.336,
		help="S-H bond length in Angstrom. Default: %(default)s.",
	)
	parser.add_argument(
		"--angle-deg",
		type=float,
		default=92.1,
		help="H-S-H bond angle in degrees. Default: %(default)s.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=7,
		help="Random seed for molecular orientations. Default: %(default)s.",
	)
	parser.add_argument(
		"--species-order",
		type=str,
		nargs="+",
		default=DEFAULT_SPECIES_ORDER,
		help="Species order for POSCAR and LAMMPS output. Default: %(default)s.",
	)
	parser.add_argument(
		"--output-prefix",
		type=str,
		default="H2S_molecular",
		help="Base prefix used for output filenames. Default: %(default)s.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("."),
		help="Directory where generated files are written. Default: current directory.",
	)
	parser.add_argument(
		"--min-interatomic-distance",
		type=float,
		default=1.0,
		help=(
			"Minimum allowed distance in Angstrom between any atom in a newly placed "
			"molecule and any previously placed atom. Prevents close-contact overlaps "
			"that cause VASP MLFF runs to explode. Default: %(default)s."
		),
	)
	parser.add_argument(
		"--max-rotation-retries",
		type=int,
		default=500,
		help=(
			"Maximum random rotations tried per molecule before aborting. "
			"Increase this or use a larger cell if placement fails. Default: %(default)s."
		),
	)
	return parser.parse_args()


def main() -> None:
	"""Build the requested H2S molecular supercell and write output files."""
	args = parse_args()
	cell_lengths = (
		float(args.cell_lengths[0]),
		float(args.cell_lengths[1]),
		float(args.cell_lengths[2]),
	)
	species_order = [str(symbol) for symbol in args.species_order]

	try:
		atoms, grid_shape = build_h2s_molecular_box(
			num_molecules=int(args.num_molecules),
			cell_lengths=cell_lengths,
			bond_length=float(args.bond_length),
			angle_deg=float(args.angle_deg),
			seed=int(args.seed),
			min_interatomic_distance=float(args.min_interatomic_distance),
			max_rotation_retries=int(args.max_rotation_retries),
		)
	except (ValueError, RuntimeError) as error:
		raise SystemExit(f"Error: {error}") from error

	atoms = reorder_atoms_by_species(atoms=atoms, species_order=species_order)
	molecules_count = len(atoms) // 3
	output_prefix = f"{args.output_prefix}_{molecules_count}molecules_{len(atoms)}atoms"
	cif_path, poscar_path, lammps_path = write_structure_outputs(
		atoms=atoms,
		output_prefix=output_prefix,
		output_dir=args.output_dir,
		species_order=species_order,
	)
	log_path = args.output_dir / f"log.initialize_{output_prefix}"
	write_initialization_log(
		output_path=log_path,
		atoms=atoms,
		num_molecules=molecules_count,
		cell_lengths=cell_lengths,
		bond_length=float(args.bond_length),
		angle_deg=float(args.angle_deg),
		seed=int(args.seed),
		grid_shape=grid_shape,
		species_order=species_order,
		cif_path=cif_path,
		poscar_path=poscar_path,
		lammps_path=lammps_path,
	)

	print(atoms)
	print(f"Number of H2S molecules: {molecules_count}")
	print(f"Number of atoms: {len(atoms)}")
	print(f"Placement grid used for molecular centers: {grid_shape}")
	print(f"Cell lengths (Angstrom): {atoms.cell.lengths()}")
	print(f"Wrote CIF:    {cif_path}")
	print(f"Wrote POSCAR: {poscar_path}")
	print(f"Wrote LAMMPS: {lammps_path}")
	print(f"Wrote log:    {log_path}")


if __name__ == "__main__":
	main()
