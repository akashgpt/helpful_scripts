#!/usr/bin/env python3
"""Generate literature-based cubic Ice X unit cells and supercells with ASE.

Ice X is the high-pressure symmetric hydrogen-bonded phase of H2O. This script
builds the idealized Ice X topology: cubic Pn-3m (#224) symmetry, an oxygen
body-centered-cubic sublattice (Wyckoff 2a), and hydrogen atoms centered exactly
halfway along symmetric O-H-O bonds (Wyckoff 4b). The conventional cubic cell
contains two H2O formula units (2 O + 4 H = 6 atoms).

Structure details
-----------------
- O at Wyckoff 2a: (0,0,0) and (1/2,1/2,1/2) — BCC sublattice
- H at Wyckoff 4b: (1/4,1/4,1/4), (3/4,3/4,1/4), (3/4,1/4,3/4), (1/4,3/4,3/4)
  Each H sits exactly midway between two BCC-nearest-neighbor O atoms;
  O-H distance = a*sqrt(3)/4; each O has 4 H neighbors.

Default lattice parameter and pressure
---------------------------------------
The default ``a = 2.78 A`` is taken from the ice VII equation-of-state at
~62 GPa (Loubeyre et al., Science 1999). This value is commonly used as an
approximate starting point for ice X models, but note two important caveats:

1. Transition pressure: For pure H2O, hydrogen-bond symmetrization (ice VII ->
   ice X) is now established to occur at ~80 GPa or above — not at 62 GPa as
   originally estimated by Hemley et al. (1987). At 62 GPa, pure H2O is still
   in the ice VII regime with asymmetrically displaced hydrogens. The 62 GPa
   figure is more applicable to H2O mixtures (e.g., H2O-He: 60-70 GPa).
   See: Salzmann et al., Nat. Commun. 13, 4976 (2022).

2. Lattice parameter at true ice X conditions: At the actual ice X transition
   pressure (~80-100 GPa for pure H2O), the equilibrium lattice parameter is
   smaller (~2.65-2.72 A). Use the --a flag to set a pressure-appropriate value
   for production simulations.

The Pn-3m structure generated here (H exactly centered) is the correct idealized
model for ice X regardless of the starting a; the thermodynamic state point must
be chosen by the user via --a and the simulation pressure.

Usage:
	python initialize_structure_ASE_Ice_X.py
	python initialize_structure_ASE_Ice_X.py --reps 4 4 4 --a 3.20 --output-dir ./ice_x
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
	"O": 15.9994,
	"H": 1.00794,
}

DEFAULT_ICE_X_LATTICE_ANGSTROM: float = 2.78
# 2.78 A corresponds to the ice VII EOS at ~62 GPa (Loubeyre et al. 1999).
# At true ice X conditions for pure H2O (~80-100 GPa), a is smaller (~2.65-2.72 A).
# This value is used as a widely-cited approximate starting geometry for ice X models.

DEFAULT_ICE_X_PRESSURE_GPA: float = 62.0
# 62 GPa is the early Hemley (1987) estimate for the ice VII -> X transition.
# Modern experiments and simulations revise this to ~80 GPa for pure H2O.
# (60-70 GPa applies to H2O mixed with He or H2, not pure H2O.)
# See: Salzmann et al., Nat. Commun. 13, 4976 (2022).

DEFAULT_ICE_X_TEMPERATURE_K: float = 300.0
DEFAULT_SPECIES_ORDER: list[str] = ["H", "O"]
ICE_X_SPACE_GROUP_DESCRIPTION: str = "Pn-3m (#224), cubic Ice X"
# O at Wyckoff 2a: (0,0,0), (1/2,1/2,1/2)
# H at Wyckoff 4b: (1/4,1/4,1/4), (3/4,3/4,1/4), (3/4,1/4,3/4), (1/4,3/4,3/4)
# Below ~300 GPa the structure is cubic Pn-3m; above ~300 GPa it distorts to
# orthorhombic P42/nnm (Salzmann et al. 2022).

ICE_X_REFERENCE_CITATION: str = (
	"Structure (Pn-3m, Wyckoff 2a O + 4b H): Hemley et al., Nature 330, 737-740 (1987) — "
	"discovery of ice X; Loubeyre et al., Science 284, 1242 (1999) — X-ray EOS and "
	"a~2.78 A at ~62 GPa. "
	"Revised transition pressure (~80 GPa for pure H2O): "
	"Salzmann et al., Nat. Commun. 13, 4976 (2022). "
	"Note: at 62 GPa pure H2O is in the ice VII regime; true ice X (centered H) "
	"requires >80 GPa, where a ~ 2.65-2.72 A."
)
ICE_X_REFERENCE_URL: str = "https://doi.org/10.1038/330737a0"
# Additional key references:
#   Loubeyre et al. (1999): https://doi.org/10.1126/science.284.5420.1660
#   Salzmann et al. (2022): https://doi.org/10.1038/s41467-022-32374-1


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


def validate_supercell_repetitions(reps: tuple[int, int, int]) -> tuple[int, int, int]:
	"""Validate supercell repetition counts.

	Args:
		reps: Requested repetition counts along the three cubic cell vectors.

	Returns:
		The validated repetition counts.

	Raises:
		ValueError: If ``reps`` does not contain exactly three positive integers.
	"""
	if len(reps) != 3:
		raise ValueError("Supercell repetitions must contain exactly three integers.")
	if any(rep_count <= 0 for rep_count in reps):
		raise ValueError("Supercell repetitions must all be positive integers.")
	return reps


def build_ice_x_unit_cell(a: float = DEFAULT_ICE_X_LATTICE_ANGSTROM) -> Atoms:
	"""Build one ideal conventional cubic Ice X unit cell.

	The two oxygen atoms form a bcc sublattice (Wyckoff 2a of Pn-3m #224).
	The four hydrogen positions lie exactly halfway along symmetric O-H-O links
	(Wyckoff 4b), giving H2O stoichiometry in the conventional cell.
	O-H bond length = a*sqrt(3)/4; each O has 4 H neighbors.
	The default ``a`` (2.78 A) is from the ice VII EOS at ~62 GPa (Loubeyre
	et al. 1999). True ice X for pure H2O requires >80 GPa where a is smaller
	(~2.65-2.72 A); see module docstring for details.

	Args:
		a: Conventional cubic lattice parameter in Angstrom.

	Returns:
		An ASE ``Atoms`` object containing the 6-atom Ice X conventional cell.
	"""
	validated_a = validate_positive_float(value=a, value_name="Ice X lattice parameter")
	symbols: list[str] = ["O", "O", "H", "H", "H", "H"]
	scaled_positions = np.array(
		[
			[0.0, 0.0, 0.0],
			[0.5, 0.5, 0.5],
			[0.25, 0.25, 0.25],
			[0.75, 0.75, 0.25],
			[0.75, 0.25, 0.75],
			[0.25, 0.75, 0.75],
		],
		dtype=float,
	)
	atoms = Atoms(
		symbols=symbols,
		scaled_positions=scaled_positions,
		cell=np.diag([validated_a, validated_a, validated_a]),
		pbc=True,
	)
	atoms.wrap()
	return atoms


def build_ice_x_supercell(
	a: float = DEFAULT_ICE_X_LATTICE_ANGSTROM,
	reps: tuple[int, int, int] = (3, 3, 3),
) -> Atoms:
	"""Build an Ice X supercell by repeating the conventional cubic cell.

	Args:
		a: Conventional cubic lattice parameter in Angstrom.
		reps: Number of repeats along ``x``, ``y``, and ``z``.

	Returns:
		An ASE ``Atoms`` object containing the requested Ice X supercell.
	"""
	validated_reps = validate_supercell_repetitions(reps=reps)
	unit_cell = build_ice_x_unit_cell(a=a)
	supercell = unit_cell.repeat(validated_reps)
	supercell.wrap()
	return supercell


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
	comment_line: str = "# LAMMPS data file written by initialize_structure_ASE_Ice_X.py",
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
	a: float,
	reps: tuple[int, int, int],
	species_order: list[str],
	cif_path: Path,
	poscar_path: Path,
	lammps_path: Path,
) -> None:
	"""Write a log file summarizing the Ice X generation inputs.

	Args:
		output_path: Destination path for the log file.
		atoms: Generated Ice X structure.
		a: Conventional cubic lattice parameter in Angstrom.
		reps: Supercell repetitions along ``x``, ``y``, and ``z``.
		species_order: Requested species ordering in the written outputs.
		cif_path: Output CIF path.
		poscar_path: Output POSCAR path.
		lammps_path: Output LAMMPS data-file path.
	"""
	output_file = Path(output_path)
	cell_lengths = np.array(atoms.cell.lengths(), dtype=float)
	timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
	lines: list[str] = [
		f"log.initialize written by {Path(__file__).name}",
		f"timestamp_utc = {timestamp_utc}",
		"",
		"[input_structure]",
		"phase = H2O Ice X",
		f"space_group = {ICE_X_SPACE_GROUP_DESCRIPTION}",
		f"reference_conditions = {DEFAULT_ICE_X_PRESSURE_GPA:.1f} GPa, {DEFAULT_ICE_X_TEMPERATURE_K:.1f} K",
		f"reference_citation = {ICE_X_REFERENCE_CITATION}",
		f"reference_url = {ICE_X_REFERENCE_URL}",
		"structure_model = bcc oxygen sublattice with centered symmetric O-H-O hydrogens",
		f"a_angstrom = {a:.10f}",
		f"default_literature_a_angstrom = {DEFAULT_ICE_X_LATTICE_ANGSTROM:.10f}",
		f"supercell_reps_abc = {reps[0]} {reps[1]} {reps[2]}",
		f"species_order = {' '.join(species_order)}",
		"",
		"[generated_structure]",
		f"natoms = {len(atoms)}",
		f"chemical_formula = {atoms.get_chemical_formula()}",
		f"cell_lengths_angstrom = {' '.join(f'{length:.10f}' for length in cell_lengths)}",
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
	"""Write CIF, POSCAR, and LAMMPS data files for Ice X.

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
		comment_line="Ice_X",
		species_order=species_order,
	)
	write_lammps_conf_lmp(
		atoms=atoms_to_write,
		output_path=lammps_path,
		species_order=species_order,
	)
	return cif_path, poscar_path, lammps_path


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for the Ice X structure generator.

	Returns:
		Parsed command-line arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Generate cubic Ice X supercells and write CIF, POSCAR, and LAMMPS files."
	)
	parser.add_argument(
		"--a",
		type=float,
		default=DEFAULT_ICE_X_LATTICE_ANGSTROM,
		help=(
			"Conventional cubic Ice X lattice parameter in Angstrom. "
			"Default is the literature 62 GPa, 300 K value: %(default)s."
		),
	)
	parser.add_argument(
		"--reps",
		type=int,
		nargs=3,
		metavar=("NX", "NY", "NZ"),
		default=(3, 3, 3),
		help="Supercell repetitions along x, y, and z. Default: %(default)s.",
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
		default="Ice_X",
		help="Base prefix used for output filenames. Default: %(default)s.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("."),
		help="Directory where generated files are written. Default: current directory.",
	)
	return parser.parse_args()


def main() -> None:
	"""Build the requested Ice X supercell and write output files."""
	args = parse_args()
	reps = (int(args.reps[0]), int(args.reps[1]), int(args.reps[2]))
	species_order = [str(symbol) for symbol in args.species_order]

	try:
		atoms = build_ice_x_supercell(a=float(args.a), reps=reps)
	except ValueError as error:
		raise SystemExit(f"Error: {error}") from error

	atoms = reorder_atoms_by_species(atoms=atoms, species_order=species_order)
	output_prefix = (
		f"{args.output_prefix}_{reps[0]}x{reps[1]}x{reps[2]}_"
		f"{len(atoms)}atoms"
	)
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
		a=float(args.a),
		reps=reps,
		species_order=species_order,
		cif_path=cif_path,
		poscar_path=poscar_path,
		lammps_path=lammps_path,
	)

	print(atoms)
	print(f"Number of H2O formula units: {len(atoms) // 3}")
	print(f"Number of atoms: {len(atoms)}")
	print(f"Cell lengths (Angstrom): {atoms.cell.lengths()}")
	print(f"Wrote CIF:    {cif_path}")
	print(f"Wrote POSCAR: {poscar_path}")
	print(f"Wrote LAMMPS: {lammps_path}")
	print(f"Wrote log:    {log_path}")


if __name__ == "__main__":
	main()
