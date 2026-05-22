"""Convert a LAMMPS ``conf.lmp`` data file to a VASP POSCAR.

This converter preserves the local QMD convention of reading species identity
from the ``Atom Type Labels`` section in ``conf.lmp`` whenever it is available.
That avoids ambiguous type-ID-to-element guesses when converting mixed systems.
"""

import argparse
from pathlib import Path

from ase.data import atomic_numbers
from ase.io import read, write


def parse_arguments() -> argparse.Namespace:
	"""Parse command-line arguments.

	Returns:
		Parsed command-line arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Convert a LAMMPS data file such as conf.lmp to a VASP POSCAR."
	)
	parser.add_argument(
		"--LAMMPS_input",
		"-li",
		default="conf.lmp",
		type=str,
		help="Input LAMMPS data file. Default: conf.lmp",
	)
	parser.add_argument(
		"--VASP_input",
		"-vi",
		default="POSCAR.ase",
		type=str,
		help="Output VASP POSCAR file. Default: POSCAR.ase",
	)
	return parser.parse_args()


def extract_atom_type_labels(lammps_input: Path) -> list[str]:
	"""Extract ordered element labels from a LAMMPS ``Atom Type Labels`` section.

	Args:
		lammps_input: Path to the LAMMPS data file.

	Returns:
		Element labels ordered by LAMMPS atom type ID.

	Raises:
		ValueError: If the ``Atom Type Labels`` section is missing or invalid.
	"""
	labels: list[str] = []
	in_labels_section: bool = False

	with lammps_input.open("r", encoding="utf-8", errors="replace") as input_file:
		for line in input_file:
			stripped_line: str = line.strip()
			if stripped_line == "Atom Type Labels":
				in_labels_section = True
				continue
			if not in_labels_section:
				continue
			if not stripped_line:
				continue

			fields: list[str] = stripped_line.split()
			if len(fields) >= 2 and fields[0].isdigit():
				labels.append(fields[1])
				continue
			if labels:
				break

	if not labels:
		raise ValueError(
			f"No Atom Type Labels section found in {lammps_input}. "
			"Add Atom Type Labels or convert with an explicit type map."
		)
	return labels


def build_z_of_type(atom_type_labels: list[str]) -> dict[int, int]:
	"""Build an ASE ``Z_of_type`` mapping from LAMMPS atom type labels.

	Args:
		atom_type_labels: Element labels ordered by LAMMPS atom type ID.

	Returns:
		Mapping from one-based LAMMPS type ID to atomic number.

	Raises:
		ValueError: If a label is not a valid element symbol.
	"""
	z_of_type: dict[int, int] = {}
	for type_id, label in enumerate(atom_type_labels, start=1):
		if label not in atomic_numbers:
			raise ValueError(f"Unknown element label for LAMMPS type {type_id}: {label}")
		z_of_type[type_id] = atomic_numbers[label]
	return z_of_type


def convert_lammps_to_poscar(lammps_input: Path, vasp_output: Path) -> None:
	"""Convert a LAMMPS data file to a VASP POSCAR file.

	Args:
		lammps_input: Input LAMMPS data file.
		vasp_output: Output POSCAR path.
	"""
	atom_type_labels: list[str] = extract_atom_type_labels(lammps_input=lammps_input)
	z_of_type: dict[int, int] = build_z_of_type(atom_type_labels=atom_type_labels)
	atoms = read(
		lammps_input.as_posix(),
		format="lammps-data",
		atom_style="atomic",
		Z_of_type=z_of_type,
	)
	atoms.wrap()
	write(vasp_output.as_posix(), atoms, format="vasp", direct=True, vasp5=True)


def main() -> None:
	"""Run the LAMMPS-to-VASP conversion."""
	args: argparse.Namespace = parse_arguments()
	lammps_input = Path(args.LAMMPS_input)
	vasp_output = Path(args.VASP_input)
	convert_lammps_to_poscar(lammps_input=lammps_input, vasp_output=vasp_output)


if __name__ == "__main__":
	main()
