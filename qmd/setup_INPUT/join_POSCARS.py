#!/usr/bin/env python3
"""Stitch two POSCAR-like files together along the z axis.

This script mirrors the CLI style and validation logic of
``join_conf_lmps.py`` but writes a stitched VASP POSCAR instead of a LAMMPS
``conf.lmp`` file. It uses ``ase.Atoms`` for reading, validation, z-axis
translation, species reordering, and final POSCAR writing.

Examples:
    Stitch two explicit input files:

    ``python join_POSCARS.py -f1 slab1/POSCAR -f2 slab2/POSCAR -so "Mg Si O H N"``

    Stitch with an explicit output directory and a larger z-gap:

    ``python join_POSCARS.py -f1 slab1/CONTCAR -f2 slab2/CONTCAR -do stitched_out -g 0.75 -so "Mg Si O H N"``

    Stitch two different filenames and write a named output file:

    ``python join_POSCARS.py -f1 /tmp/NH3_POSCAR -f2 /tmp/MgSiO3_POSCAR -do out_dir -o POSCAR_joint -so "Mg Si O H N"``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.vasp import read_vasp, write_vasp


@dataclass(frozen=True)
class XYCompatibilityReport:
    """Store validated in-plane cell lengths for z-axis stitching.

    Attributes:
        input1_lx: First input x length in Angstrom.
        input1_ly: First input y length in Angstrom.
        input2_lx: Second input x length in Angstrom.
        input2_ly: Second input y length in Angstrom.
    """

    input1_lx: float
    input1_ly: float
    input2_lx: float
    input2_ly: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Join two POSCAR-like files along the z axis."
    )
    parser.add_argument(
        "-f1",
        "--file1",
        type=str,
        required=True,
        help="Path to the first input POSCAR-like file.",
    )
    parser.add_argument(
        "-f2",
        "--file2",
        type=str,
        required=True,
        help="Path to the second input POSCAR-like file.",
    )
    parser.add_argument(
        "-do",
        "--dir_out",
        type=str,
        default="joint_POSCAR",
        help="Output directory. Default: %(default)s.",
    )
    parser.add_argument(
        "-g",
        "--gap",
        type=float,
        default=0.50,
        help="Gap between the two cells in Angstrom. Default: %(default)s.",
    )
    parser.add_argument(
        "-so",
        "--species_order",
        type=str,
        required=True,
        help='Required output species order in quotes, for example: "Mg Si O H N".',
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="POSCAR",
        help="Output POSCAR filename inside dir_out. Default: %(default)s.",
    )
    return parser.parse_args()


def validate_gap(gap: float) -> float:
    """Validate the requested inter-slab gap.

    Args:
        gap: Requested gap in Angstrom.

    Returns:
        The validated gap value.

    Raises:
        ValueError: If the gap is negative.
    """
    if gap < 0.0:
        raise ValueError("The gap must be non-negative.")
    return gap


def parse_species_order(order_text: str) -> list[str]:
    """Parse the requested output species order.

    Args:
        order_text: User-provided species order string, for example
            ``"Mg Si O H N"``.

    Returns:
        Species labels in the requested output order.

    Raises:
        ValueError: If the order string is empty or contains duplicates.
    """
    species_order: list[str] = [token for token in order_text.strip().split() if token]
    if not species_order:
        raise ValueError(
            'The species order is empty. Pass something like '
            '--species_order "Mg Si O H N".'
        )
    if len(species_order) != len(set(species_order)):
        raise ValueError("The species order contains duplicate labels.")
    return species_order


def validate_requested_species_order(
    requested_order: list[str],
    available_species: list[str],
) -> list[str]:
    """Validate that the requested output order matches the input species set.

    Args:
        requested_order: Species order requested by the user.
        available_species: Species labels present in the stitched inputs.

    Returns:
        The validated species order.

    Raises:
        ValueError: If the requested order is missing species or includes extras.
    """
    requested_species_set: set[str] = set(requested_order)
    available_species_set: set[str] = set(available_species)

    missing_species: list[str] = sorted(available_species_set - requested_species_set)
    extra_species: list[str] = sorted(requested_species_set - available_species_set)

    if missing_species or extra_species:
        error_lines: list[str] = [
            (
                "The requested --species_order must contain exactly the species "
                "present in the inputs."
            ),
            f"Requested order: {' '.join(requested_order)}",
            f"Available species: {' '.join(available_species)}",
        ]
        if missing_species:
            error_lines.append(f"Missing species: {' '.join(missing_species)}")
        if extra_species:
            error_lines.append(f"Extra species: {' '.join(extra_species)}")
        raise ValueError("\n".join(error_lines))

    return requested_order


def validate_xy_compatibility(
    atoms1: Atoms,
    atoms2: Atoms,
    tolerance: float = 1.0e-8,
) -> XYCompatibilityReport:
    """Validate that two cells can be stitched directly along z.

    Args:
        atoms1: First structure.
        atoms2: Second structure.
        tolerance: Absolute tolerance for x/y length comparisons.

    Returns:
        Validated in-plane cell lengths for both inputs.

    Raises:
        ValueError: If either structure is not square in the xy plane, or if
            the two structures have different x/y lengths.
    """
    input1_lx: float = float(atoms1.cell[0, 0])
    input1_ly: float = float(atoms1.cell[1, 1])
    input2_lx: float = float(atoms2.cell[0, 0])
    input2_ly: float = float(atoms2.cell[1, 1])

    if abs(input1_lx - input1_ly) > tolerance:
        raise ValueError(
            "The first input cell must have equal x and y lengths for z-axis stitching."
        )
    if abs(input2_lx - input2_ly) > tolerance:
        raise ValueError(
            "The second input cell must have equal x and y lengths for z-axis stitching."
        )
    if abs(input1_lx - input2_lx) > tolerance:
        raise ValueError("The two input cells have different x lengths.")
    if abs(input1_ly - input2_ly) > tolerance:
        raise ValueError("The two input cells have different y lengths.")

    return XYCompatibilityReport(
        input1_lx=input1_lx,
        input1_ly=input1_ly,
        input2_lx=input2_lx,
        input2_ly=input2_ly,
    )


def merge_available_species(atoms1: Atoms, atoms2: Atoms) -> list[str]:
    """Collect species labels from two structures while preserving encounter order.

    Args:
        atoms1: First structure.
        atoms2: Second structure.

    Returns:
        Species labels in first-seen order across both inputs.
    """
    ordered_labels: list[str] = []
    seen_labels: set[str] = set()

    for symbol in atoms1.get_chemical_symbols() + atoms2.get_chemical_symbols():
        if symbol not in seen_labels:
            ordered_labels.append(symbol)
            seen_labels.add(symbol)

    return ordered_labels


def reorder_atoms_by_species(atoms: Atoms, species_order: list[str]) -> Atoms:
    """Group atoms by a requested species ordering.

    Args:
        atoms: Input structure to reorder.
        species_order: Chemical symbols in the desired output order.

    Returns:
        A reordered copy of the input structure.
    """
    symbols: np.ndarray = np.array(atoms.get_chemical_symbols())
    ordered_indices: list[int] = []

    for symbol in species_order:
        ordered_indices.extend(np.where(symbols == symbol)[0].tolist())

    remaining_indices: list[int] = [
        index for index, symbol in enumerate(symbols) if symbol not in species_order
    ]
    ordered_indices.extend(remaining_indices)

    return atoms[ordered_indices]


def stitch_atoms_along_z(
    atoms1: Atoms,
    atoms2: Atoms,
    gap: float,
    requested_species_order: list[str],
) -> tuple[Atoms, list[str], XYCompatibilityReport]:
    """Stitch two POSCAR structures together along the z axis.

    Args:
        atoms1: First input structure.
        atoms2: Second input structure.
        gap: Gap between the two slabs in Angstrom.
        requested_species_order: User-requested output species order.

    Returns:
        A tuple containing:
            1. The stitched ``ase.Atoms`` object.
            2. Ordered merged species labels.
            3. Validated x/y compatibility report.
    """
    validated_gap: float = validate_gap(gap=gap)
    xy_report: XYCompatibilityReport = validate_xy_compatibility(atoms1=atoms1, atoms2=atoms2)

    atoms1 = atoms1.copy()
    atoms2 = atoms2.copy()
    atoms1.wrap()
    atoms2.wrap()

    shift_z: float = float(atoms1.cell[2, 2]) + validated_gap
    atoms2.translate([0.0, 0.0, shift_z])

    combined_atoms: Atoms = atoms1 + atoms2
    combined_cell: np.ndarray = atoms1.cell.array.copy()
    combined_cell[2, 2] = float(atoms1.cell[2, 2]) + float(atoms2.cell[2, 2]) + validated_gap
    combined_atoms.set_cell(combined_cell, scale_atoms=False)
    combined_atoms.wrap()

    available_species_order: list[str] = merge_available_species(atoms1=atoms1, atoms2=atoms2)
    species_order: list[str] = validate_requested_species_order(
        requested_order=requested_species_order,
        available_species=available_species_order,
    )
    combined_atoms = reorder_atoms_by_species(combined_atoms, species_order)
    return combined_atoms, species_order, xy_report


def format_float(value: float) -> str:
    """Format a floating-point value for user-facing output.

    Args:
        value: Floating-point value to serialize.

    Returns:
        A compact string representation.
    """
    return repr(float(value))


def format_xy_compatibility_report(xy_report: XYCompatibilityReport) -> str:
    """Build a human-readable x/y compatibility summary line.

    Args:
        xy_report: Validated in-plane cell length report.

    Returns:
        A short summary string for stdout or log files.
    """
    return (
        "xy_compatibility: PASS "
        f"(input1: {format_float(xy_report.input1_lx)} x {format_float(xy_report.input1_ly)} A; "
        f"input2: {format_float(xy_report.input2_lx)} x {format_float(xy_report.input2_ly)} A)"
    )


def write_poscar(
    atoms: Atoms,
    output_path: Path,
    species_order: list[str],
) -> None:
    """Write a stitched POSCAR file with a stable species ordering.

    Args:
        atoms: Structure to write.
        output_path: Destination POSCAR path.
        species_order: Desired species order in the output file.
    """
    ordered_atoms: Atoms = reorder_atoms_by_species(atoms=atoms, species_order=species_order)
    ordered_atoms = ordered_atoms.copy()
    ordered_atoms.wrap()
    write_vasp(str(output_path), ordered_atoms, direct=True, sort=False)


def write_log_file(
    output_dir: Path,
    input1: Path,
    input2: Path,
    output_file: str,
    gap: float,
    species_order: list[str],
    xy_report: XYCompatibilityReport,
) -> None:
    """Write a small execution log beside the output POSCAR.

    Args:
        output_dir: Output directory.
        input1: First input file path.
        input2: Second input file path.
        output_file: Output POSCAR filename.
        gap: Gap used between the two cells.
        species_order: User-requested output species order.
        xy_report: Validated in-plane cell length report.
    """
    log_lines: list[str] = [
        f"input1: {input1}",
        f"input2: {input2}",
        f"output_dir: {output_dir}",
        f"output_file: {output_file}",
        f"gap_between_cells: {gap}",
        f"species_order: {' '.join(species_order)}",
        format_xy_compatibility_report(xy_report=xy_report),
    ]
    (output_dir / "log.join_POSCARS").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the z-axis POSCAR stitcher."""
    args: argparse.Namespace = parse_args()

    input1: Path = Path(args.file1.strip()).expanduser()
    input2: Path = Path(args.file2.strip()).expanduser()
    dir_out: Path = Path(args.dir_out.strip()).expanduser()
    output_file: str = args.output_file.strip()
    requested_species_order: list[str] = parse_species_order(args.species_order)

    output_path: Path = dir_out / output_file

    dir_out.mkdir(parents=True, exist_ok=True)

    print(f"file1: {input1}")
    print(f"file2: {input2}")
    print(f"dir_out: {dir_out}")
    print(f"gap_between_cells: {args.gap}")
    print(f"species_order: {' '.join(requested_species_order)}")

    atoms1: Atoms = read_vasp(str(input1))
    atoms2: Atoms = read_vasp(str(input2))

    combined_atoms, species_order, xy_report = stitch_atoms_along_z(
        atoms1=atoms1,
        atoms2=atoms2,
        gap=args.gap,
        requested_species_order=requested_species_order,
    )

    print(format_xy_compatibility_report(xy_report=xy_report))

    write_poscar(
        atoms=combined_atoms,
        output_path=output_path,
        species_order=species_order,
    )
    write_log_file(
        output_dir=dir_out,
        input1=input1,
        input2=input2,
        output_file=output_file,
        gap=args.gap,
        species_order=species_order,
        xy_report=xy_report,
    )

    print(f"Merged structures from {input1} and {input2} into {output_path}")


if __name__ == "__main__":
    main()
