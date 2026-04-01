#!/usr/bin/env python3
"""Stitch two ``conf.lmp`` files together along the z axis.

This script mirrors the role of ``join_POSCARS.py`` but works directly on
LAMMPS ``atom_style atomic`` data files. It uses ``ase.Atoms`` for the
structure manipulation steps, while preserving explicit control over the final
LAMMPS formatting so the output remains easy for OVITO and the local MLMD
workflow to read.

Examples:
    Stitch two explicit input files:

    `` python join_conf_lmps.py -f1 slab1/conf.lmp -f2 slab2/conf.lmp -so "Mg Si O H N" ``
    `` python join_conf_lmps.py -f1 slab1/conf.lmp -f2 slab2/conf.lmp -so "H N" ``

    Stitch with an explicit output directory and a larger z-gap:

    ``python join_conf_lmps.py -f1 slab1/conf.lmp -f2 slab2/conf.lmp -do stitched_out -g 0.75 -so "Mg Si O H N"``

    Stitch two different filenames and write a named output file:

    ``python join_conf_lmps.py -f1 /tmp/NH3_conf.lmp -f2 /tmp/MgSiO3_conf.lmp -do out_dir -o conf.lmp -so "Mg Si O H N"``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


SECTION_HEADERS: set[str] = {
    "Atom Type Labels",
    "Masses",
    "Pair Coeffs",
    "PairIJ Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "BondBond Coeffs",
    "BondAngle Coeffs",
    "MiddleBondTorsion Coeffs",
    "EndBondTorsion Coeffs",
    "AngleTorsion Coeffs",
    "AngleAngleTorsion Coeffs",
    "BondBond13 Coeffs",
    "AngleAngle Coeffs",
    "Atoms",
    "Velocities",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Ellipsoids",
    "Lines",
    "Triangles",
    "Bodies",
}


@dataclass(frozen=True)
class AtomRecord:
    """Store one ``atom_style atomic`` atom line.

    Attributes:
        atom_id: Atom ID from the input file.
        atom_type: Integer LAMMPS type ID.
        x: Cartesian x coordinate in Angstrom.
        y: Cartesian y coordinate in Angstrom.
        z: Cartesian z coordinate in Angstrom.
    """

    atom_id: int
    atom_type: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BoxBounds:
    """Store orthorhombic LAMMPS box bounds.

    Attributes:
        xlo: Lower x bound.
        xhi: Upper x bound.
        ylo: Lower y bound.
        yhi: Upper y bound.
        zlo: Lower z bound.
        zhi: Upper z bound.
    """

    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float

    def lx(self) -> float:
        """Return the x box length.

        Returns:
            The x length in Angstrom.
        """
        return self.xhi - self.xlo

    def ly(self) -> float:
        """Return the y box length.

        Returns:
            The y length in Angstrom.
        """
        return self.yhi - self.ylo

    def lz(self) -> float:
        """Return the z box length.

        Returns:
            The z length in Angstrom.
        """
        return self.zhi - self.zlo


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


@dataclass(frozen=True)
class LammpsAtomicData:
    """Store parsed ``atom_style atomic`` LAMMPS data.

    Attributes:
        comment_line: First line of the source file.
        bounds: Orthorhombic box bounds.
        type_labels: Mapping from type ID to chemical symbol.
        masses_by_label: Mapping from chemical symbol to mass.
        atoms: Parsed atom lines from the ``Atoms`` section.
    """

    comment_line: str
    bounds: BoxBounds
    type_labels: dict[int, str]
    masses_by_label: dict[str, float]
    atoms: list[AtomRecord]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Join two LAMMPS conf.lmp files along the z axis."
    )
    parser.add_argument(
        "-f1",
        "--file1",
        type=str,
        required=True,
        help="Path to the first input conf.lmp file.",
    )
    parser.add_argument(
        "-f2",
        "--file2",
        type=str,
        required=True,
        help="Path to the second input conf.lmp file.",
    )
    parser.add_argument(
        "-do",
        "--dir_out",
        type=str,
        default="joint_conf_lmp",
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
        dest="species_order",
        required=True,
        help=(
            'Required output species order in quotes, for example: '
            '"Mg Si O H N". Deprecated aliases: -o/--order.'
        ),
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="conf.lmp",
        help="Output LAMMPS filename inside dir_out. Default: %(default)s.",
    )
    return parser.parse_args()


def is_named_section_header(line: str, section_name: str) -> bool:
    """Check whether a line matches a specific LAMMPS section header.

    Args:
        line: Candidate line from the input file.
        section_name: Section header name to match.

    Returns:
        ``True`` if the line matches the named section header.
    """
    stripped_line: str = line.strip()
    if stripped_line == section_name:
        return True
    if stripped_line.startswith(section_name):
        remainder: str = stripped_line[len(section_name):]
        if not remainder or remainder.lstrip().startswith("#"):
            return True
    return False


def is_section_header(line: str) -> bool:
    """Check whether a line begins any supported LAMMPS section.

    Args:
        line: Candidate line from the input file.

    Returns:
        ``True`` if the line is recognized as a section header.
    """
    return any(is_named_section_header(line=line, section_name=name) for name in SECTION_HEADERS)


def strip_inline_comment(line: str) -> str:
    """Remove any inline ``#`` comment from a line.

    Args:
        line: Input line that may contain an inline comment.

    Returns:
        The uncommented portion of the line with surrounding whitespace removed.
    """
    return line.split("#", 1)[0].strip()


def extract_inline_comment(line: str) -> str:
    """Extract the inline comment text from a line.

    Args:
        line: Input line that may contain an inline comment.

    Returns:
        The stripped inline comment text, or an empty string if absent.
    """
    if "#" not in line:
        return ""
    return line.split("#", 1)[1].strip()


def find_section_index(lines: list[str], section_name: str) -> int | None:
    """Find the line index of a named section header.

    Args:
        lines: Full file contents split into lines.
        section_name: Section header to locate.

    Returns:
        The zero-based line index if found, otherwise ``None``.
    """
    for index, line in enumerate(lines):
        if is_named_section_header(line=line, section_name=section_name):
            return index
    return None


def read_section_block(lines: list[str], section_name: str) -> list[str]:
    """Read the body lines for a named LAMMPS section.

    Args:
        lines: Full file contents split into lines.
        section_name: Section header name to read.

    Returns:
        Section-body lines, excluding the header line itself.

    Raises:
        ValueError: If the requested section is missing.
    """
    header_index: int | None = find_section_index(lines=lines, section_name=section_name)
    if header_index is None:
        raise ValueError(f"Section '{section_name}' was not found in the LAMMPS file.")

    body_start: int = header_index + 1
    while body_start < len(lines):
        if lines[body_start].strip():
            break
        body_start += 1

    body_end: int = body_start
    while body_end < len(lines):
        if is_section_header(lines[body_end]):
            break
        body_end += 1

    return lines[body_start:body_end]


def parse_box_bounds(lines: list[str]) -> BoxBounds:
    """Parse orthorhombic box bounds from a LAMMPS data file.

    Args:
        lines: Full file contents split into lines.

    Returns:
        Parsed orthorhombic box bounds.

    Raises:
        ValueError: If bounds are missing or a tilted box is detected.
    """
    xlo: float | None = None
    xhi: float | None = None
    ylo: float | None = None
    yhi: float | None = None
    zlo: float | None = None
    zhi: float | None = None

    for line in lines:
        fields: list[str] = line.split()
        if len(fields) >= 6 and fields[3] == "xy" and fields[4] == "xz" and fields[5] == "yz":
            raise ValueError("Tilted boxes are not supported; this stitcher expects orthorhombic conf.lmp files.")
        if len(fields) >= 4 and fields[2] == "xlo" and fields[3] == "xhi":
            xlo, xhi = float(fields[0]), float(fields[1])
        elif len(fields) >= 4 and fields[2] == "ylo" and fields[3] == "yhi":
            ylo, yhi = float(fields[0]), float(fields[1])
        elif len(fields) >= 4 and fields[2] == "zlo" and fields[3] == "zhi":
            zlo, zhi = float(fields[0]), float(fields[1])

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError("Could not parse x/y/z box bounds from the LAMMPS header.")

    return BoxBounds(
        xlo=float(xlo),
        xhi=float(xhi),
        ylo=float(ylo),
        yhi=float(yhi),
        zlo=float(zlo),
        zhi=float(zhi),
    )


def parse_atom_type_labels(lines: list[str]) -> dict[int, str]:
    """Parse the ``Atom Type Labels`` section.

    Args:
        lines: Full file contents split into lines.

    Returns:
        Mapping from type ID to chemical symbol. Returns an empty mapping if the
        section is absent.
    """
    header_index: int | None = find_section_index(lines=lines, section_name="Atom Type Labels")
    if header_index is None:
        return {}

    labels_block: list[str] = read_section_block(lines=lines, section_name="Atom Type Labels")
    type_labels: dict[int, str] = {}

    for line in labels_block:
        stripped_line: str = strip_inline_comment(line)
        if not stripped_line:
            continue
        fields: list[str] = stripped_line.split()
        if len(fields) >= 2:
            type_labels[int(fields[0])] = fields[1]

    return type_labels


def parse_masses(
    lines: list[str],
    type_labels: dict[int, str],
) -> tuple[dict[str, float], dict[int, str]]:
    """Parse the ``Masses`` section and finalize the type-label mapping.

    Args:
        lines: Full file contents split into lines.
        type_labels: Optional labels parsed from ``Atom Type Labels``.

    Returns:
        A tuple containing:
            1. Mapping from chemical symbol to mass.
            2. Final mapping from type ID to chemical symbol.

    Raises:
        ValueError: If a valid chemical symbol cannot be inferred for a type.
    """
    masses_block: list[str] = read_section_block(lines=lines, section_name="Masses")
    final_type_labels: dict[int, str] = dict(type_labels)
    masses_by_label: dict[str, float] = {}

    for line in masses_block:
        stripped_line: str = strip_inline_comment(line)
        if not stripped_line:
            continue
        fields: list[str] = stripped_line.split()
        if len(fields) < 2:
            continue

        type_id: int = int(fields[0])
        mass_value: float = float(fields[1])
        symbol: str = final_type_labels.get(type_id, extract_inline_comment(line))
        symbol = symbol.strip()

        if symbol not in atomic_numbers:
            raise ValueError(
                f"Could not infer a valid chemical symbol for type {type_id}. "
                "Please ensure 'Atom Type Labels' or mass comments are present."
            )

        final_type_labels[type_id] = symbol
        masses_by_label[symbol] = mass_value

    return masses_by_label, final_type_labels


def parse_atoms_section(lines: list[str]) -> list[AtomRecord]:
    """Parse the ``Atoms`` section for ``atom_style atomic`` data.

    Args:
        lines: Full file contents split into lines.

    Returns:
        Parsed atom records from the ``Atoms`` section.

    Raises:
        ValueError: If the ``Atoms`` section is missing or uses a different style.
    """
    header_index: int | None = find_section_index(lines=lines, section_name="Atoms")
    if header_index is None:
        raise ValueError("Section 'Atoms' was not found in the LAMMPS file.")

    atoms_header: str = lines[header_index].strip()
    if "#" in atoms_header:
        style_hint: str = atoms_header.split("#", 1)[1].strip().lower()
        if style_hint and style_hint != "atomic":
            raise ValueError("Only 'Atoms  # atomic' files are supported by this stitcher.")

    atoms_block: list[str] = read_section_block(lines=lines, section_name="Atoms")
    atom_records: list[AtomRecord] = []

    for line in atoms_block:
        stripped_line: str = strip_inline_comment(line)
        if not stripped_line:
            continue
        fields: list[str] = stripped_line.split()
        if len(fields) < 5:
            continue
        atom_records.append(
            AtomRecord(
                atom_id=int(fields[0]),
                atom_type=int(fields[1]),
                x=float(fields[2]),
                y=float(fields[3]),
                z=float(fields[4]),
            )
        )

    return atom_records


def parse_lammps_atomic_data(input_path: Path) -> LammpsAtomicData:
    """Parse one orthorhombic ``atom_style atomic`` LAMMPS data file.

    Args:
        input_path: Input LAMMPS data file path.

    Returns:
        Parsed LAMMPS data.
    """
    lines: list[str] = input_path.read_text(encoding="utf-8").splitlines()
    type_labels: dict[int, str] = parse_atom_type_labels(lines=lines)
    masses_by_label, final_type_labels = parse_masses(lines=lines, type_labels=type_labels)

    return LammpsAtomicData(
        comment_line=lines[0] if lines else f"# {input_path.name}",
        bounds=parse_box_bounds(lines=lines),
        type_labels=final_type_labels,
        masses_by_label=masses_by_label,
        atoms=parse_atoms_section(lines=lines),
    )


def wrap_value(value: float, lo: float, hi: float) -> float:
    """Wrap a coordinate back into a periodic interval.

    Args:
        value: Coordinate value to wrap.
        lo: Lower bound of the periodic interval.
        hi: Upper bound of the periodic interval.

    Returns:
        Wrapped coordinate value.
    """
    length: float = hi - lo
    if length <= 0.0:
        return value
    return ((value - lo) % length) + lo


def build_ase_atoms(data: LammpsAtomicData) -> Atoms:
    """Convert parsed LAMMPS atomic data into an ``ase.Atoms`` object.

    Args:
        data: Parsed LAMMPS data.

    Returns:
        An ``ase.Atoms`` object with wrapped Cartesian coordinates.
    """
    symbols: list[str] = []
    positions: list[list[float]] = []

    for atom in data.atoms:
        symbol: str = data.type_labels[atom.atom_type]
        symbols.append(symbol)
        positions.append(
            [
                wrap_value(atom.x, data.bounds.xlo, data.bounds.xhi) - data.bounds.xlo,
                wrap_value(atom.y, data.bounds.ylo, data.bounds.yhi) - data.bounds.ylo,
                wrap_value(atom.z, data.bounds.zlo, data.bounds.zhi) - data.bounds.zlo,
            ]
        )

    cell_matrix: np.ndarray = np.diag(
        [
            data.bounds.lx(),
            data.bounds.ly(),
            data.bounds.lz(),
        ]
    ).astype(float)

    atoms: Atoms = Atoms(symbols=symbols, positions=np.array(positions, dtype=float), cell=cell_matrix, pbc=True)
    atoms.wrap()
    return atoms


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


def merge_species_metadata(
    data1: LammpsAtomicData,
    data2: LammpsAtomicData,
    tolerance: float = 1.0e-6,
) -> tuple[list[str], dict[str, float]]:
    """Merge species labels and masses from two inputs.

    Args:
        data1: First parsed LAMMPS input.
        data2: Second parsed LAMMPS input.
        tolerance: Allowed absolute mass mismatch for repeated species labels.

    Returns:
        A tuple containing:
            1. Ordered merged species labels.
            2. Mapping from species label to mass.

    Raises:
        ValueError: If the same species label appears with conflicting masses.
    """
    ordered_labels: list[str] = []
    merged_masses: dict[str, float] = {}

    for source_data in [data1, data2]:
        source_order: list[str] = [
            source_data.type_labels[type_id]
            for type_id in sorted(source_data.type_labels)
        ]
        for label in source_order:
            mass_value: float = source_data.masses_by_label[label]
            if label in merged_masses:
                if abs(merged_masses[label] - mass_value) > tolerance:
                    raise ValueError(
                        f"Mass mismatch for species '{label}': {merged_masses[label]} vs {mass_value}."
                    )
            else:
                ordered_labels.append(label)
                merged_masses[label] = mass_value

    return ordered_labels, merged_masses


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
    data1: LammpsAtomicData,
    data2: LammpsAtomicData,
    gap: float,
    requested_species_order: list[str],
) -> tuple[Atoms, list[str], dict[str, float], XYCompatibilityReport]:
    """Stitch two LAMMPS structures together along the z axis.

    Args:
        data1: First parsed LAMMPS input.
        data2: Second parsed LAMMPS input.
        gap: Gap between the two slabs in Angstrom.
        requested_species_order: User-requested output species order.

    Returns:
        A tuple containing:
            1. The stitched ``ase.Atoms`` object.
            2. Ordered merged species labels.
            3. Mapping from species label to mass.
            4. Validated x/y compatibility report.
    """
    validated_gap: float = validate_gap(gap=gap)

    atoms1: Atoms = build_ase_atoms(data=data1)
    atoms2: Atoms = build_ase_atoms(data=data2)
    xy_report: XYCompatibilityReport = validate_xy_compatibility(atoms1=atoms1, atoms2=atoms2)

    atoms1.wrap()
    atoms2.wrap()

    shift_z: float = float(atoms1.cell[2, 2]) + validated_gap
    atoms2.translate([0.0, 0.0, shift_z])

    combined_atoms: Atoms = atoms1 + atoms2
    combined_cell: np.ndarray = atoms1.cell.array.copy()
    combined_cell[2, 2] = float(atoms1.cell[2, 2]) + float(atoms2.cell[2, 2]) + validated_gap
    combined_atoms.set_cell(combined_cell, scale_atoms=False)
    combined_atoms.wrap()

    available_species_order, masses_by_label = merge_species_metadata(data1=data1, data2=data2)
    species_order: list[str] = validate_requested_species_order(
        requested_order=requested_species_order,
        available_species=available_species_order,
    )
    combined_atoms = reorder_atoms_by_species(combined_atoms, species_order)
    return combined_atoms, species_order, masses_by_label, xy_report


def format_float(value: float) -> str:
    """Format a floating-point value for LAMMPS output.

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


def write_lammps_conf_lmp(
    atoms: Atoms,
    output_path: Path,
    species_order: list[str],
    masses_by_label: dict[str, float],
    comment_line: str,
) -> None:
    """Write an OVITO-friendly ``atom_style atomic`` LAMMPS data file.

    Args:
        atoms: Structure to write.
        output_path: Destination path for the combined ``conf.lmp``.
        species_order: Desired species order in the output file.
        masses_by_label: Mapping from species label to mass.
        comment_line: Comment line written at the top of the file.
    """
    ordered_atoms: Atoms = reorder_atoms_by_species(atoms, species_order)
    ordered_atoms = ordered_atoms.copy()
    ordered_atoms.wrap()

    type_id_by_label: dict[str, int] = {
        label: index + 1 for index, label in enumerate(species_order)
    }
    bounds: BoxBounds = BoxBounds(
        xlo=0.0,
        xhi=float(ordered_atoms.cell[0, 0]),
        ylo=0.0,
        yhi=float(ordered_atoms.cell[1, 1]),
        zlo=0.0,
        zhi=float(ordered_atoms.cell[2, 2]),
    )

    positions: np.ndarray = np.array(ordered_atoms.get_positions(), dtype=float)
    symbols: list[str] = ordered_atoms.get_chemical_symbols()

    lines: list[str] = [
        comment_line,
        "",
        f"{len(ordered_atoms)} atoms",
        f"{len(species_order)} atom types",
        "",
        f"{format_float(bounds.xlo)} {format_float(bounds.xhi)} xlo xhi",
        f"{format_float(bounds.ylo)} {format_float(bounds.yhi)} ylo yhi",
        f"{format_float(bounds.zlo)} {format_float(bounds.zhi)} zlo zhi",
        "",
        "Atom Type Labels",
        "",
    ]

    for label in species_order:
        lines.append(f"{type_id_by_label[label]} {label}")

    lines.extend(["", "Masses", ""])

    for label in species_order:
        lines.append(
            f"{type_id_by_label[label]} {format_float(masses_by_label[label])}  # {label}"
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
                    str(type_id_by_label[symbol]),
                    format_float(position_vector[0]),
                    format_float(position_vector[1]),
                    format_float(position_vector[2]),
                ]
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_log_file(
    output_dir: Path,
    input1: Path,
    input2: Path,
    output_file: str,
    gap: float,
    species_order: list[str],
    xy_report: XYCompatibilityReport,
) -> None:
    """Write a small execution log beside the output ``conf.lmp``.

    Args:
        output_dir: Output directory.
        input1: First input file path.
        input2: Second input file path.
        output_file: Output LAMMPS filename.
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
    (output_dir / "log.join_conf_lmps").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the z-axis ``conf.lmp`` stitcher."""
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

    data1: LammpsAtomicData = parse_lammps_atomic_data(input_path=input1)
    data2: LammpsAtomicData = parse_lammps_atomic_data(input_path=input2)

    combined_atoms, species_order, masses_by_label, xy_report = stitch_atoms_along_z(
        data1=data1,
        data2=data2,
        gap=args.gap,
        requested_species_order=requested_species_order,
    )

    print(format_xy_compatibility_report(xy_report=xy_report))

    write_lammps_conf_lmp(
        atoms=combined_atoms,
        output_path=output_path,
        species_order=species_order,
        masses_by_label=masses_by_label,
        comment_line=(
            f"# LAMMPS data file stitched from {input1.name} and {input2.name} "
            "by join_conf_lmps.py"
        ),
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
