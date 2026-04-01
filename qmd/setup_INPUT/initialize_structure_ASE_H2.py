#!/usr/bin/env python3
"""Generate molecular H2 structures and write POSCAR, CIF, and LAMMPS files.

This script mirrors the workflow used in ``initialize_structure_ASE_NH3_III.py``
but replaces NH3 molecules with H2 molecules. Each H2 molecule is assigned a
random orientation, while the molecular centers are distributed across an
orthorhombic simulation box whose ``x``, ``y``, and ``z`` lengths are chosen by
the user. The requested number of H2 molecules is enforced exactly, and the
script requires that this molecule count be even.

Usage:
    python initialize_structure_ASE_H2.py \
        --num-molecules 40 \
        --cell-lengths 12.0 12.0 18.0 \
        --bond-length 0.74 \
        --seed 7 \
        --output-prefix H2_molecular \
        --output-dir ./output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write


REFERENCE_LAMMPS_MASSES: dict[str, float] = {
    "H": 1.00794,
}


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
            [1.0 - 2.0 * (q3 * q3 + q4 * q4), 2.0 * (q2 * q3 - q1 * q4), 2.0 * (q2 * q4 + q1 * q3)],
            [2.0 * (q2 * q3 + q1 * q4), 1.0 - 2.0 * (q2 * q2 + q4 * q4), 2.0 * (q3 * q4 - q1 * q2)],
            [2.0 * (q2 * q4 - q1 * q3), 2.0 * (q3 * q4 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3)],
        ],
        dtype=float,
    )
    return rotation_matrix


def h2_geometry(bond_length: float = 0.74) -> np.ndarray:
    """Return an H2 geometry centered at the molecular center of mass.

    Args:
        bond_length: Hydrogen-hydrogen bond length in Angstrom.

    Returns:
        A ``2 x 3`` array containing the two hydrogen positions relative to the
        molecular center.
    """
    half_bond_length = 0.5 * bond_length
    hydrogen_offsets = np.array(
        [
            [-half_bond_length, 0.0, 0.0],
            [half_bond_length, 0.0, 0.0],
        ],
        dtype=float,
    )
    return hydrogen_offsets


def validate_num_molecules(num_molecules: int) -> int:
    """Validate the requested H2 molecule count.

    Args:
        num_molecules: Requested number of H2 molecules.

    Returns:
        The validated number of H2 molecules.

    Raises:
        ValueError: If the requested number of molecules is not positive and even.
    """
    if num_molecules <= 0:
        raise ValueError("The number of H2 molecules must be positive.")
    if num_molecules % 2 != 0:
        raise ValueError("The number of H2 molecules must be even.")
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
    if any(length <= 0.0 for length in cell_lengths):
        raise ValueError("All cell lengths must be positive.")
    return cell_lengths


def validate_bond_length(bond_length: float) -> float:
    """Validate the requested H-H bond length.

    Args:
        bond_length: Requested H-H bond length in Angstrom.

    Returns:
        The validated H-H bond length.

    Raises:
        ValueError: If the bond length is not positive.
    """
    if bond_length <= 0.0:
        raise ValueError("The H-H bond length must be positive.")
    return bond_length


def infer_grid_shape(
    num_molecules: int,
    cell_lengths: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Infer a regular placement grid that can host the requested molecules.

    Args:
        num_molecules: Number of H2 molecular centers that must be placed.
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
    """Generate molecular-center positions inside the orthorhombic box.

    Args:
        num_molecules: Number of H2 molecules to place.
        cell_lengths: Box lengths along the ``x``, ``y``, and ``z`` axes.
        rng: Random-number generator used to shuffle the candidate sites.

    Returns:
        A tuple containing:
            1. A ``num_molecules x 3`` array of molecular-center positions.
            2. The underlying regular grid shape used to generate those centers.
    """
    grid_shape = infer_grid_shape(num_molecules=num_molecules, cell_lengths=cell_lengths)
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


def build_h2_molecular_box(
    num_molecules: int = 32,
    cell_lengths: tuple[float, float, float] = (10.0, 10.0, 10.0),
    bond_length: float = 0.74,
    seed: int = 7,
) -> tuple[Atoms, tuple[int, int, int]]:
    """Build a representative molecular-H2 structure in an orthorhombic box.

    Args:
        num_molecules: Number of H2 molecules to place. Must be even.
        cell_lengths: Box lengths along the ``x``, ``y``, and ``z`` axes.
        bond_length: Hydrogen-hydrogen bond length in Angstrom.
        seed: Seed used to make the random molecular orientations reproducible.

    Returns:
        A tuple containing:
            1. An ASE ``Atoms`` object containing the molecular H2 structure.
            2. The regular center-placement grid shape used internally.
    """
    validated_num_molecules = validate_num_molecules(num_molecules=num_molecules)
    validated_cell_lengths = validate_cell_lengths(cell_lengths=cell_lengths)
    validated_bond_length = validate_bond_length(bond_length=bond_length)
    rng = np.random.default_rng(seed)
    cell = np.diag(validated_cell_lengths).astype(float)

    local_hydrogen_offsets = h2_geometry(bond_length=validated_bond_length)
    molecular_centers, grid_shape = generate_molecule_centers(
        num_molecules=validated_num_molecules,
        cell_lengths=validated_cell_lengths,
        rng=rng,
    )

    symbols: list[str] = []
    positions: list[np.ndarray] = []

    for molecular_center in molecular_centers:
        rotation_matrix = random_rotation_matrix(rng)
        for hydrogen_offset in local_hydrogen_offsets:
            rotated_offset = rotation_matrix @ hydrogen_offset
            hydrogen_position = molecular_center + rotated_offset
            symbols.append("H")
            positions.append(hydrogen_position)

    atoms = Atoms(symbols=symbols, positions=np.array(positions, dtype=float), cell=cell, pbc=True)
    atoms.wrap()
    return atoms, grid_shape


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
    cell_matrix: np.ndarray = np.array(atoms.cell.array, dtype=float)
    off_diagonal_terms: np.ndarray = cell_matrix.copy()
    np.fill_diagonal(off_diagonal_terms, 0.0)

    if not np.allclose(off_diagonal_terms, 0.0, atol=tolerance):
        raise ValueError(
            "The conf.lmp writer currently expects an orthorhombic cell."
        )

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
    comment_line: str = "# LAMMPS data file written by initialize_structure_ASE_H2.py",
) -> None:
    """Write a LAMMPS ``atom_style atomic`` data file in the local MLMD format.

    Args:
        atoms: Structure to write.
        output_path: Destination path for the ``conf.lmp`` file.
        species_order: Preferred output ordering for chemical species.
        comment_line: Comment line written at the top of the file.
    """
    output_file: Path = Path(output_path)

    # Determine ordered species present in the structure
    atom_symbols: list[str] = atoms.get_chemical_symbols()
    unique_by_appearance: list[str] = []
    for symbol in atom_symbols:
        if symbol not in unique_by_appearance:
            unique_by_appearance.append(symbol)
    ordered_species: list[str] = [s for s in species_order if s in unique_by_appearance]
    ordered_species += [s for s in unique_by_appearance if s not in ordered_species]

    type_id_by_symbol: dict[str, int] = {
        symbol: index + 1 for index, symbol in enumerate(ordered_species)
    }
    lx, ly, lz = validate_orthorhombic_cell(atoms=atoms)
    positions: np.ndarray = np.array(atoms.get_positions(), dtype=float)

    lines: list[str] = [
        comment_line,
        "",
        f"{len(atoms)} atoms",
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
        zip(atom_symbols, positions),
        start=1,
    ):
        lines.append(
            " ".join([
                str(atom_index),
                str(type_id_by_symbol[symbol]),
                format_lammps_float(position_vector[0]),
                format_lammps_float(position_vector[1]),
                format_lammps_float(position_vector[2]),
            ])
        )

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_structure_outputs(
    atoms: Atoms,
    output_prefix: str,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write CIF, POSCAR, and LAMMPS data files for the generated structure.

    Args:
        atoms: The structure to be written.
        output_prefix: Prefix used for all output filenames.
        output_dir: Directory where the files will be written.

    Returns:
        Paths to the generated CIF file, POSCAR file, and LAMMPS conf.lmp file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_path = output_dir / f"{output_prefix}.cif"
    poscar_path = output_dir / f"POSCAR_{output_prefix}"
    lammps_path = output_dir / f"conf.lmp_{output_prefix}"

    write(cif_path.as_posix(), atoms)
    write(poscar_path.as_posix(), atoms, direct=True, vasp5=True)
    write_lammps_conf_lmp(atoms, lammps_path, species_order=["H"])

    return cif_path, poscar_path, lammps_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the H2 structure generator.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a representative molecular H2 structure with a user-chosen "
            "number of molecules in an orthorhombic box and write CIF and POSCAR outputs."
        )
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=32,
        help="Number of H2 molecules to generate. Must be positive and even.",
    )
    parser.add_argument(
        "--cell-lengths",
        type=float,
        nargs=3,
        metavar=("LX", "LY", "LZ"),
        default=(10.0, 10.0, 10.0),
        help="Orthorhombic cell lengths along x, y, and z in Angstrom.",
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=0.74,
        help="H-H bond length in Angstrom. Default: %(default)s.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for molecular orientations. Default: %(default)s.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="H2_molecular",
        help="Prefix used for the output filenames. Default: %(default)s.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the generated files will be written. Default: current directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the requested H2 structure and write its output files."""
    args = parse_args()
    cell_lengths = (
        float(args.cell_lengths[0]),
        float(args.cell_lengths[1]),
        float(args.cell_lengths[2]),
    )

    try:
        atoms, grid_shape = build_h2_molecular_box(
            num_molecules=args.num_molecules,
            cell_lengths=cell_lengths,
            bond_length=args.bond_length,
            seed=args.seed,
        )
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    molecules_count = len(atoms) // 2
    print(atoms)
    print(f"Number of H2 molecules: {molecules_count}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Placement grid used for molecular centers: {grid_shape}")
    print(f"Cell lengths (Angstrom): {atoms.cell.lengths()}")

    auto_prefix = f"{args.output_prefix}_{molecules_count}molecules_{len(atoms)}atoms"
    cif_path, poscar_path, lammps_path = write_structure_outputs(
        atoms=atoms,
        output_prefix=auto_prefix,
        output_dir=args.output_dir,
    )

    print(f"Wrote CIF:    {cif_path}")
    print(f"Wrote POSCAR: {poscar_path}")
    print(f"Wrote LAMMPS: {lammps_path}")


if __name__ == "__main__":
    main()
