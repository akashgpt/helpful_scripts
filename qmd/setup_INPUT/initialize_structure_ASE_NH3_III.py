from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write

REFERENCE_LAMMPS_MASSES: dict[str, float] = {
    "H": 1.00794,
    "N": 14.0067,
}


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly random 3D rotation from a quaternion.

    Args:
        rng: Random-number generator used to build a reproducible rotation.

    Returns:
        A ``3 x 3`` rotation matrix.
    """
    u1, u2, u3 = rng.random(3)

    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    R = np.array([
        [1 - 2*(q3*q3 + q4*q4),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        [    2*(q2*q3 + q1*q4), 1 - 2*(q2*q2 + q4*q4),     2*(q3*q4 - q1*q2)],
        [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2*q2 + q3*q3)],
    ])
    return R


def ammonia_geometry(
    bond_length: float = 1.02,
    angle_deg: float = 106.7,
) -> np.ndarray:
    """Return an idealized NH3 geometry centered on nitrogen.

    Args:
        bond_length: N-H bond length in Angstrom.
        angle_deg: H-N-H bond angle in degrees.

    Returns:
        A ``3 x 3`` array containing the hydrogen positions relative to the
        nitrogen atom at the origin.
    """
    theta = np.deg2rad(angle_deg)

    # For 3 equivalent H atoms at azimuths 0,120,240 and common polar angle alpha:
    # cos(theta_HNH) = 1.5 cos^2(alpha) - 0.5
    cos_alpha_sq = (np.cos(theta) + 0.5) / 1.5
    cos_alpha_sq = np.clip(cos_alpha_sq, 0.0, 1.0)
    cos_alpha = np.sqrt(cos_alpha_sq)
    alpha = np.arccos(cos_alpha)
    sin_alpha = np.sin(alpha)

    H: list[np.ndarray] = []
    for phi in [0.0, 2*np.pi/3, 4*np.pi/3]:
        vec = bond_length * np.array([
            sin_alpha * np.cos(phi),
            sin_alpha * np.sin(phi),
            cos_alpha,
        ])
        H.append(vec)

    return np.array(H)


def reorder_atoms_by_species(atoms: Atoms, species_order: list[str]) -> Atoms:
    """Return a new Atoms object with atoms grouped by the requested species order.

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


def normalize_poscar_header(
    poscar_path: str,
    comment_line: str,
    species_order: list[str],
) -> None:
    """Rewrite the POSCAR comment and species lines to a clean grouped order.

    Args:
        poscar_path: Path to the POSCAR file to normalize.
        comment_line: Free-form POSCAR title/comment line.
        species_order: Chemical symbols in the desired output order.
    """
    with open(poscar_path, "r", encoding="utf-8") as file:
        lines: list[str] = file.read().splitlines()

    lines[0] = comment_line
    lines[5] = " ".join(species_order)

    with open(poscar_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def infer_species_in_output_order(atoms: Atoms, species_order: list[str]) -> list[str]:
    """Infer the final species order used in written output files.

    Args:
        atoms: Structure whose species must be written.
        species_order: Preferred species ordering.

    Returns:
        Species present in the structure, ordered first by ``species_order``
        and then by first appearance for any remaining species.
    """
    atom_symbols: list[str] = atoms.get_chemical_symbols()
    unique_symbols_by_appearance: list[str] = []

    for symbol in atom_symbols:
        if symbol not in unique_symbols_by_appearance:
            unique_symbols_by_appearance.append(symbol)

    ordered_symbols: list[str] = [
        symbol for symbol in species_order if symbol in unique_symbols_by_appearance
    ]
    ordered_symbols.extend(
        [
            symbol
            for symbol in unique_symbols_by_appearance
            if symbol not in ordered_symbols
        ]
    )
    return ordered_symbols


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
    comment_line: str = "# LAMMPS data file written by initialize_structure_ASE_NH3_III.py",
) -> None:
    """Write a LAMMPS ``atom_style atomic`` data file in the local MLMD format.

    Args:
        atoms: Structure to write.
        output_path: Destination path for the ``conf.lmp`` file.
        species_order: Preferred output ordering for chemical species.
        comment_line: Comment line written at the top of the file.
    """
    output_file: Path = Path(output_path)
    atoms_to_write: Atoms = reorder_atoms_by_species(atoms, species_order)
    atoms_to_write = atoms_to_write.copy()
    atoms_to_write.wrap()

    ordered_species: list[str] = infer_species_in_output_order(
        atoms=atoms_to_write,
        species_order=species_order,
    )
    type_id_by_symbol: dict[str, int] = {
        symbol: index + 1 for index, symbol in enumerate(ordered_species)
    }
    lx, ly, lz = validate_orthorhombic_cell(atoms=atoms_to_write)

    positions: np.ndarray = np.array(atoms_to_write.get_positions(), dtype=float)
    symbols: list[str] = atoms_to_write.get_chemical_symbols()

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


def build_nh3_phase_iii_supercell(
    a: float = 4.244,
    reps: tuple[int, int, int] = (2, 2, 2),
    bond_length: float = 1.02,
    angle_deg: float = 106.7,
    seed: int = 42,
) -> Atoms:
    """Build a representative NH3-III supercell.

    Args:
        a: Conventional cubic lattice parameter in Angstrom.
        reps: Number of conventional FCC cells along x, y, and z.
        bond_length: N-H bond length in Angstrom.
        angle_deg: H-N-H bond angle in degrees.
        seed: Random seed for molecular orientations.

    Returns:
        The generated NH3-III structure.
    """
    rng: np.random.Generator = np.random.default_rng(seed)

    # Conventional FCC basis for the cubic Fm-3m cell
    fcc_basis_frac: np.ndarray = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])

    nx, ny, nz = reps
    cell: np.ndarray = np.diag([a * nx, a * ny, a * nz])

    local__H_vectors: np.ndarray = ammonia_geometry(
        bond_length=bond_length,
        angle_deg=angle_deg,
    )

    symbols: list[str] = []
    positions: list[np.ndarray] = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_shift = np.array([i, j, k], dtype=float)

                for basis in fcc_basis_frac:
                    frac = (cell_shift + basis) / np.array([nx, ny, nz], dtype=float)
                    N_pos = frac @ cell

                    symbols.append("N")
                    positions.append(N_pos)

                    R: np.ndarray = random_rotation_matrix(rng)
                    for hvec in local__H_vectors:
                        H_pos: np.ndarray = N_pos + R @ hvec
                        symbols.append("H")
                        positions.append(H_pos)

    atoms: Atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.wrap()

    return atoms


if __name__ == "__main__":
    # Generate the configured NH3-III supercell and write grouped species outputs.
    atoms = build_nh3_phase_iii_supercell(
        a=4.244,
        reps=(4, 4, 4),
        # reps=(4, 4, 4),
        bond_length=1.02,
        angle_deg=106.7,
        seed=7,
    )
    atoms = reorder_atoms_by_species(atoms, ["H", "N"])

    print(atoms)
    print("Number of atoms:", len(atoms))

    # For VESTA / visualization
    write(f"NH3_III_{len(atoms)}atoms.cif", atoms)

    # For VASP
    poscar_path: str = f"POSCAR_NH3_III_{len(atoms)}atoms"
    write(poscar_path, atoms, direct=True, vasp5=True)
    normalize_poscar_header(poscar_path, "NH3_III", ["H", "N"])

    # For LAMMPS
    lammps_path: str = f"conf.lmp_NH3_III_{len(atoms)}atoms"
    write_lammps_conf_lmp(atoms, lammps_path, ["H", "N"])
    print(f"Written {lammps_path}")
