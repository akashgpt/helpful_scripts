"""Generate MgSiO3 bridgmanite (orthorhombic perovskite) supercell structures.

Bridgmanite is the dominant mineral in Earth's lower mantle, crystallizing in
the orthorhombic perovskite structure (space group Pbnm/Pnma, #62). This
script builds a conventional 20-atom unit cell from experimental Wyckoff
positions and tiles it into a supercell, then writes output in CIF, POSCAR,
and LAMMPS formats.

Crystallographic data (ambient conditions) from Horiuchi et al. (1987),
reported in the Pbnm setting:
    a = 4.7754 A, b = 4.9292 A, c = 6.8969 A
    Mg  4c  (0.5141, 0.5560, 0.25)
    Si  4b  (0.0,    0.5,    0.0 )
    O1  4c  (0.1014, 0.4660, 0.25)
    O2  8d  (0.1961, 0.2014, 0.5532)

ASE uses the equivalent Pnma setting for space group 62. This script therefore
converts the literature cell and fractional coordinates from Pbnm into ASE's
Pnma convention before calling ``ase.spacegroup.crystal``.

Usage examples
--------------
1. Run directly to generate a default supercell (edit reps/lattice params in __main__):

    $ conda run -n ase_env python initialize_structure_ASE_MgSiO3_bridgmanite.py
    $ conda run -n ase_env python $HELP_SCRIPTS_qmd/setup_INPUT/initialize_structure_ASE_MgSiO3_bridgmanite.py

   This writes three files in the current directory:
     - MgSiO3_bridgmanite_<N>atoms.cif   (for VESTA / visualization)
     - POSCAR_MgSiO3_bridgmanite_<N>atoms (for VASP)
     - conf.lmp_MgSiO3_bridgmanite_<N>atoms (for LAMMPS)

2. Import and build a custom supercell in your own script:

    >>> from initialize_structure_ASE_MgSiO3_bridgmanite import (
    ...     build_bridgmanite_supercell, reorder_atoms_by_species,
    ... )
    >>> atoms = build_bridgmanite_supercell(reps=(3, 3, 2))
    >>> atoms = reorder_atoms_by_species(atoms, ["Mg", "Si", "O"])
    >>> print(len(atoms))   # 20 * 3 * 3 * 2 = 360
    360

3. Build just the primitive unit cell (20 atoms):

    >>> from initialize_structure_ASE_MgSiO3_bridgmanite import build_bridgmanite_unit_cell
    >>> unit = build_bridgmanite_unit_cell()
    >>> print(len(unit))    # 4 Mg + 4 Si + 12 O = 20
    20

4. Use compressed lattice parameters (e.g. high-pressure conditions):

    >>> atoms = build_bridgmanite_supercell(a=4.65, b=4.82, c=6.72, reps=(4, 4, 3))
    >>> print(f"Cell volume: {atoms.get_volume():.1f} A^3")

5. Write only a LAMMPS data file from a custom structure:

    >>> from initialize_structure_ASE_MgSiO3_bridgmanite import (
    ...     build_bridgmanite_supercell, write_lammps_conf_lmp,
    ... )
    >>> atoms = build_bridgmanite_supercell(reps=(2, 2, 2))
    >>> write_lammps_conf_lmp(atoms, "conf.lmp", ["Mg", "Si", "O"])
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import write
from ase.spacegroup import crystal


# ---------------------------------------------------------------------------
# Reference masses for LAMMPS output (matching MLMD conventions)
# ---------------------------------------------------------------------------
REFERENCE_LAMMPS_MASSES: dict[str, float] = {
    "Mg": 24.305,
    "Si": 28.0855,
    "O": 15.9994,
}

OUTPUT_DECIMAL_PLACES: int = 6

HORIUCHI_PBNM_LATTICE_ANGSTROM: tuple[float, float, float] = (
    4.7754,
    4.9292,
    6.8969,
)

HORIUCHI_PBNM_BASIS: dict[str, tuple[float, float, float]] = {
    "Mg": (0.5141, 0.5560, 0.25),
    "Si": (0.0, 0.5, 0.0),
    "O1": (0.1014, 0.4660, 0.25),
    "O2": (0.1961, 0.2014, 0.5532),
}


# ===================================================================
# Structure builder
# ===================================================================

def convert_pbnm_to_pnma_cellpar(
    a: float,
    b: float,
    c: float,
) -> tuple[float, float, float]:
    """Convert orthorhombic lattice parameters from Pbnm to ASE's Pnma setting.

    Horiuchi et al. report bridgmanite in the Pbnm setting. ASE uses the
    symmetry-equivalent Pnma setting for space group 62. For the literature
    convention used here, the equivalent setting is obtained by swapping the
    crystallographic ``b`` and ``c`` axes.

    Args:
        a: Pbnm lattice parameter ``a`` in Angstrom.
        b: Pbnm lattice parameter ``b`` in Angstrom.
        c: Pbnm lattice parameter ``c`` in Angstrom.

    Returns:
        The equivalent ``(a, b, c)`` tuple in ASE's Pnma setting.
    """
    return (a, c, b)


def convert_pbnm_to_pnma_fractional_position(
    fractional_position: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert a fractional coordinate from the Pbnm to the Pnma setting.

    Args:
        fractional_position: Fractional ``(x, y, z)`` coordinate in the Pbnm
            setting used by the cited literature.

    Returns:
        The equivalent fractional ``(x, y, z)`` coordinate in ASE's Pnma
        setting.
    """
    x_coordinate, y_coordinate, z_coordinate = fractional_position
    return (x_coordinate, z_coordinate, y_coordinate)


def get_horiuchi_pnma_basis() -> list[tuple[float, float, float]]:
    """Return the Horiuchi basis transformed into ASE's Pnma setting.

    Returns:
        The symmetry-inequivalent Mg, Si, O1, and O2 fractional coordinates in
        the Pnma setting expected by ASE.
    """
    return [
        convert_pbnm_to_pnma_fractional_position(HORIUCHI_PBNM_BASIS["Mg"]),
        convert_pbnm_to_pnma_fractional_position(HORIUCHI_PBNM_BASIS["Si"]),
        convert_pbnm_to_pnma_fractional_position(HORIUCHI_PBNM_BASIS["O1"]),
        convert_pbnm_to_pnma_fractional_position(HORIUCHI_PBNM_BASIS["O2"]),
    ]


def convert_pnma_atoms_to_pbnm_atoms(pnma_atoms: Atoms) -> Atoms:
    """Convert an ASE-generated Pnma bridgmanite cell back to Pbnm axis order.

    The fractional-coordinate transform is the same ``y <-> z`` swap used above,
    because that operation is its own inverse. Returning atoms in the original
    Pbnm axis order keeps the exposed ``a, b, c`` inputs and supercell
    repetitions aligned with the literature convention.

    Args:
        pnma_atoms: Bridgmanite structure expressed in ASE's Pnma setting.

    Returns:
        A symmetry-equivalent structure in the original Pbnm axis order.
    """
    pnma_scaled_positions: np.ndarray = np.array(
        pnma_atoms.get_scaled_positions(),
        dtype=float,
    )
    pbnm_scaled_positions: list[tuple[float, float, float]] = [
        convert_pbnm_to_pnma_fractional_position(tuple(position))
        for position in pnma_scaled_positions
    ]
    pnma_a, pnma_b, pnma_c = pnma_atoms.cell.lengths()
    pbnm_atoms: Atoms = Atoms(
        symbols=pnma_atoms.get_chemical_symbols(),
        cell=[pnma_a, pnma_c, pnma_b],
        scaled_positions=pbnm_scaled_positions,
        pbc=True,
    )
    pbnm_atoms.wrap()
    return pbnm_atoms


def validate_supercell_repetitions(reps: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate supercell repetition counts.

    Args:
        reps: Requested repetition counts along the three cell vectors.

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


def build_bridgmanite_unit_cell(
    a: float = HORIUCHI_PBNM_LATTICE_ANGSTROM[0],
    b: float = HORIUCHI_PBNM_LATTICE_ANGSTROM[1],
    c: float = HORIUCHI_PBNM_LATTICE_ANGSTROM[2],
) -> Atoms:
    """Build one conventional 20-atom unit cell of MgSiO3 bridgmanite.

    The reference crystallographic data are taken from Horiuchi et al. in the
    Pbnm setting. Those values are converted internally into the equivalent
    Pnma setting expected by ASE, which then generates the conventional 20-atom
    bridgmanite unit cell (4 Mg + 4 Si + 12 O).

    Args:
        a: Pbnm lattice parameter *a* in Angstrom.
        b: Pbnm lattice parameter *b* in Angstrom.
        c: Pbnm lattice parameter *c* in Angstrom.

    Returns:
        An ASE Atoms object for the bridgmanite unit cell in the original Pbnm
        axis order used by the input lattice parameters.
    """
    pnma_a, pnma_b, pnma_c = convert_pbnm_to_pnma_cellpar(a=a, b=b, c=c)
    pnma_atoms: Atoms = crystal(
        symbols=["Mg", "Si", "O", "O"],
        basis=get_horiuchi_pnma_basis(),
        spacegroup=62,  # ASE setting 1: Pnma, equivalent to literature Pbnm
        cellpar=[pnma_a, pnma_b, pnma_c, 90, 90, 90],
        primitive_cell=False,
    )
    atoms: Atoms = convert_pnma_atoms_to_pbnm_atoms(pnma_atoms)
    return atoms


def build_bridgmanite_supercell(
    a: float = HORIUCHI_PBNM_LATTICE_ANGSTROM[0],
    b: float = HORIUCHI_PBNM_LATTICE_ANGSTROM[1],
    c: float = HORIUCHI_PBNM_LATTICE_ANGSTROM[2],
    reps: tuple[int, int, int] = (2, 2, 2),
) -> Atoms:
    """Build a bridgmanite supercell by tiling the conventional unit cell.

    Args:
        a: Pbnm lattice parameter *a* in Angstrom.
        b: Pbnm lattice parameter *b* in Angstrom.
        c: Pbnm lattice parameter *c* in Angstrom.
        reps: Number of unit-cell repetitions along the literature Pbnm ``(a, b, c)`` directions.

    Returns:
        The tiled supercell as an ASE Atoms object.
    """
    reps = validate_supercell_repetitions(reps)
    unit_cell: Atoms = build_bridgmanite_unit_cell(a=a, b=b, c=c)
    supercell: Atoms = unit_cell.repeat(reps)
    supercell.wrap()
    return supercell


# ===================================================================
# Atom reordering (group by species for VASP / LAMMPS)
# ===================================================================

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

    # Append any species not listed in species_order (safety fallback)
    remaining_indices: list[int] = [
        index for index, symbol in enumerate(symbols) if symbol not in species_order
    ]
    ordered_indices.extend(remaining_indices)

    return atoms[ordered_indices]


# ===================================================================
# POSCAR normalisation
# ===================================================================

def format_fixed_precision_float(
    value: float,
    decimal_places: int = OUTPUT_DECIMAL_PLACES,
) -> str:
    """Format a floating-point value with a fixed number of decimals.

    Args:
        value: Floating-point value to serialize.
        decimal_places: Number of digits to keep after the decimal point.

    Returns:
        The formatted floating-point string.
    """
    return f"{float(value):.{decimal_places}f}"


def format_poscar_numeric_line(
    line: str,
    expected_numeric_fields: int,
    decimal_places: int = OUTPUT_DECIMAL_PLACES,
) -> str:
    """Format the leading numeric fields of one POSCAR line.

    Args:
        line: Raw POSCAR line.
        expected_numeric_fields: Number of leading numeric fields to format.
        decimal_places: Number of digits to keep after the decimal point.

    Returns:
        The reformatted POSCAR line.
    """
    fields: list[str] = line.split()
    formatted_fields: list[str] = [
        format_fixed_precision_float(float(field), decimal_places)
        for field in fields[:expected_numeric_fields]
    ]
    formatted_fields.extend(fields[expected_numeric_fields:])
    return " ".join(formatted_fields)


def normalize_poscar_file(
    poscar_path: str,
    comment_line: str,
    species_order: list[str],
    decimal_places: int = OUTPUT_DECIMAL_PLACES,
) -> None:
    """Rewrite the POSCAR header and numeric precision.

    Args:
        poscar_path: Path to the POSCAR file to normalize.
        comment_line: Free-form POSCAR title/comment line.
        species_order: Chemical symbols in the desired output order.
        decimal_places: Number of digits to keep after the decimal point.
    """
    with open(poscar_path, "r", encoding="utf-8") as file:
        lines: list[str] = file.read().splitlines()

    lines[0] = comment_line
    lines[1] = format_fixed_precision_float(float(lines[1]), decimal_places)
    lines[5] = " ".join(species_order)

    for line_index in range(2, 5):
        lines[line_index] = format_poscar_numeric_line(
            line=lines[line_index],
            expected_numeric_fields=3,
            decimal_places=decimal_places,
        )

    atom_counts: list[int] = [int(token) for token in lines[6].split()]
    total_atoms: int = sum(atom_counts)
    coordinate_mode_line_index: int = 7

    if lines[7].strip().lower().startswith("selective"):
        coordinate_mode_line_index = 8

    first_coordinate_line_index: int = coordinate_mode_line_index + 1
    last_coordinate_line_index: int = first_coordinate_line_index + total_atoms

    for line_index in range(first_coordinate_line_index, last_coordinate_line_index):
        lines[line_index] = format_poscar_numeric_line(
            line=lines[line_index],
            expected_numeric_fields=3,
            decimal_places=decimal_places,
        )

    with open(poscar_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


# ===================================================================
# LAMMPS conf.lmp writer (atom_style atomic)
# ===================================================================

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


def format_lammps_float(
    value: float,
    decimal_places: int = OUTPUT_DECIMAL_PLACES,
) -> str:
    """Format a floating-point value for the LAMMPS data file.

    Args:
        value: Floating-point value to serialize.
        decimal_places: Number of digits to keep after the decimal point.

    Returns:
        A fixed-precision string representation suitable for ``conf.lmp``.
    """
    return format_fixed_precision_float(value, decimal_places)


def write_lammps_conf_lmp(
    atoms: Atoms,
    output_path: str | Path,
    species_order: list[str],
    comment_line: str = "# LAMMPS data file written by initialize_structure_ASE_MgSiO3_bridgmanite.py",
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


def write_initialization_log(
    output_path: str | Path,
    atoms: Atoms,
    a: float,
    b: float,
    c: float,
    reps: tuple[int, int, int],
    species_order: list[str],
    cif_path: str,
    poscar_path: str,
    lammps_path: str,
) -> None:
    """Write a log file summarizing the structure-generation inputs.

    Args:
        output_path: Destination path for the log file.
        atoms: Generated bridgmanite structure.
        a: Input Pbnm lattice parameter ``a`` in Angstrom.
        b: Input Pbnm lattice parameter ``b`` in Angstrom.
        c: Input Pbnm lattice parameter ``c`` in Angstrom.
        reps: Supercell repetitions along the Pbnm ``(a, b, c)`` directions.
        species_order: Requested species ordering in the written outputs.
        cif_path: Output CIF path.
        poscar_path: Output POSCAR path.
        lammps_path: Output LAMMPS data-file path.
    """
    output_file: Path = Path(output_path)
    cell_lengths: np.ndarray = np.array(atoms.cell.lengths(), dtype=float)
    timestamp_utc: str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        f"log.initialize written by {Path(__file__).name}",
        f"timestamp_utc = {timestamp_utc}",
        "",
        "[input_structure]",
        "phase = MgSiO3 bridgmanite",
        "space_group_reference = Pbnm/Pnma (#62)",
        "reference_structure = Horiuchi et al. (1987)",
        f"a_angstrom = {format_fixed_precision_float(a)}",
        f"b_angstrom = {format_fixed_precision_float(b)}",
        f"c_angstrom = {format_fixed_precision_float(c)}",
        f"supercell_reps_abc = {reps[0]} {reps[1]} {reps[2]}",
        f"species_order = {' '.join(species_order)}",
        f"output_decimal_places = {OUTPUT_DECIMAL_PLACES}",
        "",
        "[generated_structure]",
        f"natoms = {len(atoms)}",
        f"chemical_formula = {atoms.get_chemical_formula()}",
        f"cell_lengths_angstrom = {' '.join(format_fixed_precision_float(length) for length in cell_lengths)}",
        f"cell_volume_angstrom3 = {format_fixed_precision_float(atoms.get_volume())}",
        "",
        "[output_files]",
        f"cif = {cif_path}",
        f"poscar = {poscar_path}",
        f"lammps = {lammps_path}",
    ]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===================================================================
# Main entry point
# ===================================================================

if __name__ == "__main__":
    # Bridgmanite species order: Mg, Si, O (cation-first convention)
    SPECIES_ORDER: list[str] = ["Mg", "Si", "O"]

    # Default to the 20-atom unit cell for quick sanity checks.
    # Increase these repetitions later to generate larger supercells.
    SUPERCELL_REPS: tuple[int, int, int] = (2, 2, 2)
    input_a: float = HORIUCHI_PBNM_LATTICE_ANGSTROM[0]
    input_b: float = HORIUCHI_PBNM_LATTICE_ANGSTROM[1]
    input_c: float = HORIUCHI_PBNM_LATTICE_ANGSTROM[2]

    # HORIUCHI_PBNM_LATTICE_ANGSTROM = (4.7754, 4.9292, 6.8969)
    # input_a, input_b, input_c = (4.67, 4.67, 6.64) # (4.67, 4.67, 6.64) # for roughly a 50 GPa, 2000 K compressed cell (ratios changed though for simplicity from (4.598, 4.746, 6.640))

    atoms: Atoms = build_bridgmanite_supercell(
        a=input_a,
        b=input_b,
        c=input_c,
        reps=SUPERCELL_REPS,
    )
    atoms = reorder_atoms_by_species(atoms, SPECIES_ORDER)

    print(atoms)
    print("Number of atoms:", len(atoms))
    print(f"Cell: {atoms.cell.lengths()}")
    print(f"Composition: {atoms.get_chemical_formula()}")

    # CIF for VESTA / visualization
    cif_path: str = f"MgSiO3_bridgmanite_{len(atoms)}atoms.cif"
    write(cif_path, atoms)
    print(f"Written {cif_path}")

    # POSCAR for VASP
    poscar_path: str = f"POSCAR_MgSiO3_bridgmanite_{len(atoms)}atoms"
    write(poscar_path, atoms, direct=True, vasp5=True)
    normalize_poscar_file(poscar_path, "MgSiO3_bridgmanite", SPECIES_ORDER)
    print(f"Written {poscar_path}")

    # conf.lmp for LAMMPS
    lammps_path: str = f"conf.lmp_MgSiO3_bridgmanite_{len(atoms)}atoms"
    write_lammps_conf_lmp(atoms, lammps_path, SPECIES_ORDER)
    print(f"Written {lammps_path}")

    # Initialization log with all structure-generation inputs
    log_path: str = f"log.initialize_MgSiO3_bridgmanite_{len(atoms)}atoms"
    write_initialization_log(
        output_path=log_path,
        atoms=atoms,
        a=input_a,
        b=input_b,
        c=input_c,
        reps=SUPERCELL_REPS,
        species_order=SPECIES_ORDER,
        cif_path=cif_path,
        poscar_path=poscar_path,
        lammps_path=lammps_path,
    )
    print(f"Written {log_path}")
