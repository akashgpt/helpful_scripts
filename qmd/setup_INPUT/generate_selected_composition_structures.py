"""Generate selected He-substituted MgSiO3 POSCAR and LAMMPS files.

This script starts from the base 360-atom bridgmanite POSCAR, then creates the
selected compositions listed in ``log.selected_compositions``. Each output
keeps the total atom count fixed by replacing stoichiometric MgSiO3 content
with He:

    1 Mg + 1 Si + 3 O -> 5 He

To concentrate the He-rich region toward one end of the z axis without moving
any atoms, the script replaces the highest-z Mg, Si, and O atoms needed for
each composition. The generated outputs always use the species order:

    Mg, Si, O, He

Usage:
    python generate_selected_composition_structures.py
"""

from __future__ import print_function

import csv
import glob
import os
import re
from typing import Dict
from typing import List
from typing import Tuple


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
print("Base directory: {0}".format(BASE_DIR))
_base_poscar_matches: List[str] = sorted(
    glob.glob(os.path.join(BASE_DIR, "POSCAR_MgSiO3_bridgmanite_*atoms"))
)
if len(_base_poscar_matches) != 1:
    raise FileNotFoundError(
        "Expected exactly 1 base POSCAR matching 'POSCAR_MgSiO3_bridgmanite_*atoms', "
        "found {0}: {1}".format(len(_base_poscar_matches), _base_poscar_matches)
    )
BASE_POSCAR_PATH: str = _base_poscar_matches[0]
COMPOSITION_CSV_PATH: str = os.path.join(
    BASE_DIR,
    "He_MgSiO3_composition_series.csv",
)
SELECTED_LOG_PATH: str = os.path.join(
    BASE_DIR,
    "log.selected_compositions",
)
OUTPUT_DIR: str = os.path.join(
    BASE_DIR,
    "selected_compositions",
)

OUTPUT_DECIMAL_PLACES: int = 6
BASE_SPECIES_ORDER: List[str] = ["Mg", "Si", "O"]
OUTPUT_SPECIES_ORDER: List[str] = ["Mg", "Si", "O", "He"]
SPECIES_PER_FORMULA_UNIT: Dict[str, int] = {
    "Mg": 1,
    "Si": 1,
    "O": 3,
}
LAMMPS_MASSES: Dict[str, float] = {
    "Mg": 24.305000,
    "Si": 28.085500,
    "O": 15.999400,
    "He": 4.002602,
}


class PoscarData(object):
    """Container for parsed POSCAR data."""

    def __init__(
        self,
        comment_line: str,
        cell_vectors: List[List[float]],
        species_to_coords: Dict[str, List[List[float]]],
    ) -> None:
        """Initialize the parsed POSCAR container.

        Args:
            comment_line: First POSCAR comment line.
            cell_vectors: ``3 x 3`` cell matrix as nested float lists.
            species_to_coords: Mapping from species label to fractional
                coordinate list.
        """
        self.comment_line = comment_line
        self.cell_vectors = cell_vectors
        self.species_to_coords = species_to_coords


def format_fixed_precision_float(value: float) -> str:
    """Format a float with fixed decimal precision.

    Args:
        value: Floating-point value to format.

    Returns:
        A fixed-width decimal string.
    """
    return "{0:.{1}f}".format(float(value), OUTPUT_DECIMAL_PLACES)


def parse_selected_ids(log_path: str) -> List[int]:
    """Parse the selected composition IDs from the log file.

    Args:
        log_path: Path to ``log.selected_compositions``.

    Returns:
        Selected row IDs in the order written in the log file.
    """
    with open(log_path, "r") as file_handle:
        log_text = file_handle.read()

    match = re.search(r"Selected rows \(IDs: \[(.*?)\]\):", log_text)
    if match is None:
        raise ValueError("Could not find selected IDs in log.selected_compositions.")

    id_tokens = [token.strip() for token in match.group(1).split(",")]
    return [int(token) for token in id_tokens if token]


def load_composition_rows(csv_path: str) -> Dict[int, Dict[str, str]]:
    """Load composition rows keyed by integer row ID.

    Args:
        csv_path: Path to the composition CSV.

    Returns:
        Mapping from row ID to parsed CSV row.
    """
    rows_by_id: Dict[int, Dict[str, str]] = {}
    with open(csv_path, "r") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            rows_by_id[int(row["id"])] = row
    return rows_by_id


def parse_poscar(poscar_path: str) -> PoscarData:
    """Parse the base POSCAR file.

    Args:
        poscar_path: Path to the POSCAR file.

    Returns:
        Parsed POSCAR data grouped by species.
    """
    with open(poscar_path, "r") as file_handle:
        lines = [line.rstrip("\n") for line in file_handle]

    comment_line = lines[0]
    scale_factor = float(lines[1].strip())
    cell_vectors: List[List[float]] = []
    for line_index in range(2, 5):
        vector = [float(value) * scale_factor for value in lines[line_index].split()]
        cell_vectors.append(vector)

    species_labels = lines[5].split()
    species_counts = [int(token) for token in lines[6].split()]
    coordinate_mode = lines[7].strip().lower()
    if coordinate_mode != "direct":
        raise ValueError("This generator expects the base POSCAR to use Direct coordinates.")

    first_coordinate_line = 8
    species_to_coords: Dict[str, List[List[float]]] = {}
    running_index = first_coordinate_line
    for species_label, count in zip(species_labels, species_counts):
        species_coords: List[List[float]] = []
        for _ in range(count):
            coords = [float(value) for value in lines[running_index].split()[:3]]
            species_coords.append(coords)
            running_index += 1
        species_to_coords[species_label] = species_coords

    return PoscarData(
        comment_line=comment_line,
        cell_vectors=cell_vectors,
        species_to_coords=species_to_coords,
    )


def sort_indices_by_descending_z(coords: List[List[float]]) -> List[int]:
    """Return coordinate indices sorted by descending z value.

    Args:
        coords: Fractional coordinate list.

    Returns:
        Indices sorted from highest z to lowest z.
    """
    return sorted(
        range(len(coords)),
        key=lambda index: (coords[index][2], coords[index][1], coords[index][0], -index),
        reverse=True,
    )


def select_replacement_indices(
    base_data: PoscarData,
    n_formula_units_to_replace: int,
) -> Dict[str, List[int]]:
    """Choose stoichiometric replacement indices concentrated at high z.

    Args:
        base_data: Parsed base POSCAR data.
        n_formula_units_to_replace: Number of MgSiO3 units to replace.

    Returns:
        Mapping from base species to selected index list.
    """
    selected_indices: Dict[str, List[int]] = {}
    for species_label in BASE_SPECIES_ORDER:
        n_required = SPECIES_PER_FORMULA_UNIT[species_label] * n_formula_units_to_replace
        candidate_indices = sort_indices_by_descending_z(
            base_data.species_to_coords[species_label]
        )
        selected_indices[species_label] = candidate_indices[:n_required]
    return selected_indices


def split_species_coordinates(
    base_data: PoscarData,
    selected_indices: Dict[str, List[int]],
) -> Tuple[Dict[str, List[List[float]]], List[List[float]]]:
    """Split coordinates into surviving species and replaced He positions.

    Args:
        base_data: Parsed base POSCAR data.
        selected_indices: Selected replacement indices by species.

    Returns:
        Tuple of ``(remaining_species_coords, he_coords)``.
    """
    remaining_species_coords: Dict[str, List[List[float]]] = {}
    he_coords: List[List[float]] = []

    for species_label in BASE_SPECIES_ORDER:
        chosen_index_set = set(selected_indices[species_label])
        remaining_species_coords[species_label] = []
        for atom_index, coord in enumerate(base_data.species_to_coords[species_label]):
            if atom_index in chosen_index_set:
                he_coords.append(coord)
            else:
                remaining_species_coords[species_label].append(coord)

    he_coords.sort(
        key=lambda coord: (coord[2], coord[1], coord[0]),
        reverse=True,
    )
    return remaining_species_coords, he_coords


def get_output_stem(row_id: int, n_he: int, n_mgsio3: int) -> str:
    """Build a descriptive filename stem for one composition.

    Args:
        row_id: Selected composition row ID.
        n_he: Number of He atoms in the structure.
        n_mgsio3: Number of MgSiO3 formula units remaining.

    Returns:
        A filename stem without the POSCAR/conf prefix.
    """
    return "id{0:02d}__He{1:03d}__MgSiO3_{2:02d}fu__360atoms".format(
        row_id,
        n_he,
        n_mgsio3,
    )


def write_poscar(
    output_path: str,
    comment_line: str,
    cell_vectors: List[List[float]],
    remaining_species_coords: Dict[str, List[List[float]]],
    he_coords: List[List[float]],
) -> None:
    """Write one POSCAR with species ordered as Mg, Si, O, He.

    Args:
        output_path: Destination POSCAR path.
        comment_line: POSCAR comment line.
        cell_vectors: ``3 x 3`` cell matrix as nested float lists.
        remaining_species_coords: Surviving Mg/Si/O coordinates.
        he_coords: He coordinates.
    """
    species_counts = [
        len(remaining_species_coords["Mg"]),
        len(remaining_species_coords["Si"]),
        len(remaining_species_coords["O"]),
        len(he_coords),
    ]
    ordered_coords = (
        remaining_species_coords["Mg"]
        + remaining_species_coords["Si"]
        + remaining_species_coords["O"]
        + he_coords
    )

    lines = [
        comment_line,
        format_fixed_precision_float(1.0),
    ]
    for vector in cell_vectors:
        lines.append(" ".join(format_fixed_precision_float(value) for value in vector))
    lines.append(" ".join(OUTPUT_SPECIES_ORDER))
    lines.append(" ".join("{0:3d}".format(count) for count in species_counts))
    lines.append("Direct")
    for coord in ordered_coords:
        lines.append(" ".join(format_fixed_precision_float(value) for value in coord))

    with open(output_path, "w") as file_handle:
        file_handle.write("\n".join(lines) + "\n")


def fractional_to_cartesian(
    cell_vectors: List[List[float]],
    frac_coord: List[float],
) -> List[float]:
    """Convert one fractional coordinate into Cartesian coordinates.

    Args:
        cell_vectors: ``3 x 3`` cell matrix as nested float lists.
        frac_coord: Fractional coordinate list.

    Returns:
        Cartesian coordinate list.
    """
    x_value = (
        frac_coord[0] * cell_vectors[0][0]
        + frac_coord[1] * cell_vectors[1][0]
        + frac_coord[2] * cell_vectors[2][0]
    )
    y_value = (
        frac_coord[0] * cell_vectors[0][1]
        + frac_coord[1] * cell_vectors[1][1]
        + frac_coord[2] * cell_vectors[2][1]
    )
    z_value = (
        frac_coord[0] * cell_vectors[0][2]
        + frac_coord[1] * cell_vectors[1][2]
        + frac_coord[2] * cell_vectors[2][2]
    )
    return [x_value, y_value, z_value]


def write_lammps_conf(
    output_path: str,
    cell_vectors: List[List[float]],
    remaining_species_coords: Dict[str, List[List[float]]],
    he_coords: List[List[float]],
) -> None:
    """Write one LAMMPS ``conf.lmp`` file with Mg, Si, O, He ordering.

    Args:
        output_path: Destination ``conf.lmp`` path.
        cell_vectors: ``3 x 3`` cell matrix as nested float lists.
        remaining_species_coords: Surviving Mg/Si/O coordinates.
        he_coords: He coordinates.
    """
    ordered_coords = (
        [("Mg", coord) for coord in remaining_species_coords["Mg"]]
        + [("Si", coord) for coord in remaining_species_coords["Si"]]
        + [("O", coord) for coord in remaining_species_coords["O"]]
        + [("He", coord) for coord in he_coords]
    )
    type_by_species = {
        "Mg": 1,
        "Si": 2,
        "O": 3,
        "He": 4,
    }

    lines = [
        "# LAMMPS data file written by generate_selected_composition_structures.py",
        "",
        "{0} atoms".format(len(ordered_coords)),
        "4 atom types",
        "",
        "0.0 {0} xlo xhi".format(format_fixed_precision_float(cell_vectors[0][0])),
        "0.0 {0} ylo yhi".format(format_fixed_precision_float(cell_vectors[1][1])),
        "0.0 {0} zlo zhi".format(format_fixed_precision_float(cell_vectors[2][2])),
        "",
        "Atom Type Labels",
        "",
    ]

    for species_label in OUTPUT_SPECIES_ORDER:
        lines.append("{0} {1}".format(type_by_species[species_label], species_label))

    lines.extend(["", "Masses", ""])
    for species_label in OUTPUT_SPECIES_ORDER:
        lines.append(
            "{0} {1}  # {2}".format(
                type_by_species[species_label],
                format_fixed_precision_float(LAMMPS_MASSES[species_label]),
                species_label,
            )
        )

    lines.extend(["", "Atoms  # atomic", ""])
    for atom_id, (species_label, frac_coord) in enumerate(ordered_coords, start=1):
        cart_coord = fractional_to_cartesian(cell_vectors, frac_coord)
        lines.append(
            "{0} {1} {2} {3} {4}".format(
                atom_id,
                type_by_species[species_label],
                format_fixed_precision_float(cart_coord[0]),
                format_fixed_precision_float(cart_coord[1]),
                format_fixed_precision_float(cart_coord[2]),
            )
        )

    with open(output_path, "w") as file_handle:
        file_handle.write("\n".join(lines) + "\n")


def compute_z_statistics(
    cell_vectors: List[List[float]],
    remaining_species_coords: Dict[str, List[List[float]]],
    he_coords: List[List[float]],
) -> Tuple[float, float, float]:
    """Compute z-mean separation statistics for one generated structure.

    Args:
        cell_vectors: ``3 x 3`` cell matrix as nested float lists.
        remaining_species_coords: Surviving Mg/Si/O coordinates.
        he_coords: He coordinates.

    Returns:
        Tuple of ``(he_z_mean, rest_z_mean, absolute_gap)`` in Angstrom.
    """
    he_z_values = [
        fractional_to_cartesian(cell_vectors, coord)[2]
        for coord in he_coords
    ]
    rest_coords = (
        remaining_species_coords["Mg"]
        + remaining_species_coords["Si"]
        + remaining_species_coords["O"]
    )
    rest_z_values = [
        fractional_to_cartesian(cell_vectors, coord)[2]
        for coord in rest_coords
    ]
    he_z_mean = sum(he_z_values) / float(len(he_z_values))
    rest_z_mean = sum(rest_z_values) / float(len(rest_z_values))
    return he_z_mean, rest_z_mean, abs(he_z_mean - rest_z_mean)


def write_summary_log(output_dir: str, summary_lines: List[str]) -> None:
    """Write a summary log for the generated structures.

    Args:
        output_dir: Destination output directory.
        summary_lines: Preformatted summary lines.
    """
    log_path = os.path.join(output_dir, "log.generated_selected_structures")
    with open(log_path, "w") as file_handle:
        file_handle.write("\n".join(summary_lines) + "\n")


def main() -> None:
    """Generate all selected composition structures."""
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    selected_ids = parse_selected_ids(SELECTED_LOG_PATH)
    rows_by_id = load_composition_rows(COMPOSITION_CSV_PATH)
    base_data = parse_poscar(BASE_POSCAR_PATH)

    summary_lines = [
        "Generated He-substituted structures from the base 360-atom MgSiO3 POSCAR.",
        "Selection strategy: replace the highest-z Mg, Si, and O atoms needed",
        "for each stoichiometric MgSiO3 -> 5 He substitution, without moving atoms.",
        "Species order in all outputs: Mg Si O He",
        "",
        "id  n_He  n_MgSiO3  he_z_mean_A  rest_z_mean_A  abs_gap_A  stem",
        "--  ----  --------  -----------  -------------  ---------  ----",
    ]

    for row_id in selected_ids:
        row = rows_by_id[row_id]
        n_he = int(row["n_He"])
        n_mgsio3 = int(row["n_MgSiO3"])
        n_formula_units_to_replace = n_he // 5

        selected_indices = select_replacement_indices(
            base_data=base_data,
            n_formula_units_to_replace=n_formula_units_to_replace,
        )
        remaining_species_coords, he_coords = split_species_coordinates(
            base_data=base_data,
            selected_indices=selected_indices,
        )

        stem = get_output_stem(
            row_id=row_id,
            n_he=n_he,
            n_mgsio3=n_mgsio3,
        )
        poscar_path = os.path.join(OUTPUT_DIR, "POSCAR__{0}".format(stem))
        lammps_path = os.path.join(OUTPUT_DIR, "conf.lmp__{0}".format(stem))

        write_poscar(
            output_path=poscar_path,
            comment_line=stem,
            cell_vectors=base_data.cell_vectors,
            remaining_species_coords=remaining_species_coords,
            he_coords=he_coords,
        )
        write_lammps_conf(
            output_path=lammps_path,
            cell_vectors=base_data.cell_vectors,
            remaining_species_coords=remaining_species_coords,
            he_coords=he_coords,
        )

        he_z_mean, rest_z_mean, abs_gap = compute_z_statistics(
            cell_vectors=base_data.cell_vectors,
            remaining_species_coords=remaining_species_coords,
            he_coords=he_coords,
        )
        summary_lines.append(
            "{0:2d}  {1:4d}  {2:8d}  {3:11s}  {4:13s}  {5:9s}  {6}".format(
                row_id,
                n_he,
                n_mgsio3,
                format_fixed_precision_float(he_z_mean),
                format_fixed_precision_float(rest_z_mean),
                format_fixed_precision_float(abs_gap),
                stem,
            )
        )

        print("Wrote {0}".format(poscar_path))
        print("Wrote {0}".format(lammps_path))

    write_summary_log(
        output_dir=OUTPUT_DIR,
        summary_lines=summary_lines,
    )
    print("Wrote {0}".format(os.path.join(OUTPUT_DIR, "log.generated_selected_structures")))


if __name__ == "__main__":
    main()
