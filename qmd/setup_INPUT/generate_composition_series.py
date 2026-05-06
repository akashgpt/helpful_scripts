"""Generate He-MgSiO3 composition series CSV for constant-N substitution.

Each step removes one MgSiO3 formula unit (5 atoms) and adds 5 He atoms,
keeping the total atom count fixed.  Edit the COLUMNS dict below to add,
remove, or redefine any derived quantity.

Usage:
    python generate_composition_series.py                # writes CSV to stdout
    python generate_composition_series.py  > log.selected_compositions 2>&1  # writes CSV to stdout and selected rows to stderr
    python generate_composition_series.py -o series.csv  # writes CSV to file
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import Callable

# selected IDs (for X_He__MgSiO3_atoms): 2 (0.07), 19 (0.25), 37 (0.50), 55 (0.75), 66 (0.90), 72 (0.99)
# selected IDs (for x_He): 2 (0.07), 6 (0.27), 13 (0.5)
# 19, 37, 55

# ===================================================================
# Tuneable parameters
# ===================================================================

N_MGSIO3_BASE: int = 32        # formula units in the pure-MgSiO3 cell
ATOMS_PER_MGSIO3: int = 5      # Mg + Si + 3 O
HE_STEP: int = 5               # He atoms added per substitution step
DECIMAL_PLACES: int = 4        # precision for float columns

# selected_IDs: list[int] = [19, 37, 55, 66, 72, 2, 6, 13]  # IDs of rows to print to screen (1-based)
# selected_IDs: list[int] = [9, 17, 25, 32, 2, 7, 3]  # IDs of rows to print to screen (1-based)
selected_IDs: list[int] = []  # IDs of rows to print to screen (1-based)


# ===================================================================
# Column definitions
# ===================================================================
# Each entry is  "column_name": callable(n_He, n_MgSiO3) -> value
# Columns appear in the CSV in insertion order.
# Return int, float, or the string "inf" / "nan" for special cases.
#
# To add a new column, just add a new entry here and rerun.
# ===================================================================

COLUMNS: dict[str, Callable[[int, int], int | float | str]] = {

    "n_He": lambda n_He, n_MgSiO3:
        n_He,

    "n_MgSiO3": lambda n_He, n_MgSiO3:
        n_MgSiO3,

    "n_MgSiO3_atoms": lambda n_He, n_MgSiO3:
        n_MgSiO3 * ATOMS_PER_MGSIO3,

    "n_atoms_total": lambda n_He, n_MgSiO3:
        n_He + n_MgSiO3 * ATOMS_PER_MGSIO3,

    # Mole fraction: each He atom and each MgSiO3 unit counts as one species
    "x_He": lambda n_He, n_MgSiO3:
        n_He / (n_He + n_MgSiO3) if (n_He + n_MgSiO3) > 0 else math.nan,

    # Ratio of He atoms to MgSiO3 atoms
    "X_He__MgSiO3_atoms": lambda n_He, n_MgSiO3:
        n_He / (n_MgSiO3 * ATOMS_PER_MGSIO3 + n_He) if (n_MgSiO3 * ATOMS_PER_MGSIO3 + n_He) > 0 else math.inf,

    # --- add new columns below this line ---
    # Example: He atom fraction (wrt total atoms, not formula units)
    # "He_atom_fraction": lambda n_He, n_MgSiO3:
    #     n_He / (n_He + n_MgSiO3 * ATOMS_PER_MGSIO3)
    #         if (n_He + n_MgSiO3 * ATOMS_PER_MGSIO3) > 0 else math.nan,
}


# ===================================================================
# Series generation
# ===================================================================

def generate_series() -> list[dict[str, int | float | str]]:
    """Build the full composition series from pure MgSiO3 to pure He.

    Returns:
        A list of row dicts, one per composition step.
    """
    rows: list[dict[str, int | float | str]] = []
    n_atoms_total: int = N_MGSIO3_BASE * ATOMS_PER_MGSIO3  # constant

    for n_He in range(0, n_atoms_total + 1, HE_STEP):
        n_MgSiO3: int = N_MGSIO3_BASE - n_He // HE_STEP

        row: dict[str, int | float | str] = {"id": len(rows) + 1}
        for col_name, col_func in COLUMNS.items():
            row[col_name] = col_func(n_He, n_MgSiO3)
        rows.append(row)

    return rows


def format_value(value: int | float | str) -> str:
    """Format a single cell value for CSV output.

    Args:
        value: The computed column value.

    Returns:
        A string suitable for writing to CSV.
    """
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return "nan"
        return f"{value:.{DECIMAL_PLACES}f}"
    return str(value)


def write_csv(rows: list[dict[str, int | float | str]], output: str | None) -> None:
    """Write the composition series to a CSV file or stdout.

    Args:
        rows: List of row dicts from generate_series().
        output: File path to write, or None for stdout.
    """
    col_names: list[str] = ["id"] + list(COLUMNS.keys())

    if output is not None:
        dest = open(output, "w", newline="", encoding="utf-8")
    else:
        dest = sys.stdout

    try:
        writer = csv.writer(dest)
        writer.writerow(col_names)
        for row in rows:
            writer.writerow([format_value(row[c]) for c in col_names])
    finally:
        if output is not None:
            dest.close()


def print_selected_rows(
    rows: list[dict[str, int | float | str]],
    ids: list[int],
) -> None:
    """Print a formatted table of selected rows to stderr.

    Args:
        rows: Full list of row dicts from generate_series().
        ids: 1-based row IDs to display.
    """
    col_names: list[str] = ["id"] + list(COLUMNS.keys())

    # Build formatted cell values for selected rows
    selected: list[list[str]] = []
    for row in rows:
        if row["id"] in ids:
            selected.append([format_value(row[c]) for c in col_names])

    if not selected:
        print("\nNo rows matched selected_IDs.", file=sys.stderr)
        return

    # Compute column widths (header vs data)
    col_widths: list[int] = [
        max(len(name), *(len(r[i]) for r in selected))
        for i, name in enumerate(col_names)
    ]

    # Print
    sep: str = "  "
    header: str = sep.join(name.rjust(w) for name, w in zip(col_names, col_widths))
    divider: str = sep.join("-" * w for w in col_widths)

    print(f"\nSelected rows (IDs: {ids}):", file=sys.stderr)
    print(header, file=sys.stderr)
    print(divider, file=sys.stderr)
    for cells in selected:
        print(sep.join(c.rjust(w) for c, w in zip(cells, col_widths)), file=sys.stderr)
    print(file=sys.stderr)


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    """Parse arguments and write the composition series CSV."""
    parser = argparse.ArgumentParser(
        description="Generate He-MgSiO3 composition series CSV.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="He_MgSiO3_composition_series.csv",
        help="Output CSV path (default: He_MgSiO3_composition_series.csv).",
    )
    args = parser.parse_args()

    rows = generate_series()
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)

    # Print selected rows to screen
    if selected_IDs:
        print_selected_rows(rows, selected_IDs)


if __name__ == "__main__":
    main()
