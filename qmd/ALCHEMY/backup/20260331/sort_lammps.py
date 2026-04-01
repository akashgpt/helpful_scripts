#!/usr/bin/env python3

"""
Sort a LAMMPS data file's Atoms section by existing LAMMPS type ID.

Why this version:
- Keeps the original LAMMPS type mapping (no ASE type remapping).
- Reorders only lines in the Atoms section.
- Leaves atom IDs unchanged to avoid breaking cross-references.

Notes:
- Supports common Atoms styles via header hint, e.g.:
    Atoms
    Atoms # atomic
    Atoms # full
- If style is unknown, defaults to type column index used by 'atomic/charge' style.

Usage:
    python sort_lammps.py                       # conf.lmp -> conf.lmp (in-place with backup)
    python sort_lammps.py input.lmp output.lmp  # custom files
    python $mldp/ALCHEMY/sort_lammps.py conf.lmp

Created by Akash Gupta on 2026-02
"""

import argparse
from pathlib import Path

SECTION_HEADERS = {
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


def is_section_header(line):
    text = line.strip()
    if not text:
        return False
    for section in SECTION_HEADERS:
        if text == section or text.startswith(section + " #"):
            return True
    return False


def detect_type_col(atoms_header):
    header = atoms_header.lower()
    if "#" in header:
        style = header.split("#", 1)[1].strip()
    else:
        style = "atomic"

    if style in {"full", "molecular", "bond", "angle"}:
        return 2
    return 1


def parse_box_bounds(lines):
    xlo = xhi = ylo = yhi = zlo = zhi = None
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[2] == "xlo" and fields[3] == "xhi":
            xlo, xhi = float(fields[0]), float(fields[1])
        elif len(fields) >= 4 and fields[2] == "ylo" and fields[3] == "yhi":
            ylo, yhi = float(fields[0]), float(fields[1])
        elif len(fields) >= 4 and fields[2] == "zlo" and fields[3] == "zhi":
            zlo, zhi = float(fields[0]), float(fields[1])

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError("Could not parse x/y/z box bounds from LAMMPS header")

    return (xlo, xhi), (ylo, yhi), (zlo, zhi)


def wrap_value(value, lo, hi):
    length = hi - lo
    if length <= 0:
        return value
    return ((value - lo) % length) + lo


def parse_atom_line(line, type_col):
    data_part = line.split("#", 1)[0].strip()
    fields = data_part.split()
    if len(fields) <= type_col:
        raise ValueError(f"Cannot parse atom line: {line.rstrip()}")

    atom_id = int(fields[0])
    atom_type = int(fields[type_col])
    return atom_id, atom_type


def rewrite_atom_line(line, new_id, xlo, xhi, ylo, yhi, zlo, zhi):
    if "#" in line:
        data_part, comment_part = line.split("#", 1)
        comment_text = "#" + comment_part.rstrip("\n")
    else:
        data_part = line
        comment_text = ""

    fields = data_part.strip().split()
    if not fields:
        return line if line.endswith("\n") else line + "\n"

    fields[0] = str(new_id)

    if len(fields) >= 5:
        x = wrap_value(float(fields[2]), xlo, xhi)
        y = wrap_value(float(fields[3]), ylo, yhi)
        z = wrap_value(float(fields[4]), zlo, zhi)
        fields[2] = f"{x:.12g}"
        fields[3] = f"{y:.12g}"
        fields[4] = f"{z:.12g}"

    rebuilt = " ".join(fields)

    if comment_text:
        rebuilt = rebuilt + " " + comment_text

    return rebuilt + "\n"


def sort_atoms_section(lines):
    atoms_header_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            atoms_header_idx = idx
            break

    if atoms_header_idx is None:
        raise ValueError("No 'Atoms' section found.")

    (xlo, xhi), (ylo, yhi), (zlo, zhi) = parse_box_bounds(lines[:atoms_header_idx])
    type_col = detect_type_col(lines[atoms_header_idx])

    # First atom data line: after header and optional blank lines/comments.
    start = atoms_header_idx + 1
    while start < len(lines):
        stripped = lines[start].strip()
        if not stripped or stripped.startswith("#"):
            start += 1
            continue
        break

    # End of Atoms data: next known section header.
    end = start
    while end < len(lines):
        stripped = lines[end].strip()
        if is_section_header(stripped):
            break
        end += 1

    atom_lines = []
    passthrough_lines = []

    for line in lines[start:end]:
        stripped = line.strip()
        if not stripped:
            passthrough_lines.append(line)
            continue
        if stripped.startswith("#"):
            passthrough_lines.append(line)
            continue
        atom_lines.append(line)

    parsed = []
    for line in atom_lines:
        atom_id, atom_type = parse_atom_line(line, type_col)
        parsed.append((atom_type, atom_id, line))

    parsed.sort(key=lambda x: (x[0], x[1]))

    sorted_atom_lines = []
    for new_id, item in enumerate(parsed, start=1):
        sorted_atom_lines.append(rewrite_atom_line(item[2], new_id, xlo, xhi, ylo, yhi, zlo, zhi))

    # Reconstruct: keep header/preamble and post-sections unchanged.
    # Put sorted atom entries first in Atoms data block, then passthrough comments/blanks.
    new_lines = lines[:start] + sorted_atom_lines + passthrough_lines + lines[end:]
    return new_lines


def main():
    parser = argparse.ArgumentParser(
        description="Sort Atoms section in a LAMMPS data file by existing LAMMPS type ID."
    )
    parser.add_argument("input_file", nargs="?", default="conf.lmp", help="Input file (default: conf.lmp)")
    parser.add_argument(
        "output_file", nargs="?", default="conf.lmp", help="Output file (default: conf.lmp)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    sorted_lines = sort_atoms_section(lines)

    if input_path.name == "conf.lmp" and output_path.name == "conf.lmp":
        backup_path = input_path.with_name("old.conf.lmp")
        backup_path.write_text("".join(lines), encoding="utf-8")

    output_path.write_text("".join(sorted_lines), encoding="utf-8")

    print(f"Sorted LAMMPS input written to {output_path}")


if __name__ == "__main__":
    main()
