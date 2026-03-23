#!/usr/bin/env python3

"""
##############################################################################
# collect_plumed_cv_parameters.py
#
# Collect per-zone CV ranges from `md/ZONE_*/plumed.info` files for one or
# more iterations and print paste-ready `PLUMED_CV_PARAMETERS` blocks.
#
# Usage:
#   python collect_plumed_cv_parameters.py sim_data_ML_v4
#   python collect_plumed_cv_parameters.py sim_data_ML_v4 --iteration v8_i2
#   python collect_plumed_cv_parameters.py sim_data_ML_v4 --energy-field kjmol
#   python collect_plumed_cv_parameters.py sim_data_ML_v4 -CV3min 0 -CV3max 300
#
# Notes:
#   - CV1 and CV2 are read from each iteration's `md/ZONE_*/plumed.info`.
#   - CV3 defaults to `0 1` for every zone unless overridden with
#     `-CV3min` and `-CV3max`.
#   - By default, CV1 is read from the `Energy/TOTEN (kJ/mol)` line, matching
#     the values currently being referenced from `plumed.info`.
#   - Use `--energy-field ev` if you want the `Energy/TOTEN (eV)` line instead.
#   - The script prints the formatted blocks to stdout and can optionally write
#     the same content to a separate file.
##############################################################################
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys


ITERATION_INDEX_RE = re.compile(r"_i(\d+)$")
ZONE_INDEX_RE = re.compile(r"ZONE_(\d+)$")
HEADER_LINE = "PLUMED_CV_PARAMETERS #Format: CV1_MIN CV1_MAX CV2_MIN CV2_MAX CV3_MIN CV3_MAX"
ENERGY_LABELS = {
    "ev": "Energy/TOTEN (eV):",
    "kjmol": "Energy/TOTEN (kJ/mol):",
}


@dataclass(frozen=True)
class ZoneCvParameters:
    """Store formatted CV parameters for one zone.

    Attributes:
        zone_id: Numeric zone identifier, e.g. 8 for `ZONE_8`.
        cv1_min: Minimum value for CV1.
        cv1_max: Maximum value for CV1.
        cv2_min: Minimum value for CV2.
        cv2_max: Maximum value for CV2.
        cv3_min: Minimum value for CV3.
        cv3_max: Maximum value for CV3.
    """

    zone_id: int
    cv1_min: str
    cv1_max: str
    cv2_min: str
    cv2_max: str
    cv3_min: str
    cv3_max: str

    def format_parameter_line(self) -> str:
        """Return one paste-ready parameter line for `TRAIN_MLMD_parameters.txt`.

        Returns:
            str: Space-delimited CV limits in the expected order.
        """

        parameter_values = " ".join(
            [self.cv1_min, self.cv1_max, self.cv2_min, self.cv2_max, self.cv3_min, self.cv3_max]
        )
        return f"{parameter_values} # ZONE_{self.zone_id}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Collect PLUMED CV parameter lines from per-zone plumed.info files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        type=Path,
        default=Path("sim_data_ML_v4"),
        help="Base simulation directory containing iteration folders. Default: sim_data_ML_v4",
    )
    parser.add_argument(
        "--iteration",
        action="append",
        default=[],
        help="Specific iteration directory name to process. Repeat to process multiple iterations.",
    )
    parser.add_argument(
        "--parameter-file",
        type=Path,
        help="Path to TRAIN_MLMD parameter file. Default: <base_dir>/TRAIN_MLMD_parameters.txt",
    )
    parser.add_argument(
        "--energy-field",
        choices=("ev", "kjmol"),
        default="kjmol",
        help="Which Energy/TOTEN line from plumed.info to use for CV1. Default: kjmol",
    )
    parser.add_argument(
        "-CV3min",
        "--cv3min",
        type=float,
        default=0.0,
        help="Override CV3 minimum for every zone. Default: 0",
    )
    parser.add_argument(
        "-CV3max",
        "--cv3max",
        type=float,
        default=1.0,
        help="Override CV3 maximum for every zone. Default: 1",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output file for the formatted report.",
    )
    return parser.parse_args()


def format_cli_number(value: float) -> str:
    """Format a numeric CLI value without unnecessary trailing zeros.

    Args:
        value: Numeric value to format.

    Returns:
        str: Compact string representation.
    """

    return f"{value:g}"


def read_lines(path: Path) -> list[str]:
    """Read a text file into memory.

    Args:
        path: File to read.

    Returns:
        list[str]: All lines from the file.
    """

    with path.open("r", encoding="utf-8") as handle:
        return handle.readlines()


def read_single_value_keyword(parameter_file: Path, keyword: str) -> str | None:
    """Read a single-value keyword from a TRAIN_MLMD parameter file.

    Args:
        parameter_file: Parameter file to scan.
        keyword: Keyword whose next non-comment line should be returned.

    Returns:
        str | None: Value line for the keyword, or None if not found.
    """

    lines = read_lines(parameter_file)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.split()[0] != keyword:
            continue

        for next_line in lines[index + 1 :]:
            candidate = next_line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            return candidate.split("#", 1)[0].strip()

    return None


def iteration_sort_key(iteration_dir: Path) -> tuple[int, str]:
    """Build a stable sort key for iteration directories.

    Args:
        iteration_dir: Iteration directory path.

    Returns:
        tuple[int, str]: Numeric iteration index and directory name.
    """

    match = ITERATION_INDEX_RE.search(iteration_dir.name)
    if match is None:
        return sys.maxsize, iteration_dir.name
    return int(match.group(1)), iteration_dir.name


def discover_iterations(base_dir: Path, parameter_file: Path, requested_iterations: list[str]) -> list[Path]:
    """Determine which iteration directories should be processed.

    Args:
        base_dir: Base simulation directory.
        parameter_file: TRAIN_MLMD parameter file for metadata lookup.
        requested_iterations: Explicit iteration names from the CLI.

    Returns:
        list[Path]: Iteration directories in processing order.

    Raises:
        FileNotFoundError: If a requested iteration does not exist.
        RuntimeError: If no iteration directories are found.
    """

    if requested_iterations:
        iteration_dirs = []
        for iteration_name in requested_iterations:
            iteration_dir = (base_dir / iteration_name).resolve()
            if not iteration_dir.is_dir():
                raise FileNotFoundError(f"Iteration directory not found: {iteration_dir}")
            iteration_dirs.append(iteration_dir)
        return sorted(iteration_dirs, key=iteration_sort_key)

    runid_prefix = read_single_value_keyword(parameter_file, "RUNID_PREFIX") if parameter_file.is_file() else None
    if runid_prefix:
        candidates = [path for path in base_dir.glob(f"{runid_prefix}*") if path.is_dir()]
    else:
        candidates = [path for path in base_dir.iterdir() if path.is_dir() and (path / "md").is_dir()]

    iteration_dirs = sorted(candidates, key=iteration_sort_key)
    if not iteration_dirs:
        raise RuntimeError(f"No iteration directories found under {base_dir}")
    return iteration_dirs


def extract_range_from_line(line: str, label: str) -> tuple[str, str]:
    """Extract the two values from a `label: min to max` line.

    Args:
        line: Line to parse.
        label: Expected line prefix.

    Returns:
        tuple[str, str]: Minimum and maximum values as strings.

    Raises:
        ValueError: If the line does not have the expected format.
    """

    if not line.strip().startswith(label):
        raise ValueError(f"Line does not start with '{label}': {line.rstrip()}")

    _, value_text = line.split(":", 1)
    parts = [part.strip() for part in value_text.split("to")]
    if len(parts) != 2:
        raise ValueError(f"Could not parse min/max range from line: {line.rstrip()}")
    return parts[0], parts[1]


def parse_zone_id(zone_dir: Path) -> int:
    """Extract the numeric zone identifier from a zone directory name.

    Args:
        zone_dir: Zone directory path.

    Returns:
        int: Numeric zone identifier.

    Raises:
        ValueError: If the directory name does not look like `ZONE_<n>`.
    """

    match = ZONE_INDEX_RE.fullmatch(zone_dir.name)
    if match is None:
        raise ValueError(f"Invalid zone directory name: {zone_dir.name}")
    return int(match.group(1))


def parse_plumed_info(
    plumed_info_file: Path,
    energy_field: str,
    cv3_min: str,
    cv3_max: str,
) -> ZoneCvParameters:
    """Parse one `plumed.info` file into a zone CV record.

    Args:
        plumed_info_file: Path to `plumed.info`.
        energy_field: Which Energy/TOTEN line to use for CV1 (`ev` or `kjmol`).
        cv3_min: CV3 minimum to apply to this zone.
        cv3_max: CV3 maximum to apply to this zone.

    Returns:
        ZoneCvParameters: Parsed and formatted CV values.

    Raises:
        ValueError: If required lines cannot be found.
    """

    energy_label = ENERGY_LABELS[energy_field]
    cv1_min: str | None = None
    cv1_max: str | None = None
    cv2_min: str | None = None
    cv2_max: str | None = None

    for line in read_lines(plumed_info_file):
        stripped = line.strip()
        if stripped.startswith(energy_label):
            cv1_min, cv1_max = extract_range_from_line(line, energy_label)
        elif stripped.startswith("Volume"):
            cv2_min, cv2_max = extract_range_from_line(line, "Volume")

    if cv1_min is None or cv1_max is None:
        raise ValueError(f"Could not find '{energy_label}' in {plumed_info_file}")
    if cv2_min is None or cv2_max is None:
        raise ValueError(f"Could not find 'Volume' line in {plumed_info_file}")

    zone_id = parse_zone_id(plumed_info_file.parent)
    return ZoneCvParameters(
        zone_id=zone_id,
        cv1_min=cv1_min,
        cv1_max=cv1_max,
        cv2_min=cv2_min,
        cv2_max=cv2_max,
        cv3_min=cv3_min,
        cv3_max=cv3_max,
    )


def collect_iteration_records(
    iteration_dir: Path,
    expected_zone_count: int | None,
    energy_field: str,
    cv3_min: str,
    cv3_max: str,
) -> list[ZoneCvParameters]:
    """Collect CV records for all zones in one iteration.

    Args:
        iteration_dir: Iteration directory to scan.
        expected_zone_count: Expected number of zones, or None to skip that check.
        energy_field: Which Energy/TOTEN line to use for CV1.
        cv3_min: CV3 minimum to apply to every zone.
        cv3_max: CV3 maximum to apply to every zone.

    Returns:
        list[ZoneCvParameters]: Zone records sorted by zone id.

    Raises:
        FileNotFoundError: If expected `plumed.info` files are missing.
        RuntimeError: If no zone records can be collected.
    """

    md_dir = iteration_dir / "md"
    if not md_dir.is_dir():
        raise FileNotFoundError(f"MD directory not found: {md_dir}")

    zone_records: list[ZoneCvParameters] = []
    found_zone_ids: set[int] = set()

    for zone_dir in sorted((path for path in md_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
        if ZONE_INDEX_RE.fullmatch(zone_dir.name) is None:
            continue

        zone_id = parse_zone_id(zone_dir)
        plumed_info_file = zone_dir / "plumed.info"
        if not plumed_info_file.is_file():
            raise FileNotFoundError(f"Missing plumed.info file: {plumed_info_file}")

        zone_records.append(parse_plumed_info(plumed_info_file, energy_field, cv3_min, cv3_max))
        found_zone_ids.add(zone_id)

    if not zone_records:
        raise RuntimeError(f"No zone plumed.info files found in {md_dir}")

    if expected_zone_count is not None:
        missing_zone_ids = sorted(set(range(1, expected_zone_count + 1)) - found_zone_ids)
        if missing_zone_ids:
            missing_labels = ", ".join(f"ZONE_{zone_id}" for zone_id in missing_zone_ids)
            raise FileNotFoundError(f"Missing zones in {iteration_dir.name}: {missing_labels}")

    return sorted(zone_records, key=lambda record: record.zone_id)


def build_iteration_block(iteration_name: str, energy_field: str, records: list[ZoneCvParameters]) -> str:
    """Build the formatted text block for one iteration.

    Args:
        iteration_name: Iteration name, e.g. `v8_i2`.
        energy_field: Which Energy/TOTEN line supplied CV1.
        records: Zone records for the iteration.

    Returns:
        str: Human-readable, paste-ready text block.
    """

    energy_label = ENERGY_LABELS[energy_field].replace(":", "")
    lines = [
        "#" * 80,
        f"# Iteration: {iteration_name}",
        f"# CV1 source from plumed.info: {energy_label}",
        HEADER_LINE,
    ]

    for record in records:
        lines.append(record.format_parameter_line())

    return "\n".join(lines)


def maybe_write_output(output_path: Path, content: str) -> None:
    """Write the formatted report to an optional output file.

    Args:
        output_path: Output file path.
        content: Text to write.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    """Run the CLI workflow for collecting PLUMED CV parameters."""

    args = parse_args()
    base_dir = args.base_dir.resolve()
    parameter_file = (args.parameter_file or (base_dir / "TRAIN_MLMD_parameters.txt")).resolve()

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    expected_zone_count_text = read_single_value_keyword(parameter_file, "N_ZONES_PTX") if parameter_file.is_file() else None
    expected_zone_count = int(expected_zone_count_text) if expected_zone_count_text else None

    cv3_min = format_cli_number(args.cv3min)
    cv3_max = format_cli_number(args.cv3max)

    iteration_dirs = discover_iterations(base_dir, parameter_file, args.iteration)
    iteration_blocks = []

    for iteration_dir in iteration_dirs:
        records = collect_iteration_records(
            iteration_dir=iteration_dir,
            expected_zone_count=expected_zone_count,
            energy_field=args.energy_field,
            cv3_min=cv3_min,
            cv3_max=cv3_max,
        )
        iteration_blocks.append(build_iteration_block(iteration_dir.name, args.energy_field, records))

    report = "\n\n".join(iteration_blocks) + "\n"
    print(report, end="")

    if args.output is not None:
        output_path = args.output.resolve()
        maybe_write_output(output_path, report)
        print(f"\nSaved report to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
