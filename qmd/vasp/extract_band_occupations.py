#!/usr/bin/env python3
"""Summarize the occupied-band table from a VASP OUTCAR file.

This helper parses the second-last ``band No.  band energies     occupation``
table by default, because the final occurrence in an active or truncated
``OUTCAR`` can be incomplete. It writes a compact key-value summary that can be
consumed by the shell-based ``data_4_analysis*`` workflows.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


BAND_TABLE_HEADER: str = "band No.  band energies     occupation"
BAND_LINE_PATTERN: re.Pattern[str] = re.compile(
	r"^\s*(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$"
)
INTEGER_TOLERANCE: float = 1.0e-8
ZERO_TOLERANCE: float = 1.0e-8


@dataclass(frozen=True)
class BandRecord:
	"""Store one row from a VASP band-occupation table.

	Attributes:
		band_number: Sequential band index from the table.
		energy: Band energy in eV.
		occupation: Occupation value reported by VASP.
	"""

	band_number: int
	energy: float
	occupation: float


@dataclass(frozen=True)
class BandSummary:
	"""Store the aggregate occupancy summary for one selected band table.

	Attributes:
		table_occurrence_count: Number of matching tables found in the OUTCAR.
		selected_table_index_1based: One-based index of the table used.
		selected_table_line_1based: One-based line number of the chosen table.
		total_bands: Highest ``band No.`` found in the chosen table.
		full_occupation_bands: Bands with non-zero integer occupation.
		partial_occupation_bands: Bands with non-zero non-integer occupation.
		zero_occupation_bands: Bands with effectively zero occupation.
		nonzero_occupation_bands: All bands with occupation greater than zero.
		negative_energy_occupied_bands: Occupied bands with negative energy.
		non_negative_energy_occupied_bands: Occupied bands with non-negative energy.
		min_occupied_energy: Minimum energy among occupied bands.
		max_occupied_energy: Maximum energy among occupied bands.
		flag_no_nonzero_occupied_bands: Whether no occupied bands were found.
	"""

	table_occurrence_count: int
	selected_table_index_1based: int
	selected_table_line_1based: int
	total_bands: int
	full_occupation_bands: int
	partial_occupation_bands: int
	zero_occupation_bands: int
	nonzero_occupation_bands: int
	negative_energy_occupied_bands: int
	non_negative_energy_occupied_bands: int
	min_occupied_energy: float | None
	max_occupied_energy: float | None
	flag_no_nonzero_occupied_bands: bool


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments.

	Returns:
		The parsed arguments namespace.
	"""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Summarize occupied-band information from a VASP OUTCAR."
	)
	parser.add_argument(
		"--outcar",
		type=Path,
		default=Path("OUTCAR"),
		help="Path to the OUTCAR file. Default: %(default)s",
	)
	parser.add_argument(
		"--output",
		type=Path,
		required=True,
		help="Path to the summary output file.",
	)
	parser.add_argument(
		"--selection",
		choices=("second_last", "last"),
		default="second_last",
		help="Which matching table to use. Default: %(default)s",
	)
	return parser.parse_args()


def read_outcar_lines(outcar_path: Path) -> list[str]:
	"""Read an OUTCAR file as text lines.

	Args:
		outcar_path: Path to the OUTCAR file.

	Returns:
		The full file as a list of lines.

	Raises:
		FileNotFoundError: If the file does not exist.
	"""
	return outcar_path.read_text(errors="ignore").splitlines()


def find_table_start_indices(lines: list[str]) -> list[int]:
	"""Locate all band-table headers in the OUTCAR.

	Args:
		lines: OUTCAR lines.

	Returns:
		Zero-based line indices where the header occurs.
	"""
	return [
		index
		for index, line in enumerate(lines)
		if BAND_TABLE_HEADER in line
	]


def choose_table_index(table_indices: list[int], selection: str) -> int:
	"""Choose which band table to analyze.

	Args:
		table_indices: Zero-based indices of all matching tables.
		selection: Requested selection strategy.

	Returns:
		The zero-based index into ``table_indices`` for the chosen table.

	Raises:
		ValueError: If no matching table exists.
	"""
	if not table_indices:
		raise ValueError(
			f"No '{BAND_TABLE_HEADER}' table was found in the OUTCAR."
		)

	if selection == "second_last" and len(table_indices) >= 2:
		return len(table_indices) - 2
	return len(table_indices) - 1


def parse_band_table(lines: list[str], header_line_index: int) -> list[BandRecord]:
	"""Parse band rows immediately following one header line.

	Args:
		lines: OUTCAR lines.
		header_line_index: Zero-based line index of the chosen header.

	Returns:
		The parsed band rows.

	Raises:
		ValueError: If no valid band rows are found after the header.
	"""
	records: list[BandRecord] = []

	for line in lines[header_line_index + 1 :]:
		match: re.Match[str] | None = BAND_LINE_PATTERN.match(line)
		if match is None:
			if records:
				break
			if line.strip() == "":
				continue
			break
		records.append(
			BandRecord(
				band_number=int(match.group(1)),
				energy=float(match.group(2)),
				occupation=float(match.group(3)),
			)
		)

	if not records:
		raise ValueError(
			"No band rows were parsed after the selected band-occupation header."
		)

	return records


def is_effectively_zero(value: float) -> bool:
	"""Check whether a floating-point value is effectively zero.

	Args:
		value: Value to classify.

	Returns:
		True if the value is within the configured zero tolerance.
	"""
	return math.isclose(value, 0.0, abs_tol=ZERO_TOLERANCE)


def is_effectively_integer(value: float) -> bool:
	"""Check whether a floating-point value is effectively an integer.

	Args:
		value: Value to classify.

	Returns:
		True if the value is within the configured integer tolerance.
	"""
	return math.isclose(value, round(value), abs_tol=INTEGER_TOLERANCE)


def summarize_records(
	records: list[BandRecord],
	table_occurrence_count: int,
	selected_table_index_1based: int,
	selected_table_line_1based: int,
) -> BandSummary:
	"""Build the occupancy summary for one band table.

	Args:
		records: Parsed band rows.
		table_occurrence_count: Number of matching tables in the OUTCAR.
		selected_table_index_1based: One-based index of the chosen table.
		selected_table_line_1based: One-based line number of the chosen header.

	Returns:
		The aggregate summary.
	"""
	full_occupation_bands: int = 0
	partial_occupation_bands: int = 0
	zero_occupation_bands: int = 0
	negative_energy_occupied_bands: int = 0
	non_negative_energy_occupied_bands: int = 0
	occupied_energies: list[float] = []

	for record in records:
		if is_effectively_zero(record.occupation):
			zero_occupation_bands += 1
			continue

		if is_effectively_integer(record.occupation):
			full_occupation_bands += 1
		else:
			partial_occupation_bands += 1
		occupied_energies.append(record.energy)

		if record.energy < -ZERO_TOLERANCE:
			negative_energy_occupied_bands += 1
		else:
			non_negative_energy_occupied_bands += 1

	nonzero_occupation_bands: int = full_occupation_bands + partial_occupation_bands
	min_occupied_energy: float | None = None
	max_occupied_energy: float | None = None
	if occupied_energies:
		min_occupied_energy = min(occupied_energies)
		max_occupied_energy = max(occupied_energies)

	return BandSummary(
		table_occurrence_count=table_occurrence_count,
		selected_table_index_1based=selected_table_index_1based,
		selected_table_line_1based=selected_table_line_1based,
		total_bands=records[-1].band_number,
		full_occupation_bands=full_occupation_bands,
		partial_occupation_bands=partial_occupation_bands,
		zero_occupation_bands=zero_occupation_bands,
		nonzero_occupation_bands=nonzero_occupation_bands,
		negative_energy_occupied_bands=negative_energy_occupied_bands,
		non_negative_energy_occupied_bands=non_negative_energy_occupied_bands,
		min_occupied_energy=min_occupied_energy,
		max_occupied_energy=max_occupied_energy,
		flag_no_nonzero_occupied_bands=(nonzero_occupation_bands == 0),
	)


def write_summary(summary: BandSummary, output_path: Path) -> None:
	"""Write the band summary as key-value pairs.

	Args:
		summary: Summary to serialize.
		output_path: Destination file path.
	"""
	lines: list[str] = [
		f"table_occurrence_count={summary.table_occurrence_count}",
		f"selected_table_index_1based={summary.selected_table_index_1based}",
		f"selected_table_line_1based={summary.selected_table_line_1based}",
		f"total_bands={summary.total_bands}",
		f"full_occupation_bands={summary.full_occupation_bands}",
		f"partial_occupation_bands={summary.partial_occupation_bands}",
		f"zero_occupation_bands={summary.zero_occupation_bands}",
		f"nonzero_occupation_bands={summary.nonzero_occupation_bands}",
		f"negative_energy_occupied_bands={summary.negative_energy_occupied_bands}",
		f"non_negative_energy_occupied_bands={summary.non_negative_energy_occupied_bands}",
		(
			"min_occupied_energy="
			f"{'nan' if summary.min_occupied_energy is None else f'{summary.min_occupied_energy:.10f}'}"
		),
		(
			"max_occupied_energy="
			f"{'nan' if summary.max_occupied_energy is None else f'{summary.max_occupied_energy:.10f}'}"
		),
		(
			"flag_no_nonzero_occupied_bands="
			f"{'yes' if summary.flag_no_nonzero_occupied_bands else 'no'}"
		),
	]
	output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	"""Run the OUTCAR band-occupation summarizer."""
	args: argparse.Namespace = parse_args()
	lines: list[str] = read_outcar_lines(args.outcar)
	table_indices: list[int] = find_table_start_indices(lines)
	selection_index: int = choose_table_index(table_indices, args.selection)
	header_line_index: int = table_indices[selection_index]
	records: list[BandRecord] = parse_band_table(lines, header_line_index)
	summary: BandSummary = summarize_records(
		records=records,
		table_occurrence_count=len(table_indices),
		selected_table_index_1based=selection_index + 1,
		selected_table_line_1based=header_line_index + 1,
	)
	write_summary(summary, args.output)


if __name__ == "__main__":
	main()
