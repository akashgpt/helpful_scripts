#!/usr/bin/env python3
"""Format the curated DeePMD validation comparison table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TextIO


DEFAULT_TSV = Path(__file__).with_name(
	"VALIDATION_RMSE_COMPARISON__71MgSiO3_5He__20260518.tsv"
)

DISPLAY_COLUMNS = [
	("model", "model"),
	("validation_mode", "mode"),
	("parameter_count", "params"),
	("training_gpu_time", "GPU train time"),
	("energy_rmse_per_atom", "val E RMSE (eV/atom)"),
	("force_rmse", "val F RMSE (eV/A)"),
	("virial_rmse_per_atom", "val V RMSE (eV/atom)"),
	("train_total_rmse", "train total RMSE"),
	("train_e_rmse_per_atom", "train E RMSE (eV/atom)"),
	("train_f_rmse", "train F RMSE (eV/A)"),
	("train_v_rmse_per_atom", "train V RMSE (eV/atom)"),
]

FLOAT_COLUMNS = {
	"energy_rmse_per_atom",
	"force_rmse",
	"virial_rmse_per_atom",
	"train_total_rmse",
	"train_e_rmse_per_atom",
	"train_f_rmse",
	"train_v_rmse_per_atom",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments.

	Args:
		argv: Optional argument list. Uses ``sys.argv`` when omitted.

	Returns:
		Parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Print the compact validation/training RMSE comparison table."
	)
	parser.add_argument(
		"input_tsv",
		nargs="?",
		type=Path,
		default=DEFAULT_TSV,
		help=f"Curated validation TSV. Default: {DEFAULT_TSV.name}",
	)
	parser.add_argument(
		"--format",
		choices=("markdown", "tsv"),
		default="markdown",
		help="Output format. Use 'tsv' for the exact tab-separated header.",
	)
	parser.add_argument(
		"--digits",
		type=int,
		default=5,
		help="Digits after the decimal for RMSE values.",
	)
	return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
	"""Read the curated validation comparison TSV.

	Args:
		path: Path to the input TSV file.

	Returns:
		A list of row dictionaries.
	"""
	with path.open("r", encoding="utf-8", newline="") as handle:
		return list(csv.DictReader(handle, delimiter="\t"))


def display_mode(value: str) -> str:
	"""Return a compact display label for the validation mode.

	Args:
		value: Raw ``validation_mode`` value from the TSV.

	Returns:
		Human-readable mode label.
	"""
	if value == "noncompressed_reference":
		return "noncompressed ref"
	return value


def format_int(value: str) -> str:
	"""Format an integer field with thousands separators.

	Args:
		value: Raw integer text.

	Returns:
		Formatted integer text.
	"""
	return f"{int(value):,}"


def format_float(value: str, digits: int) -> str:
	"""Format an RMSE field with a fixed number of decimal places.

	Args:
		value: Raw float text.
		digits: Number of digits after the decimal point.

	Returns:
		Formatted float text.
	"""
	return f"{float(value):.{digits}f}"


def format_cell(row: dict[str, str], key: str, digits: int) -> str:
	"""Format one table cell.

	Args:
		row: Source row from the input TSV.
		key: Source column name.
		digits: Number of digits after the decimal point for RMSE fields.

	Returns:
		Formatted cell text.
	"""
	value = row[key]
	if key == "validation_mode":
		return display_mode(value)
	if key == "parameter_count":
		return format_int(value)
	if key in FLOAT_COLUMNS:
		return format_float(value, digits)
	return value


def build_table(rows: list[dict[str, str]], digits: int) -> tuple[list[str], list[list[str]]]:
	"""Build display headers and rows.

	Args:
		rows: Source rows from the curated TSV.
		digits: Number of digits after the decimal point for RMSE fields.

	Returns:
		Display headers and formatted row values.
	"""
	headers = [label for _, label in DISPLAY_COLUMNS]
	display_rows = [
		[format_cell(row, key, digits) for key, _ in DISPLAY_COLUMNS]
		for row in rows
	]
	return headers, display_rows


def write_markdown(headers: list[str], rows: list[list[str]], output: TextIO) -> None:
	"""Write a GitHub-flavored Markdown table.

	Args:
		headers: Display column headers.
		rows: Display row values.
		output: Output stream.
	"""
	print("| " + " | ".join(headers) + " |", file=output)
	print("|" + "|".join(["---"] * len(headers)) + "|", file=output)
	for row in rows:
		print("| " + " | ".join(row) + " |", file=output)


def write_tsv(headers: list[str], rows: list[list[str]], output: TextIO) -> None:
	"""Write a tab-separated table.

	Args:
		headers: Display column headers.
		rows: Display row values.
		output: Output stream.
	"""
	writer = csv.writer(output, delimiter="\t", lineterminator="\n")
	writer.writerow(headers)
	writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
	"""Run the table formatter.

	Args:
		argv: Optional argument list. Uses ``sys.argv`` when omitted.

	Returns:
		Process exit code.
	"""
	args = parse_args(argv)
	rows = read_rows(args.input_tsv)
	headers, display_rows = build_table(rows, args.digits)
	if args.format == "markdown":
		write_markdown(headers, display_rows, sys.stdout)
	else:
		write_tsv(headers, display_rows, sys.stdout)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
