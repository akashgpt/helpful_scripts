#!/usr/bin/env python3
"""Summarize VASP MLFF training storage and basis usage from ML_LOGFILE."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SpeciesBasisUsage:
	"""Basis usage for one chemical species."""

	symbol: str
	before_sparsification: int
	after_sparsification: int
	lconf_old: int | None = None
	lconf_new_pre_sparsification: int | None = None


@dataclass(frozen=True)
class MlffSummary:
	"""Compact MLFF training summary parsed from VASP outputs."""

	total_memory_mb: float | None
	memory_components_mb: dict[str, float]
	ml_mb: int | None
	ml_mconf: int | None
	ml_mconf_new: int | None
	sprsc_step: int | None
	stored_structures_before_sparsification: int | None
	stored_structures_after_sparsification: int | None
	species_basis: list[SpeciesBasisUsage]


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Summarize ML_MCONF and ML_MB usage from VASP ML_LOGFILE/OUTCAR.",
	)
	parser.add_argument("--ml-logfile", default="ML_LOGFILE", help="Path to ML_LOGFILE.")
	parser.add_argument("--outcar", default="OUTCAR", help="Optional OUTCAR path for run-recorded tag values.")
	parser.add_argument("--incar", default="INCAR", help="Optional INCAR path for fallback tag values.")
	parser.add_argument("--output", required=True, help="Output summary path.")
	return parser.parse_args()


def parse_int_tag_from_incar(incar_path: Path, tag_name: str) -> int | None:
	"""Return an integer INCAR tag value if present.

	Args:
		incar_path: INCAR file path.
		tag_name: Tag to read, e.g. ``ML_MB``.

	Returns:
		Integer tag value, or ``None`` if the tag is absent/unparseable.
	"""
	if not incar_path.is_file():
		return None
	pattern = re.compile(rf"^\s*{re.escape(tag_name)}\s*=?\s*([0-9]+)", re.IGNORECASE)
	for line in incar_path.read_text(errors="ignore").splitlines():
		match = pattern.search(line)
		if match:
			return int(match.group(1))
	return None


def parse_int_tag_from_outcar(outcar_path: Path, tag_name: str) -> int | None:
	"""Return an integer OUTCAR tag value if present.

	OUTCAR records the values actually used by the run, which is safer than
	reading a possibly edited INCAR after the job started.

	Args:
		outcar_path: OUTCAR file path.
		tag_name: Tag to read, e.g. ``ML_MB``.

	Returns:
		Integer tag value, or ``None`` if the tag is absent/unparseable.
	"""
	if not outcar_path.is_file():
		return None
	pattern = re.compile(rf"^\s*{re.escape(tag_name)}\s*=\s*([0-9]+)\b", re.IGNORECASE)
	for line in outcar_path.read_text(errors="ignore").splitlines():
		match = pattern.search(line)
		if match:
			return int(match.group(1))
	return None


def parse_int_setting(lines: Sequence[str], tag_name: str) -> int | None:
	"""Parse a VASP ML setting line from ML_LOGFILE."""
	pattern = re.compile(rf":\s*([0-9]+)\s*(?:\([^)]*\))?\s*{re.escape(tag_name)}\b")
	for line in lines:
		match = pattern.search(line)
		if match:
			return int(match.group(1))
	return None


def parse_memory_components(lines: Sequence[str]) -> tuple[float | None, dict[str, float]]:
	"""Parse the MLFF memory estimate section from ML_LOGFILE."""
	total_memory_mb: float | None = None
	components: dict[str, float] = {}
	component_pattern = re.compile(r"^\s*\|--\s*(.+?)\s*:\s*([0-9.]+)\s*$")
	total_pattern = re.compile(r"Total memory consumption\s*:\s*([0-9.]+)")
	for line in lines:
		component_match = component_pattern.search(line)
		if component_match:
			component_name = re.sub(r"\s+", "_", component_match.group(1).strip().lower())
			components[component_name] = float(component_match.group(2))
			continue
		total_match = total_pattern.search(line)
		if total_match:
			total_memory_mb = float(total_match.group(1))
	return total_memory_mb, components


def latest_line_with_prefix(lines: Sequence[str], prefix: str) -> str | None:
	"""Return the latest non-comment line with a given prefix."""
	selected_line: str | None = None
	for line in lines:
		if line.startswith(prefix):
			selected_line = line
	return selected_line


def parse_lconf_line(line: str | None) -> dict[str, tuple[int, int]]:
	"""Parse the latest LCONF line into old/new local-reference counts.

	Returns:
		Mapping from species symbol to ``(old, new_pre_sparsification)``.
	"""
	if not line:
		return {}
	tokens = line.split()
	values: dict[str, tuple[int, int]] = {}
	index = 2
	while index + 2 < len(tokens):
		symbol = tokens[index]
		try:
			old_count = int(tokens[index + 1])
			new_count = int(tokens[index + 2])
		except ValueError:
			break
		values[symbol] = (old_count, new_count)
		index += 3
	return values


def parse_sprsc_line(line: str | None, lconf_values: dict[str, tuple[int, int]]) -> tuple[int | None, int | None, int | None, list[SpeciesBasisUsage]]:
	"""Parse the latest SPRSC line.

	Args:
		line: Latest ``SPRSC`` line.
		lconf_values: Optional LCONF old/new counts for the same/latest update.

	Returns:
		``(step, nstr_prev, nstr_spar, species_basis)``.
	"""
	if not line:
		return None, None, None, []
	tokens = line.split()
	if len(tokens) < 4:
		return None, None, None, []
	try:
		step = int(tokens[1])
		nstr_prev = int(tokens[2])
		nstr_spar = int(tokens[3])
	except ValueError:
		return None, None, None, []

	species_basis: list[SpeciesBasisUsage] = []
	index = 4
	while index + 2 < len(tokens):
		symbol = tokens[index]
		try:
			before_sparsification = int(tokens[index + 1])
			after_sparsification = int(tokens[index + 2])
		except ValueError:
			break
		lconf_old: int | None = None
		lconf_new: int | None = None
		if symbol in lconf_values:
			lconf_old, lconf_new = lconf_values[symbol]
		species_basis.append(
			SpeciesBasisUsage(
				symbol=symbol,
				before_sparsification=before_sparsification,
				after_sparsification=after_sparsification,
				lconf_old=lconf_old,
				lconf_new_pre_sparsification=lconf_new,
			),
		)
		index += 3
	return step, nstr_prev, nstr_spar, species_basis


def parse_mlff_summary(ml_logfile_path: Path, outcar_path: Path, incar_path: Path) -> MlffSummary:
	"""Parse the MLFF training summary from available files."""
	lines = ml_logfile_path.read_text(errors="ignore").splitlines() if ml_logfile_path.is_file() else []
	total_memory_mb, memory_components_mb = parse_memory_components(lines)
	ml_mb = parse_int_tag_from_outcar(outcar_path, "ML_MB") or parse_int_setting(lines, "ML_MB") or parse_int_tag_from_incar(incar_path, "ML_MB")
	ml_mconf = parse_int_tag_from_outcar(outcar_path, "ML_MCONF") or parse_int_setting(lines, "ML_MCONF") or parse_int_tag_from_incar(incar_path, "ML_MCONF")
	ml_mconf_new = parse_int_tag_from_outcar(outcar_path, "ML_MCONF_NEW") or parse_int_setting(lines, "ML_MCONF_NEW") or parse_int_tag_from_incar(incar_path, "ML_MCONF_NEW")

	lconf_values = parse_lconf_line(latest_line_with_prefix(lines, "LCONF"))
	sprsc_step, nstr_prev, nstr_spar, species_basis = parse_sprsc_line(
		latest_line_with_prefix(lines, "SPRSC"),
		lconf_values,
	)

	return MlffSummary(
		total_memory_mb=total_memory_mb,
		memory_components_mb=memory_components_mb,
		ml_mb=ml_mb,
		ml_mconf=ml_mconf,
		ml_mconf_new=ml_mconf_new,
		sprsc_step=sprsc_step,
		stored_structures_before_sparsification=nstr_prev,
		stored_structures_after_sparsification=nstr_spar,
		species_basis=species_basis,
	)


def format_fraction(numerator: int | None, denominator: int | None) -> str:
	"""Format numerator/denominator as a percentage string."""
	if numerator is None or denominator in (None, 0):
		return "NA"
	return f"{100.0 * numerator / denominator:.2f}"


def write_summary(summary: MlffSummary, output_path: Path) -> None:
	"""Write a key-value MLFF training summary."""
	lines: list[str] = [
		"# MLFF training and memory summary",
		"# stored_structures_after_sparsification is SPRSC nstr_spar; compare with ML_MCONF.",
		"# basis_after_sparsification is SPRSC nlrc_spar; compare per species with ML_MB.",
		f"mlff_memory_total_mb={summary.total_memory_mb if summary.total_memory_mb is not None else 'NA'}",
		f"mlff_memory_total_gb={summary.total_memory_mb / 1024.0:.2f}" if summary.total_memory_mb is not None else "mlff_memory_total_gb=NA",
		f"mlff_ml_mb={summary.ml_mb if summary.ml_mb is not None else 'NA'}",
		f"mlff_ml_mconf={summary.ml_mconf if summary.ml_mconf is not None else 'NA'}",
		f"mlff_ml_mconf_new={summary.ml_mconf_new if summary.ml_mconf_new is not None else 'NA'}",
		f"mlff_sprsc_step={summary.sprsc_step if summary.sprsc_step is not None else 'NA'}",
		f"mlff_stored_structures_before_sparsification={summary.stored_structures_before_sparsification if summary.stored_structures_before_sparsification is not None else 'NA'}",
		f"mlff_stored_structures_after_sparsification={summary.stored_structures_after_sparsification if summary.stored_structures_after_sparsification is not None else 'NA'}",
		f"mlff_stored_structures_percent_of_ml_mconf={format_fraction(summary.stored_structures_after_sparsification, summary.ml_mconf)}",
	]
	for name, value in sorted(summary.memory_components_mb.items()):
		lines.append(f"mlff_memory_component_{name}_mb={value}")
	for basis in summary.species_basis:
		prefix = f"mlff_basis_{basis.symbol}"
		lines.extend(
			[
				f"{prefix}_before_sparsification={basis.before_sparsification}",
				f"{prefix}_after_sparsification={basis.after_sparsification}",
				f"{prefix}_percent_of_ml_mb={format_fraction(basis.after_sparsification, summary.ml_mb)}",
				f"{prefix}_lconf_old={basis.lconf_old if basis.lconf_old is not None else 'NA'}",
				f"{prefix}_lconf_new_pre_sparsification={basis.lconf_new_pre_sparsification if basis.lconf_new_pre_sparsification is not None else 'NA'}",
			],
		)
	output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	"""Run the MLFF summary extraction."""
	args = parse_args()
	summary = parse_mlff_summary(Path(args.ml_logfile), Path(args.outcar), Path(args.incar))
	write_summary(summary, Path(args.output))


if __name__ == "__main__":
	main()
