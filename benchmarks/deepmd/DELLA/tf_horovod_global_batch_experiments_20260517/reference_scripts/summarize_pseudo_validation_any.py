#!/usr/bin/env python3
"""Summarize DeePMD pseudo-validation logs for global-batch experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path(
	"/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/"
	"sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/"
	"tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517"
)
METRIC_PATTERN = re.compile(r"DEEPMD INFO\s+([A-Za-z ]+(?:/Natoms)?)\s+:\s+([0-9.eE+-]+)")
NUMBER_PATTERN = re.compile(r"number of test data\s+:\s+([0-9]+)")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_PERCENTILES = {
	"p0p135": 0.0013498980,
	"p15p865": 0.1586552539,
	"p50": 0.5,
	"p84p135": 0.8413447461,
	"p99p865": 0.9986501020,
}


@dataclass(frozen=True)
class CaseInfo:
	"""Experiment metadata for one trained model."""

	case_id: str
	group: str
	status: str
	gpus: int
	steps: int
	scale_by_worker: str


@dataclass(frozen=True)
class SystemInfo:
	"""Pseudo-validation system metadata."""

	dataset_id: str
	system_id: str
	split: str
	system_dir: str


@dataclass(frozen=True)
class TestMetrics:
	"""Parsed metrics from one ``dp test`` log.

	Units follow DeePMD's log output: energy is ``Energy RMSE/Natoms`` in
	eV/atom, force is ``Force RMSE`` in eV/A, and virial is
	``Virial RMSE/Natoms`` in eV per atom. The virial value is not converted to
	stress in GPa.
	"""

	nframes: int
	energy_rmse_per_atom: float
	force_rmse: float
	virial_rmse_per_atom: float


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Summarize pseudo-validation dp test logs.")
	parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Global-batch experiment root.")
	parser.add_argument(
		"--matrix",
		type=Path,
		default=None,
		help="Experiment matrix TSV. Defaults to <root>/EXPERIMENT_MATRIX.tsv.",
	)
	parser.add_argument(
		"--validation-root",
		type=Path,
		default=None,
		help="Pseudo-validation root. Defaults to <root>/pseudo_validation_20260517.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output TSV path. Defaults to <root>/pseudo_validation_20260517/PSEUDO_VALIDATION_SUMMARY.tsv.",
	)
	return parser.parse_args()


def read_cases(matrix_path: Path) -> dict[str, CaseInfo]:
	"""Read experiment matrix metadata.

	Args:
		matrix_path: Path to ``EXPERIMENT_MATRIX.tsv``.

	Returns:
		Mapping from case id to metadata.
	"""
	cases: dict[str, CaseInfo] = {}
	with matrix_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		for row in reader:
			cases[row["case_id"]] = CaseInfo(
				case_id=row["case_id"],
				group=row["group"],
				status=row["status"],
				gpus=int(row["gpus"]),
				steps=int(row["steps"]),
				scale_by_worker=row["scale_by_worker"],
			)
	return cases


def read_systems(systems_path: Path) -> dict[str, SystemInfo]:
	"""Read pseudo-validation system metadata.

	Args:
		systems_path: Path to ``PSEUDO_VALIDATION_SYSTEMS.tsv``.

	Returns:
		Mapping from system id to metadata.
	"""
	systems: dict[str, SystemInfo] = {}
	with systems_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		for row in reader:
			systems[row["system_id"]] = SystemInfo(
				dataset_id=row["dataset_id"],
				system_id=row["system_id"],
				split=row["split"],
				system_dir=row["system_dir"],
			)
	return systems


def parse_log(log_path: Path) -> TestMetrics | None:
	"""Parse a DeePMD ``dp test`` log.

	Args:
		log_path: Path to ``log.dp_test``.

	Returns:
		Parsed metrics, or ``None`` when the log is incomplete.
	"""
	if not log_path.exists():
		return None
	text = log_path.read_text(encoding="utf-8", errors="replace")
	number_matches = NUMBER_PATTERN.findall(text)
	if not number_matches:
		return None
	metrics: dict[str, float] = {}
	for name, value in METRIC_PATTERN.findall(text):
		metrics[" ".join(name.split())] = float(value)
	required = ["Energy RMSE/Natoms", "Force RMSE", "Virial RMSE/Natoms"]
	if any(name not in metrics for name in required):
		return None
	return TestMetrics(
		nframes=int(number_matches[-1]),
		energy_rmse_per_atom=metrics["Energy RMSE/Natoms"],
		force_rmse=metrics["Force RMSE"],
		virial_rmse_per_atom=metrics["Virial RMSE/Natoms"],
	)


def weighted_mean(values: list[tuple[float, int]]) -> float:
	"""Calculate a frame-weighted mean.

	Args:
		values: ``(value, weight)`` pairs.

	Returns:
		Weighted mean, or ``nan`` if no positive weights exist.
	"""
	total_weight = sum(weight for _, weight in values)
	if total_weight <= 0:
		return math.nan
	return sum(value * weight for value, weight in values) / total_weight


def quantile(sorted_values: list[float], probability: float) -> float:
	"""Calculate a linearly interpolated quantile.

	Args:
		sorted_values: Values sorted in ascending order.
		probability: Quantile probability between 0 and 1.

	Returns:
		Interpolated quantile, or ``nan`` for an empty list.
	"""
	if not sorted_values:
		return math.nan
	if len(sorted_values) == 1:
		return sorted_values[0]
	position = probability * (len(sorted_values) - 1)
	lower_index = math.floor(position)
	upper_index = math.ceil(position)
	if lower_index == upper_index:
		return sorted_values[lower_index]
	lower_value = sorted_values[lower_index]
	upper_value = sorted_values[upper_index]
	fraction = position - lower_index
	return lower_value + (upper_value - lower_value) * fraction


def stable_seed(parts: tuple[str, ...]) -> int:
	"""Create a deterministic random seed from text labels.

	Args:
		parts: Labels that define one bootstrap problem.

	Returns:
		Integer seed.
	"""
	digest = hashlib.sha256("\t".join(parts).encode("utf-8")).hexdigest()
	return int(digest[:16], 16)


def bootstrap_weighted_mean_percentiles(
	values: list[tuple[float, int]],
	seed_parts: tuple[str, ...],
	samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float]:
	"""Estimate bootstrap percentiles for a weighted mean.

	The bootstrap resamples validation systems with replacement and keeps each
	system's frame count as its weight. This measures system-to-system spread in
	the pseudo-validation set while preserving the frame-weighted point estimate.

	Args:
		values: ``(value, nframes)`` pairs for one aggregate metric.
		seed_parts: Labels used to seed the deterministic bootstrap.
		samples: Number of bootstrap resamples.

	Returns:
		Dictionary with the point estimate and bootstrap percentile values.
	"""
	clean_values = [(value, weight) for value, weight in values if weight > 0 and math.isfinite(value)]
	point = weighted_mean(clean_values)
	if not clean_values or not math.isfinite(point):
		return {"point": math.nan, **{name: math.nan for name in BOOTSTRAP_PERCENTILES}}
	if len(clean_values) == 1:
		return {"point": point, **{name: point for name in BOOTSTRAP_PERCENTILES}}
	random_number_generator = random.Random(stable_seed(seed_parts))
	bootstrap_values: list[float] = []
	item_count = len(clean_values)
	for _ in range(samples):
		resampled = [clean_values[random_number_generator.randrange(item_count)] for _ in range(item_count)]
		bootstrap_values.append(weighted_mean(resampled))
	bootstrap_values.sort()
	return {"point": point, **{name: quantile(bootstrap_values, probability) for name, probability in BOOTSTRAP_PERCENTILES.items()}}


def format_float(value: float) -> str:
	"""Format a floating-point value for TSV output.

	Args:
		value: Number to format.

	Returns:
		Formatted number or ``NA``.
	"""
	if not math.isfinite(value):
		return "NA"
	return f"{value:.8g}"


def empty_percentile_columns() -> dict[str, str]:
	"""Return empty percentile columns for non-aggregate rows.

	Returns:
		Mapping with all percentile columns set to ``NA``.
	"""
	columns: dict[str, str] = {}
	for metric_name in ["energy_rmse_per_atom", "force_rmse", "virial_rmse_per_atom"]:
		for percentile_name in BOOTSTRAP_PERCENTILES:
			columns[f"{metric_name}_{percentile_name}"] = "NA"
	return columns


def format_percentile_columns(metric_name: str, percentiles: dict[str, float]) -> dict[str, str]:
	"""Format percentile columns for one metric.

	Args:
		metric_name: Metric column prefix.
		percentiles: Bootstrap percentile dictionary.

	Returns:
		Formatted TSV columns.
	"""
	return {f"{metric_name}_{name}": format_float(percentiles[name]) for name in BOOTSTRAP_PERCENTILES}


def main() -> None:
	"""Write per-system and aggregate pseudo-validation summaries."""
	args = parse_args()
	root: Path = args.root
	validation_root = args.validation_root or root / "pseudo_validation_20260517"
	matrix_path = args.matrix or root / "EXPERIMENT_MATRIX.tsv"
	output_path = args.output or validation_root / "PSEUDO_VALIDATION_SUMMARY.tsv"
	cases = read_cases(matrix_path)
	systems = read_systems(validation_root / "PSEUDO_VALIDATION_SYSTEMS.tsv")
	rows: list[dict[str, str]] = []
	aggregate_inputs: dict[tuple[str, str, str], list[tuple[TestMetrics, int]]] = {}
	for case_id, case in cases.items():
		for system_id, system in systems.items():
			log_path = validation_root / "results" / case_id / system_id / "log.dp_test"
			metrics = parse_log(log_path)
			if metrics is None:
				rows.append(
					{
						"row_type": "system",
						"case_id": case_id,
						"group": case.group,
						"status": case.status,
						"gpus": str(case.gpus),
						"steps": str(case.steps),
						"scale_by_worker": case.scale_by_worker,
						"dataset_id": system.dataset_id,
						"split": system.split,
						"system_id": system_id,
						"nframes": "NA",
						"energy_rmse_per_atom": "NA",
						"force_rmse": "NA",
						"virial_rmse_per_atom": "NA",
						**empty_percentile_columns(),
					}
				)
				continue
			rows.append(
				{
					"row_type": "system",
					"case_id": case_id,
					"group": case.group,
					"status": case.status,
					"gpus": str(case.gpus),
					"steps": str(case.steps),
					"scale_by_worker": case.scale_by_worker,
					"dataset_id": system.dataset_id,
					"split": system.split,
					"system_id": system_id,
					"nframes": str(metrics.nframes),
					"energy_rmse_per_atom": f"{metrics.energy_rmse_per_atom:.8g}",
					"force_rmse": f"{metrics.force_rmse:.8g}",
					"virial_rmse_per_atom": f"{metrics.virial_rmse_per_atom:.8g}",
					**empty_percentile_columns(),
				}
			)
			aggregate_inputs.setdefault((case_id, system.dataset_id, system.split), []).append((metrics, metrics.nframes))
			aggregate_inputs.setdefault((case_id, system.dataset_id, "all"), []).append((metrics, metrics.nframes))
			aggregate_inputs.setdefault((case_id, "all", "all"), []).append((metrics, metrics.nframes))
	for (case_id, dataset_id, split), values in sorted(aggregate_inputs.items()):
		case = cases[case_id]
		energy_percentiles = bootstrap_weighted_mean_percentiles(
			[(item.energy_rmse_per_atom, weight) for item, weight in values],
			(case_id, dataset_id, split, "energy_rmse_per_atom"),
		)
		force_percentiles = bootstrap_weighted_mean_percentiles(
			[(item.force_rmse, weight) for item, weight in values],
			(case_id, dataset_id, split, "force_rmse"),
		)
		virial_percentiles = bootstrap_weighted_mean_percentiles(
			[(item.virial_rmse_per_atom, weight) for item, weight in values],
			(case_id, dataset_id, split, "virial_rmse_per_atom"),
		)
		rows.append(
			{
				"row_type": "aggregate",
				"case_id": case_id,
				"group": case.group,
				"status": case.status,
				"gpus": str(case.gpus),
				"steps": str(case.steps),
				"scale_by_worker": case.scale_by_worker,
				"dataset_id": dataset_id,
				"split": split,
				"system_id": "weighted_by_nframes",
				"nframes": str(sum(weight for _, weight in values)),
				"energy_rmse_per_atom": format_float(energy_percentiles["point"]),
				"force_rmse": format_float(force_percentiles["point"]),
				"virial_rmse_per_atom": format_float(virial_percentiles["point"]),
				**format_percentile_columns("energy_rmse_per_atom", energy_percentiles),
				**format_percentile_columns("force_rmse", force_percentiles),
				**format_percentile_columns("virial_rmse_per_atom", virial_percentiles),
			}
		)
	fieldnames = [
		"row_type",
		"case_id",
		"group",
		"status",
		"gpus",
		"steps",
		"scale_by_worker",
		"dataset_id",
		"split",
		"system_id",
		"nframes",
		"energy_rmse_per_atom",
		"force_rmse",
		"virial_rmse_per_atom",
		"energy_rmse_per_atom_p0p135",
		"energy_rmse_per_atom_p15p865",
		"energy_rmse_per_atom_p50",
		"energy_rmse_per_atom_p84p135",
		"energy_rmse_per_atom_p99p865",
		"force_rmse_p0p135",
		"force_rmse_p15p865",
		"force_rmse_p50",
		"force_rmse_p84p135",
		"force_rmse_p99p865",
		"virial_rmse_per_atom_p0p135",
		"virial_rmse_per_atom_p15p865",
		"virial_rmse_per_atom_p50",
		"virial_rmse_per_atom_p84p135",
		"virial_rmse_per_atom_p99p865",
	]
	with output_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)
	print(output_path)


if __name__ == "__main__":
	main()
