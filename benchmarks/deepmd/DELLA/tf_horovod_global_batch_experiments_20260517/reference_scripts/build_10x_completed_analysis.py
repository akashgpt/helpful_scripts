#!/usr/bin/env python3
"""Build completed-case manifests and training summaries for 10x DeePMD runs."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(
	"/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/"
	"sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/"
	"tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517"
)
MATRIX_10X = ROOT / "EXPERIMENT_MATRIX_10x_steps.tsv"
MATRIX_CONTINUATION = ROOT / "EXPERIMENT_MATRIX_10x_continuations.tsv"
COMPLETED_MATRIX = ROOT / "EXPERIMENT_MATRIX_10x_COMPLETED.tsv"
PSEUDOVAL_MATRIX = ROOT / "EXPERIMENT_MATRIX_10x_COMPLETED_FOR_PSEUDOVAL.tsv"
TRAINING_SUMMARY = ROOT / "TRAINING_SUMMARY_10x_COMPLETED.tsv"
TRAIN_TIME_PATTERN = re.compile(r"average training time:\s+([0-9.]+)\s+s/batch")
WALL_TIME_PATTERN = re.compile(r"wall time:\s+([0-9.]+)\s+s")
SKIP_CASES = {"reuse_16gpu_10k_10x"}


@dataclass(frozen=True)
class Case:
	"""One completed 10x analysis case."""

	case_id: str
	source_case_id: str
	group: str
	status: str
	gpus: int
	nodes: int
	steps: int
	decay_steps: int
	scale_by_worker: str
	run_dir: Path
	note: str


def read_rows(path: Path) -> list[dict[str, str]]:
	"""Read a TSV file into row dictionaries.

	Args:
		path: TSV path.

	Returns:
		Rows from the TSV, or an empty list if the file is absent.
	"""
	if not path.exists():
		return []
	with path.open("r", encoding="utf-8", newline="") as handle:
		return list(csv.DictReader(handle, delimiter="\t"))


def resolve_run_dir(value: str) -> Path:
	"""Resolve a run directory from a matrix value.

	Args:
		value: Relative or absolute run directory.

	Returns:
		Absolute run directory.
	"""
	path = Path(value)
	if path.is_absolute():
		return path
	return ROOT / path


def build_cases() -> list[Case]:
	"""Build the completed 10x case list.

	Returns:
		Completed cases with existing ``lcurve.out`` files.
	"""
	cases: list[Case] = []
	for row in read_rows(MATRIX_10X):
		case_id = row["case_id"]
		if case_id in SKIP_CASES:
			continue
		run_dir = resolve_run_dir(row["run_dir"])
		if not (run_dir / "lcurve.out").exists():
			continue
		cases.append(
			Case(
				case_id=case_id,
				source_case_id=row["source_case_id"],
				group=row["group"],
				status="completed",
				gpus=int(row["gpus"]),
				nodes=int(row["nodes"]),
				steps=int(row["steps"]),
				decay_steps=int(row["decay_steps"]),
				scale_by_worker=row["scale_by_worker"],
				run_dir=run_dir,
				note=row.get("note", ""),
			)
		)
	for row in read_rows(MATRIX_CONTINUATION):
		run_dir = resolve_run_dir(row["run_dir"])
		if not (run_dir / "lcurve.out").exists():
			continue
		cases.append(
			Case(
				case_id=row["case_id"],
				source_case_id=row["source_case_id"],
				group=row["group"],
				status="completed_restart",
				gpus=int(row["gpus"]),
				nodes=int(row["nodes"]),
				steps=int(row["steps"]),
				decay_steps=int(row["decay_steps"]),
				scale_by_worker=row["scale_by_worker"],
				run_dir=run_dir,
				note=row.get("note", ""),
			)
		)
	return cases


def parse_lcurve(path: Path) -> tuple[str, str, str, str, str, str, str]:
	"""Parse final and late-window RMSE values from ``lcurve.out``.

	The parsed energy, force, and virial values are training-curve metrics from
	DeePMD, not pseudo-validation `dp test` metrics. Pseudo-validation units are
	documented in the archived pseudo-validation summarizers.

	Args:
		path: DeePMD learning-curve file.

	Returns:
		Final step, final total, energy, force, virial RMSE values, late mean
		force RMSE, and late best force RMSE.
	"""
	rows: list[tuple[int, float, float, float, float]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			stripped = line.strip()
			if not stripped or stripped.startswith("#"):
				continue
			parts = stripped.split()
			if len(parts) < 5:
				continue
			rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
	if not rows:
		return ("NA", "NA", "NA", "NA", "NA", "NA", "NA")
	final_step, final_total, final_energy, final_force, final_virial = rows[-1]
	late_rows = [row for row in rows if row[0] >= max(0, final_step - 1000)]
	late_force = [row[3] for row in late_rows]
	return (
		str(final_step),
		f"{final_total:.8g}",
		f"{final_energy:.8g}",
		f"{final_force:.8g}",
		f"{final_virial:.8g}",
		f"{sum(late_force) / len(late_force):.8g}",
		f"{min(late_force):.8g}",
	)


def parse_slurm(run_dir: Path) -> tuple[str, str]:
	"""Parse average train and wall time from the latest Slurm log.

	Args:
		run_dir: Run directory.

	Returns:
		Average training seconds per batch and rank-0 wall time.
	"""
	logs = sorted(run_dir.glob("slurm-*.out"))
	if not logs:
		return ("NA", "NA")
	text = logs[-1].read_text(encoding="utf-8", errors="replace")
	train_times = TRAIN_TIME_PATTERN.findall(text)
	wall_times = WALL_TIME_PATTERN.findall(text)
	return (train_times[-1] if train_times else "NA", wall_times[-1] if wall_times else "NA")


def write_completed_matrix(cases: list[Case]) -> None:
	"""Write completed-case matrices for analysis and pseudo-validation.

	Args:
		cases: Completed cases.
	"""
	with COMPLETED_MATRIX.open("w", encoding="utf-8", newline="") as handle:
		fieldnames = [
			"case_id",
			"source_case_id",
			"group",
			"status",
			"gpus",
			"nodes",
			"steps",
			"decay_steps",
			"scale_by_worker",
			"run_dir",
			"note",
		]
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		for case in cases:
			writer.writerow(
				{
					"case_id": case.case_id,
					"source_case_id": case.source_case_id,
					"group": case.group,
					"status": case.status,
					"gpus": case.gpus,
					"nodes": case.nodes,
					"steps": case.steps,
					"decay_steps": case.decay_steps,
					"scale_by_worker": case.scale_by_worker,
					"run_dir": case.run_dir.relative_to(ROOT),
					"note": case.note,
				}
			)
	with PSEUDOVAL_MATRIX.open("w", encoding="utf-8", newline="") as handle:
		fieldnames = [
			"case_id",
			"group",
			"status",
			"gpus",
			"nodes",
			"steps",
			"decay_steps",
			"scale_by_worker",
			"source_or_output",
		]
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		for case in cases:
			writer.writerow(
				{
					"case_id": case.case_id,
					"group": case.group,
					"status": case.status,
					"gpus": case.gpus,
					"nodes": case.nodes,
					"steps": case.steps,
					"decay_steps": case.decay_steps,
					"scale_by_worker": case.scale_by_worker,
					"source_or_output": case.run_dir.relative_to(ROOT),
				}
			)


def write_training_summary(cases: list[Case]) -> None:
	"""Write training summary for completed cases.

	Args:
		cases: Completed cases.
	"""
	fieldnames = [
		"case_id",
		"group",
		"status",
		"gpus",
		"nodes",
		"steps",
		"scale_by_worker",
		"final_step",
		"final_rmse",
		"final_rmse_e",
		"final_rmse_f",
		"final_rmse_v",
		"late_mean_rmse_f",
		"late_best_rmse_f",
		"avg_train_s_per_batch",
		"wall_time_s",
		"gpu_seconds",
		"run_dir",
	]
	with TRAINING_SUMMARY.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		for case in cases:
			final_step, final_rmse, final_e, final_f, final_v, late_mean_f, late_best_f = parse_lcurve(case.run_dir / "lcurve.out")
			train_time, wall_time = parse_slurm(case.run_dir)
			try:
				gpu_seconds = float(wall_time) * case.gpus
				gpu_seconds_text = f"{gpu_seconds:.3f}"
			except ValueError:
				gpu_seconds_text = "NA"
			writer.writerow(
				{
					"case_id": case.case_id,
					"group": case.group,
					"status": case.status,
					"gpus": case.gpus,
					"nodes": case.nodes,
					"steps": case.steps,
					"scale_by_worker": case.scale_by_worker,
					"final_step": final_step,
					"final_rmse": final_rmse,
					"final_rmse_e": final_e,
					"final_rmse_f": final_f,
					"final_rmse_v": final_v,
					"late_mean_rmse_f": late_mean_f,
					"late_best_rmse_f": late_best_f,
					"avg_train_s_per_batch": train_time,
					"wall_time_s": wall_time,
					"gpu_seconds": gpu_seconds_text,
					"run_dir": case.run_dir,
				}
			)


def main() -> None:
	"""Build completed matrices and training summary."""
	cases = build_cases()
	write_completed_matrix(cases)
	write_training_summary(cases)
	print(f"completed_cases\t{len(cases)}")
	print(f"matrix\t{COMPLETED_MATRIX}")
	print(f"pseudoval_matrix\t{PSEUDOVAL_MATRIX}")
	print(f"training_summary\t{TRAINING_SUMMARY}")


if __name__ == "__main__":
	main()
