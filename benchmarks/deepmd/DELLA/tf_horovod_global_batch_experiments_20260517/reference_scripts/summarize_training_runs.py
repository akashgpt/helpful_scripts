#!/usr/bin/env python3
"""Summarize DeePMD lcurve and Slurm timing files for the experiment matrix."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path(
	"/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/"
	"sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/"
	"tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517"
)
WORKING_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = WORKING_ROOT / "EXPERIMENT_MATRIX.tsv"
TRAIN_TIME_PATTERN = re.compile(r"average training time:\s+([0-9.]+)\s+s/batch")
WALL_TIME_PATTERN = re.compile(r"wall time:\s+([0-9.]+)\s+s")


@dataclass(frozen=True)
class MatrixRow:
	"""One row from the experiment matrix."""

	case_id: str
	group: str
	status: str
	gpus: int
	nodes: int
	steps: int
	decay_steps: int
	scale_by_worker: str
	source_or_output: str


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Summarize completed DeePMD runs.")
	parser.add_argument(
		"--output-root",
		type=Path,
		default=DEFAULT_OUTPUT_ROOT,
		help="Generated experiment root containing new runs and matrix.",
	)
	parser.add_argument(
		"--matrix",
		type=Path,
		default=MATRIX_PATH,
		help="Experiment matrix TSV.",
	)
	return parser.parse_args()


def read_matrix(matrix_path: Path) -> list[MatrixRow]:
	"""Read matrix rows from TSV.

	Args:
		matrix_path: Matrix TSV path.

	Returns:
		Matrix rows.
	"""
	with matrix_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		return [
			MatrixRow(
				case_id=str(row["case_id"]),
				group=str(row["group"]),
				status=str(row["status"]),
				gpus=int(row["gpus"]),
				nodes=int(row["nodes"]),
				steps=int(row["steps"]),
				decay_steps=int(row["decay_steps"]),
				scale_by_worker=str(row["scale_by_worker"]),
				source_or_output=str(row["source_or_output"]),
			)
			for row in reader
		]


def resolve_run_dir(row: MatrixRow, output_root: Path) -> Path:
	"""Resolve a matrix row to a run directory.

	Args:
		row: Matrix row.
		output_root: Generated scratch root.

	Returns:
		Absolute run directory.
	"""
	path = Path(row.source_or_output)
	if path.is_absolute():
		return path
	return output_root / path


def parse_lcurve(lcurve_path: Path) -> tuple[str, str, str, str, str]:
	"""Parse final and late-window force metrics from lcurve.out.

	Args:
		lcurve_path: DeePMD lcurve path.

	Returns:
		Final step, final total RMSE, final force RMSE, late mean force RMSE,
		and late best force RMSE as strings.
	"""
	if not lcurve_path.exists():
		return ("NA", "NA", "NA", "NA", "NA")
	rows: list[tuple[int, float, float]] = []
	with lcurve_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			stripped = line.strip()
			if not stripped or stripped.startswith("#"):
				continue
			parts = stripped.split()
			if len(parts) < 4:
				continue
			rows.append((int(parts[0]), float(parts[1]), float(parts[3])))
	if not rows:
		return ("NA", "NA", "NA", "NA", "NA")
	final_step, final_rmse, final_force = rows[-1]
	late_rows = [row for row in rows if row[0] >= max(0, final_step - 1000)]
	late_force_values = [row[2] for row in late_rows]
	late_mean_force = sum(late_force_values) / len(late_force_values)
	late_best_force = min(late_force_values)
	return (
		str(final_step),
		f"{final_rmse:.6g}",
		f"{final_force:.6g}",
		f"{late_mean_force:.6g}",
		f"{late_best_force:.6g}",
	)


def parse_slurm(run_dir: Path) -> tuple[str, str]:
	"""Parse average train time and wall time from Slurm logs.

	Args:
		run_dir: Run directory.

	Returns:
		Average train time and rank-0 wall time as strings.
	"""
	slurm_logs = sorted(run_dir.glob("slurm-*.out"))
	if not slurm_logs:
		return ("NA", "NA")
	text = slurm_logs[-1].read_text(encoding="utf-8", errors="replace")
	train_times = TRAIN_TIME_PATTERN.findall(text)
	wall_times = WALL_TIME_PATTERN.findall(text)
	train_time = train_times[-1] if train_times else "NA"
	wall_time = wall_times[-1] if wall_times else "NA"
	return (train_time, wall_time)


def main() -> None:
	"""Print a TSV summary for all matrix rows."""
	args = parse_args()
	rows = read_matrix(args.matrix)
	header = [
		"case_id",
		"group",
		"status",
		"gpus",
		"steps",
		"scale_by_worker",
		"final_step",
		"final_rmse",
		"final_rmse_f",
		"late_mean_rmse_f",
		"late_best_rmse_f",
		"avg_train_s_per_batch",
		"wall_time_s",
		"run_dir",
	]
	print("\t".join(header))
	for row in rows:
		run_dir = resolve_run_dir(row, args.output_root)
		final_step, final_rmse, final_force, late_mean_force, late_best_force = parse_lcurve(run_dir / "lcurve.out")
		train_time, wall_time = parse_slurm(run_dir)
		values = [
			row.case_id,
			row.group,
			row.status,
			str(row.gpus),
			str(row.steps),
			row.scale_by_worker,
			final_step,
			final_rmse,
			final_force,
			late_mean_force,
			late_best_force,
			train_time,
			wall_time,
			str(run_dir),
		]
		print("\t".join(values))


if __name__ == "__main__":
	main()
