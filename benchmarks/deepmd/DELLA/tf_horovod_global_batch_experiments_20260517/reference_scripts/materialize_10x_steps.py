#!/usr/bin/env python3
"""Materialize 10x-step follow-up DeePMD global-batch experiments."""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(
	"/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/"
	"sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/"
	"tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517"
)
MATRIX_PATH = ROOT / "EXPERIMENT_MATRIX.tsv"
OUTPUT_GROUP = "long_steps_10x"
OUTPUT_RUN_ROOT = ROOT / "runs" / OUTPUT_GROUP
OUTPUT_MATRIX = ROOT / "EXPERIMENT_MATRIX_10x_steps.tsv"
SKIP_CASES = {"1gpu_20k_linear"}
TIME_LIMIT = "01:00:00"


@dataclass(frozen=True)
class Case:
	"""One source experiment case from the existing matrix."""

	case_id: str
	group: str
	status: str
	gpus: int
	nodes: int
	steps: int
	decay_steps: int
	scale_by_worker: str
	source_or_output: str


@dataclass(frozen=True)
class NewCase:
	"""One materialized 10x-step follow-up case."""

	source_case_id: str
	case_id: str
	group: str
	gpus: int
	nodes: int
	steps: int
	decay_steps: int
	scale_by_worker: str
	source_dir: Path
	run_dir: Path
	priority: str
	note: str


def read_cases(matrix_path: Path) -> list[Case]:
	"""Read the existing experiment matrix.

	Args:
		matrix_path: Existing matrix TSV.

	Returns:
		Existing cases in matrix order.
	"""
	with matrix_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		return [
			Case(
				case_id=row["case_id"],
				group=row["group"],
				status=row["status"],
				gpus=int(row["gpus"]),
				nodes=int(row["nodes"]),
				steps=int(row["steps"]),
				decay_steps=int(row["decay_steps"]),
				scale_by_worker=row["scale_by_worker"],
				source_or_output=row["source_or_output"],
			)
			for row in reader
		]


def resolve_source_dir(case: Case) -> Path:
	"""Resolve an existing case to its run directory.

	Args:
		case: Existing matrix case.

	Returns:
		Absolute source run directory.
	"""
	path = Path(case.source_or_output)
	if path.is_absolute():
		return path
	return ROOT / path


def choose_save_freq(steps: int) -> int:
	"""Choose a conservative checkpoint interval for longer jobs.

	Args:
		steps: Number of optimizer steps.

	Returns:
		Checkpoint interval.
	"""
	if steps <= 10000:
		return 1000
	if steps <= 50000:
		return 5000
	return 10000


def priority_for(case: Case) -> tuple[str, str]:
	"""Classify a 10x case by scientific value.

	Args:
		case: Existing matrix case.

	Returns:
		Priority label and short note.
	"""
	if case.scale_by_worker == "none":
		return ("diagnostic", "LR none was poor in the short sweep; keep only as a control.")
	if case.case_id == "2gpu_5k_linear":
		return ("diagnostic", "Short 2GPU/5k was an outlier failure; keep only to test whether longer training rescues it.")
	if case.case_id in {"reuse_4gpu_10k", "reuse_8gpu_10k", "8gpu_7k_linear", "reuse_1gpu_10k"}:
		return ("core", "Directly tests the current quality/cost conclusion.")
	return ("secondary", "Useful for trend confirmation.")


def load_json(path: Path) -> dict[str, Any]:
	"""Load JSON from disk.

	Args:
		path: JSON path.

	Returns:
		Parsed JSON dictionary.
	"""
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
	"""Write formatted JSON to disk.

	Args:
		path: Destination path.
		data: JSON-compatible dictionary.
	"""
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2)
		handle.write("\n")


def update_input(source_input: Path, output_input: Path, steps: int, decay_steps: int) -> None:
	"""Create a 10x-step DeePMD input from a source run input.

	Args:
		source_input: Existing case ``myinput.json``.
		output_input: Destination ``myinput.json``.
		steps: New optimizer step count.
		decay_steps: New LR decay steps.
	"""
	data = load_json(source_input)
	data["training"]["numb_steps"] = steps
	data["training"]["save_freq"] = choose_save_freq(steps)
	data["training"]["disp_freq"] = 10
	data["learning_rate"]["decay_steps"] = decay_steps
	write_json(output_input, data)


def rewrite_sbatch(source_sbatch: Path, output_sbatch: Path, new_case: NewCase) -> None:
	"""Create a 10x-step Slurm script from an existing validated launcher.

	Args:
		source_sbatch: Existing launcher.
		output_sbatch: Destination launcher.
		new_case: New case metadata.
	"""
	text = source_sbatch.read_text(encoding="utf-8")
	text = re.sub(r"^#SBATCH --job-name=.*$", f"#SBATCH --job-name=dpgb10x_{new_case.gpus}g_{new_case.steps}_{new_case.scale_by_worker}"[:64], text, flags=re.MULTILINE)
	text = re.sub(r"^#SBATCH --time=.*$", f"#SBATCH --time={TIME_LIMIT}", text, flags=re.MULTILINE)
	if f"CASE {new_case.source_case_id}" in text:
		text = text.replace(f"CASE {new_case.source_case_id}", f"CASE {new_case.case_id}")
	else:
		text = text.replace('echo "JOB_START $(date --iso-8601=seconds)"', f'echo "JOB_START $(date --iso-8601=seconds)"\necho "CASE {new_case.case_id}"')
	header = (
		"# 10x-step follow-up generated from an existing validated launcher.\n"
		f"# Source case: {new_case.source_case_id}\n"
		f"# Source dir: {new_case.source_dir}\n"
		f"# New steps: {new_case.steps}; decay_steps: {new_case.decay_steps}\n"
		f"# Priority: {new_case.priority}; {new_case.note}\n"
	)
	output_sbatch.write_text(text.replace("#!/bin/bash\n", f"#!/bin/bash\n{header}", 1), encoding="utf-8")


def write_run_info(path: Path, new_case: NewCase) -> None:
	"""Write a human-readable run note.

	Args:
		path: Destination Markdown path.
		new_case: New case metadata.
	"""
	rank_batches = new_case.gpus * new_case.steps
	path.write_text(
		f"""# {new_case.case_id}

10x-step follow-up for `{new_case.source_case_id}`.

```text
source_dir = {new_case.source_dir}
gpus = {new_case.gpus}
nodes = {new_case.nodes}
numb_steps = {new_case.steps}
decay_steps = {new_case.decay_steps}
scale_by_worker = {new_case.scale_by_worker}
approx_rank_batches = {rank_batches}
priority = {new_case.priority}
note = {new_case.note}
walltime = {TIME_LIMIT}
```

Submit from this directory with:

```bash
sbatch run_srun_train_mem.sbatch
```
""",
		encoding="utf-8",
	)


def make_new_case(case: Case) -> NewCase:
	"""Create new-case metadata for a source case.

	Args:
		case: Existing case.

	Returns:
		10x follow-up case metadata.
	"""
	source_dir = resolve_source_dir(case)
	case_id = f"{case.case_id}_10x"
	priority, note = priority_for(case)
	return NewCase(
		source_case_id=case.case_id,
		case_id=case_id,
		group=OUTPUT_GROUP,
		gpus=case.gpus,
		nodes=case.nodes,
		steps=case.steps * 10,
		decay_steps=case.decay_steps * 10,
		scale_by_worker=case.scale_by_worker,
		source_dir=source_dir,
		run_dir=OUTPUT_RUN_ROOT / case_id,
		priority=priority,
		note=note,
	)


def materialize(new_case: NewCase) -> None:
	"""Create one 10x follow-up run directory.

	Args:
		new_case: New case metadata.
	"""
	new_case.run_dir.mkdir(parents=True, exist_ok=True)
	update_input(
		new_case.source_dir / "myinput.json",
		new_case.run_dir / "myinput.json",
		new_case.steps,
		new_case.decay_steps,
	)
	rewrite_sbatch(
		new_case.source_dir / "run_srun_train_mem.sbatch",
		new_case.run_dir / "run_srun_train_mem.sbatch",
		new_case,
	)
	write_run_info(new_case.run_dir / "RUN_INFO.md", new_case)


def write_matrix(new_cases: list[NewCase]) -> None:
	"""Write a manifest for the 10x follow-up cases.

	Args:
		new_cases: Materialized 10x cases.
	"""
	fieldnames = [
		"case_id",
		"source_case_id",
		"group",
		"priority",
		"gpus",
		"nodes",
		"steps",
		"decay_steps",
		"scale_by_worker",
		"run_dir",
		"note",
	]
	with OUTPUT_MATRIX.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		for case in new_cases:
			writer.writerow(
				{
					"case_id": case.case_id,
					"source_case_id": case.source_case_id,
					"group": case.group,
					"priority": case.priority,
					"gpus": case.gpus,
					"nodes": case.nodes,
					"steps": case.steps,
					"decay_steps": case.decay_steps,
					"scale_by_worker": case.scale_by_worker,
					"run_dir": case.run_dir.relative_to(ROOT),
					"note": case.note,
				}
			)


def write_submit_script(new_cases: list[NewCase]) -> None:
	"""Write submission helpers for the 10x follow-up cases.

	Args:
		new_cases: Materialized 10x cases.
	"""
	for priority in ["core", "secondary", "diagnostic", "all"]:
		selected = new_cases if priority == "all" else [case for case in new_cases if case.priority == priority]
		lines = [
			"#!/bin/bash",
			"set -euo pipefail",
			"",
			f"# Submit {priority} 10x-step cases.",
			f"# All jobs use walltime {TIME_LIMIT}.",
			"",
		]
		for case in selected:
			lines.append(f"(cd {case.run_dir.relative_to(ROOT)} && sbatch run_srun_train_mem.sbatch)")
		lines.append("")
		path = ROOT / f"submit_10x_{priority}.sh"
		path.write_text("\n".join(lines), encoding="utf-8")
		path.chmod(0o755)


def copy_self() -> None:
	"""Copy this materialization script into the experiment scripts directory."""
	destination = ROOT / "scripts" / "materialize_10x_steps.py"
	destination.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(Path(__file__), destination)


def main() -> None:
	"""Materialize all requested 10x-step follow-up runs."""
	cases = [case for case in read_cases(MATRIX_PATH) if case.case_id not in SKIP_CASES]
	new_cases = [make_new_case(case) for case in cases]
	for new_case in new_cases:
		materialize(new_case)
	write_matrix(new_cases)
	write_submit_script(new_cases)
	copy_self()
	print(f"materialized\t{len(new_cases)}\t{OUTPUT_RUN_ROOT}")
	print(f"matrix\t{OUTPUT_MATRIX}")


if __name__ == "__main__":
	main()
