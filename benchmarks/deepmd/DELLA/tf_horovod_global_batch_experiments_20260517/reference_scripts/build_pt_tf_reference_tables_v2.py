#!/usr/bin/env python3
"""Build compact PT/TF benchmark reference tables with explicit leading columns."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SCRATCH_ROOT = Path(
	"/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/"
	"v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/"
	"global_batch_experiments_20260517"
)
REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference_results"
COMPARISON_TSV = SCRATCH_ROOT / "PT_TF_NONE_TRAIN_VAL_COMPARISON_20260523.tsv"

LEADING_COLUMNS = [
	"framework",
	"name",
	"checkpoint_step",
	"final_total_steps",
	"best_TRAIN_total_loss_step",
	"TRAIN_total_loss",
	"TRAIN_E_RMSE_eV_per_atom",
	"VAL_E_RMSE_eV_per_atom",
	"TRAIN_F_RMSE_eV_per_A",
	"VAL_F_RMSE_eV_per_A",
]
EXTRA_COLUMNS = [
	"checkpoint_label",
	"group",
	"status",
	"gpus",
	"TRAIN_runtime_elapsed",
	"TRAIN_runtime_s",
	"VAL_runtime_elapsed",
	"VAL_runtime_s",
	"TRAIN_V_RMSE_eV_per_atom",
	"VAL_V_RMSE_eV_per_atom",
	"validation_frames_total",
	"unique_training_frames_total",
	"source_result_root",
	"source_run_dir",
]


def read_tsv(path: Path) -> List[Dict[str, str]]:
	"""Read a TSV file.

	Args:
		path: File path to read.

	Returns:
		List of dictionaries keyed by the TSV header.
	"""
	with path.open(newline="", encoding="utf-8") as handle:
		return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
	"""Write a TSV file with the requested model-selection columns first.

	Args:
		path: Destination path.
		rows: Table rows.
	"""
	rows = list(rows)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=LEADING_COLUMNS + EXTRA_COLUMNS, delimiter="\t", extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)


def case_base(case_id: str) -> str:
	"""Remove final/best checkpoint suffixes from a case id.

	Args:
		case_id: Case id.

	Returns:
		Base case id.
	"""
	for suffix in ("__final", "__best_total"):
		if case_id.endswith(suffix):
			return case_id[: -len(suffix)]
	return case_id


def infer_run_dir(row: Dict[str, str]) -> str:
	"""Infer the scratch run directory for a comparison row.

	Args:
		row: Row from the PT/TF comparison table.

	Returns:
		Run directory, or an empty string if unknown.
	"""
	base = case_base(row["case_id"])
	if row.get("framework") == "TF":
		return str(SCRATCH_ROOT / "runs" / "none_100k" / base)
	if row.get("framework") != "PT":
		return ""
	if base.startswith("ptss"):
		mapping = {
			"ptss1g_100000": "1gpu_100k_stepscaled_none_pt",
			"ptss4g_25000": "4gpu_25k_stepscaled_none_pt",
			"ptss8g_12500": "8gpu_12500_stepscaled_none_pt",
			"ptss16g_6250": "16gpu_6250_stepscaled_none_pt",
		}
		target = mapping.get(base)
		return str(SCRATCH_ROOT / "runs" / "pt_step_scaled_none" / target) if target else ""
	mapping = {
		"pt1g_100k_none": "1gpu_100k_none_pt",
		"pt4g_100k_none": "4gpu_100k_none_pt",
		"pt8g_100k_none": "8gpu_100k_none_pt",
		"pt16g_100k_none": "16gpu_100k_none_pt",
	}
	target = mapping.get(base)
	return str(SCRATCH_ROOT / "runs" / "pt_none_100k" / target) if target else ""


def load_training_metadata() -> Dict[str, Dict[str, str]]:
	"""Load scratch-side TF metadata.

	Returns:
		Mapping from case id to metadata.
	"""
	metadata: Dict[str, Dict[str, str]] = {}
	for name in ("TRAINING_SUMMARY.tsv", "TRAINING_SUMMARY_10x_COMPLETED.tsv"):
		path = SCRATCH_ROOT / name
		if not path.exists():
			continue
		for row in read_tsv(path):
			metadata[row["case_id"]] = row
	return metadata


def lcurve_row(run_dir: Path, target_step: int) -> Tuple[int, float, float, float, str]:
	"""Return the lcurve row closest to a target step.

	Args:
		run_dir: Run directory containing ``lcurve.out``.
		target_step: Target training step.

	Returns:
		Step, total loss, energy RMSE, force RMSE, and virial RMSE string.
	"""
	best: Optional[Tuple[int, float, float, float, str]] = None
	best_distance: Optional[int] = None
	with (run_dir / "lcurve.out").open(encoding="utf-8") as handle:
		for line in handle:
			stripped = line.strip()
			if not stripped or stripped.startswith("#"):
				continue
			parts = stripped.split()
			if len(parts) < 4:
				continue
			step = int(float(parts[0]))
			distance = abs(step - target_step)
			if best_distance is None or distance < best_distance:
				best_distance = distance
				best = (step, float(parts[1]), float(parts[2]), float(parts[3]), parts[4] if len(parts) > 4 else "")
			if distance == 0:
				break
	if best is None:
		raise ValueError(f"No lcurve rows found in {run_dir / 'lcurve.out'}")
	return best


def best_total_loss_step(run_dir: Path) -> str:
	"""Find the actual lcurve step with the lowest total training loss.

	Args:
		run_dir: Run directory containing ``lcurve.out``.

	Returns:
		Training step as a string.
	"""
	best_step: Optional[int] = None
	best_total: Optional[float] = None
	with (run_dir / "lcurve.out").open(encoding="utf-8") as handle:
		for line in handle:
			stripped = line.strip()
			if not stripped or stripped.startswith("#"):
				continue
			parts = stripped.split()
			if len(parts) < 2:
				continue
			step = int(float(parts[0]))
			total = float(parts[1])
			if best_total is None or total < best_total:
				best_total = total
				best_step = step
	return "" if best_step is None else str(best_step)


def count_unique_training_frames(run_dir: Path) -> str:
	"""Count unique training frames from a run input file.

	Args:
		run_dir: Run directory containing ``myinput.json``.

	Returns:
		Unique training-frame count, or an empty string if unavailable.
	"""
	input_path = run_dir / "myinput.json"
	if not input_path.exists():
		return ""
	with input_path.open(encoding="utf-8") as handle:
		data = json.load(handle)
	systems = {
		Path(path)
		for path in data.get("training", {}).get("training_data", {}).get("systems", [])
	}
	total = 0
	for system in systems:
		for coord in system.glob("set.*/coord.npy"):
			total += int(np.load(coord, mmap_mode="r").shape[0])
	return str(total)


def comparison_rows() -> Dict[str, Dict[str, str]]:
	"""Build PT rows and current TF none-100k rows.

	Returns:
		Rows keyed by case id.
	"""
	rows: Dict[str, Dict[str, str]] = {}
	for row in read_tsv(COMPARISON_TSV):
		if row.get("framework") not in {"PT", "TF"}:
			continue
		run_dir = infer_run_dir(row)
		best_step = row["checkpoint_step"]
		unique_train_frames = ""
		if run_dir and Path(run_dir).exists():
			best_step = best_total_loss_step(Path(run_dir)) or best_step
			unique_train_frames = count_unique_training_frames(Path(run_dir))
		rows[row["case_id"]] = {
			"framework": row["framework"],
			"name": row["case_id"],
			"checkpoint_step": row["checkpoint_step"],
			"final_total_steps": row["steps"],
			"best_TRAIN_total_loss_step": best_step,
			"TRAIN_total_loss": row["TRAIN_total_loss_at_checkpoint"],
			"TRAIN_E_RMSE_eV_per_atom": row["TRAIN_E_RMSE_per_atom_eV"],
			"VAL_E_RMSE_eV_per_atom": row["VAL_E_RMSE_per_atom_eV"],
			"TRAIN_F_RMSE_eV_per_A": row["TRAIN_F_RMSE_eV_per_A"],
			"VAL_F_RMSE_eV_per_A": row["VAL_F_RMSE_eV_per_A"],
			"checkpoint_label": row["checkpoint_label"],
			"group": row["group"],
			"status": row["status"],
			"gpus": row["gpus"],
			"TRAIN_runtime_elapsed": row["TRAIN_runtime_elapsed"],
			"TRAIN_runtime_s": row["TRAIN_runtime_s"],
			"VAL_runtime_elapsed": row["VAL_runtime_elapsed"],
			"VAL_runtime_s": row["VAL_runtime_s"],
			"TRAIN_V_RMSE_eV_per_atom": row.get("TRAIN_V_RMSE_per_atom_eV", ""),
			"VAL_V_RMSE_eV_per_atom": row["VAL_V_RMSE_per_atom_eV"],
			"validation_frames_total": row["frames_total"],
			"unique_training_frames_total": unique_train_frames,
			"source_result_root": row["result_root"],
			"source_run_dir": run_dir,
		}
	return rows


def add_historical_tf_rows(rows: Dict[str, Dict[str, str]]) -> None:
	"""Add older TF validation rows without duplicating explicit final rows.

	Args:
		rows: Row mapping to update in place.
	"""
	metadata = load_training_metadata()
	for summary in SCRATCH_ROOT.glob("pseudo_validation*/PSEUDO_VALIDATION_SUMMARY.tsv"):
		for row in read_tsv(summary):
			if row.get("row_type") != "aggregate" or row.get("dataset_id") != "all" or row.get("split") != "all":
				continue
			case_id = row["case_id"]
			if case_id in rows or f"{case_id}__final" in rows or f"{case_id}__best_total" in rows:
				continue
			if case_id not in metadata:
				continue
			run_dir = Path(metadata[case_id]["run_dir"])
			total_steps = int(float(metadata[case_id].get("final_step") or metadata[case_id].get("steps") or row.get("steps") or 0))
			step, total, energy, force, virial = lcurve_row(run_dir, total_steps)
			rows[case_id] = {
				"framework": "TF",
				"name": case_id,
				"checkpoint_step": str(step),
				"final_total_steps": str(total_steps),
				"best_TRAIN_total_loss_step": best_total_loss_step(run_dir),
				"TRAIN_total_loss": f"{total:.8g}",
				"TRAIN_E_RMSE_eV_per_atom": f"{energy:.8g}",
				"VAL_E_RMSE_eV_per_atom": row["energy_rmse_per_atom"],
				"TRAIN_F_RMSE_eV_per_A": f"{force:.8g}",
				"VAL_F_RMSE_eV_per_A": row["force_rmse"],
				"checkpoint_label": "final",
				"group": metadata[case_id].get("group", ""),
				"status": metadata[case_id].get("status", ""),
				"gpus": metadata[case_id].get("gpus", ""),
				"TRAIN_runtime_elapsed": "",
				"TRAIN_runtime_s": metadata[case_id].get("wall_time_s", ""),
				"VAL_runtime_elapsed": "",
				"VAL_runtime_s": "",
				"TRAIN_V_RMSE_eV_per_atom": virial,
				"VAL_V_RMSE_eV_per_atom": row.get("virial_rmse_per_atom", ""),
				"validation_frames_total": "",
				"unique_training_frames_total": count_unique_training_frames(run_dir),
				"source_result_root": str(summary.parent),
				"source_run_dir": str(run_dir),
			}


def sorted_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
	"""Sort rows by validation energy RMSE.

	Args:
		rows: Rows to sort.

	Returns:
		Sorted rows.
	"""
	return sorted(rows, key=lambda row: float(row["VAL_E_RMSE_eV_per_atom"]))


def write_markdown_summary(all_rows: List[Dict[str, str]]) -> None:
	"""Write the markdown companion note.

	Args:
		all_rows: Full sorted table.
	"""
	pt_rows = [row for row in all_rows if row["framework"] == "PT"]
	tf_rows = [row for row in all_rows if row["framework"] == "TF"]
	text = f"""# PT/TF Validation Reference Tables

Date: 2026-05-23

These compact reference tables summarize the Della NH3/H2 DeePMD TF/Horovod
and PT benchmark validation runs. The first columns are intentionally ordered
for model selection:

```text
{chr(9).join(LEADING_COLUMNS)}
```

Extra columns record checkpoint label, group/status/GPU count, runtimes,
virial metrics, validation frame counts, unique training-frame counts when
available, and scratch provenance paths.

Files:

- `PT_TF_VALIDATION_REFERENCE_20260523.tsv`: all TF and PT rows, sorted by
  validation energy RMSE per atom.
- `PT_VALIDATION_REFERENCE_20260523.tsv`: PT-only slice.
- `TF_VALIDATION_REFERENCE_20260523.tsv`: TF-only slice.

Row counts:

- all: {len(all_rows)}
- PT: {len(pt_rows)}
- TF: {len(tf_rows)}

Main caution: the 10x-labelled TF runs here use the same ~13.6k-frame training
pool as the 100k-none runs; the label reflects training schedule/steps, not
10x more distinct frames.
"""
	(REFERENCE_ROOT / "PT_TF_VALIDATION_REFERENCE_20260523.md").write_text(text, encoding="utf-8")


def main() -> None:
	"""Build and save compact benchmark reference tables."""
	REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
	rows_by_name = comparison_rows()
	add_historical_tf_rows(rows_by_name)
	all_rows = sorted_rows(rows_by_name.values())
	write_tsv(REFERENCE_ROOT / "PT_TF_VALIDATION_REFERENCE_20260523.tsv", all_rows)
	write_tsv(REFERENCE_ROOT / "PT_VALIDATION_REFERENCE_20260523.tsv", [row for row in all_rows if row["framework"] == "PT"])
	write_tsv(REFERENCE_ROOT / "TF_VALIDATION_REFERENCE_20260523.tsv", [row for row in all_rows if row["framework"] == "TF"])
	write_markdown_summary(all_rows)
	print(f"Wrote {len(all_rows)} rows to {REFERENCE_ROOT}")


if __name__ == "__main__":
	main()
