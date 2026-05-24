#!/usr/bin/env python3
"""Build compact PT/TF benchmark reference tables.

The scratch experiment directory contains the full run tree and pseudo-validation
outputs. This script extracts the decision-facing columns first, then appends a
small amount of provenance so the benchmark folder can be read without loading
raw logs or checkpoints.
"""

from __future__ import annotations

import csv
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
	"frames_total",
	"source_result_root",
	"source_run_dir",
]


def read_tsv(path: Path) -> List[Dict[str, str]]:
	"""Read a tab-separated file into dictionaries.

	Args:
		path: File to read.

	Returns:
		List of row dictionaries keyed by the TSV header.
	"""
	with path.open(newline="", encoding="utf-8") as handle:
		return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
	"""Write rows to a tab-separated file.

	Args:
		path: Destination TSV path.
		rows: Row dictionaries to write.
	"""
	rows = list(rows)
	fieldnames = LEADING_COLUMNS + EXTRA_COLUMNS
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
		writer.writeheader()
		writer.writerows(rows)


def case_base(case_id: str) -> str:
	"""Remove checkpoint-label suffixes from a case id.

	Args:
		case_id: Case id, possibly ending in ``__final`` or ``__best_total``.

	Returns:
		Base case id.
	"""
	for suffix in ("__final", "__best_total"):
		if case_id.endswith(suffix):
			return case_id[: -len(suffix)]
	return case_id


def load_training_metadata() -> Dict[str, Dict[str, str]]:
	"""Load scratch-side run metadata used to locate TF lcurve files.

	Returns:
		Mapping from base case id to summary metadata.
	"""
	metadata: Dict[str, Dict[str, str]] = {}
	for name in ("TRAINING_SUMMARY.tsv", "TRAINING_SUMMARY_10x_COMPLETED.tsv"):
		path = SCRATCH_ROOT / name
		if not path.exists():
			continue
		for row in read_tsv(path):
			metadata[row["case_id"]] = row
	if COMPARISON_TSV.exists():
		for row in read_tsv(COMPARISON_TSV):
			if row.get("framework") != "TF":
				continue
			base = case_base(row["case_id"])
			metadata.setdefault(base, {})
			metadata[base]["run_dir"] = str(SCRATCH_ROOT / "runs" / "none_100k" / base)
			metadata[base]["steps"] = row.get("steps", "")
			metadata[base]["wall_time_s"] = row.get("TRAIN_runtime_s", "")
	return metadata


def lcurve_row(run_dir: Path, target_step: int) -> Tuple[int, float, float, float, str]:
	"""Return the lcurve row nearest a target training step.

	Args:
		run_dir: DeePMD run directory containing ``lcurve.out``.
		target_step: Desired training step.

	Returns:
		Step, total loss, energy RMSE, force RMSE, and virial RMSE as available.
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
				virial = parts[4] if len(parts) > 4 else ""
				best_distance = distance
				best = (step, float(parts[1]), float(parts[2]), float(parts[3]), virial)
			if distance == 0:
				break
	if best is None:
		raise ValueError(f"No lcurve rows found in {run_dir / 'lcurve.out'}")
	return best


def best_total_loss_step(run_dir: Path) -> int:
	"""Find the training step with the lowest lcurve total loss.

	Args:
		run_dir: DeePMD run directory containing ``lcurve.out``.

	Returns:
		Training step at the lowest total loss.
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
	if best_step is None:
		raise ValueError(f"No lcurve rows found in {run_dir / 'lcurve.out'}")
	return best_step


def count_unique_training_frames(run_dir: Path) -> str:
	"""Count unique training frames from ``myinput.json`` when available.

	Args:
		run_dir: DeePMD run directory.

	Returns:
		Stringified frame count, or an empty string when unavailable.
	"""
	input_path = run_dir / "myinput.json"
	if not input_path.exists():
		return ""
	import json

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
	"""Build PT rows and current TF none-100k rows from the comparison TSV.

	Returns:
		Rows keyed by case id.
	"""
	rows: Dict[str, Dict[str, str]] = {}
	raw_rows = read_tsv(COMPARISON_TSV)
	best_steps = {
		case_base(row["case_id"]): row["checkpoint_step"]
		for row in raw_rows
		if row.get("checkpoint_label") == "best_total"
	}
	for row in raw_rows:
		if row.get("framework") not in {"PT", "TF"}:
			continue
		base = case_base(row["case_id"])
		run_dir = ""
		if row.get("framework") == "TF":
			run_dir = str(SCRATCH_ROOT / "runs" / "none_100k" / base)
		rows[row["case_id"]] = {
			"framework": row["framework"],
			"name": row["case_id"],
			"checkpoint_step": row["checkpoint_step"],
			"final_total_steps": row["steps"],
			"best_TRAIN_total_loss_step": best_steps.get(base, row["checkpoint_step"]),
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
			"frames_total": row["frames_total"],
			"source_result_root": row["result_root"],
			"source_run_dir": run_dir,
		}
	return rows


def historical_tf_rows(rows: Dict[str, Dict[str, str]]) -> None:
	"""Add historical TF validation rows from pseudo-validation summaries.

	Args:
		rows: Existing row mapping to update in place.
	"""
	metadata = load_training_metadata()
	for summary in SCRATCH_ROOT.glob("pseudo_validation*/PSEUDO_VALIDATION_SUMMARY.tsv"):
		for row in read_tsv(summary):
			if row.get("row_type") != "aggregate" or row.get("dataset_id") != "all" or row.get("split") != "all":
				continue
			case_id = row["case_id"]
			base = case_base(case_id)
			if case_id in rows or base not in metadata:
				continue
			run_dir = Path(metadata[base]["run_dir"])
			total_steps = int(float(metadata[base].get("final_step") or metadata[base].get("steps") or row.get("steps") or 0))
			step, total, energy, force, virial = lcurve_row(run_dir, total_steps)
			rows[case_id] = {
				"framework": "TF",
				"name": case_id,
				"checkpoint_step": str(step),
				"final_total_steps": str(total_steps),
				"best_TRAIN_total_loss_step": str(best_total_loss_step(run_dir)),
				"TRAIN_total_loss": f"{total:.8g}",
				"TRAIN_E_RMSE_eV_per_atom": f"{energy:.8g}",
				"VAL_E_RMSE_eV_per_atom": row["energy_rmse_per_atom"],
				"TRAIN_F_RMSE_eV_per_A": f"{force:.8g}",
				"VAL_F_RMSE_eV_per_A": row["force_rmse"],
				"checkpoint_label": "final",
				"group": metadata[base].get("group", ""),
				"status": metadata[base].get("status", ""),
				"gpus": metadata[base].get("gpus", ""),
				"TRAIN_runtime_elapsed": "",
				"TRAIN_runtime_s": metadata[base].get("wall_time_s", ""),
				"VAL_runtime_elapsed": "",
				"VAL_runtime_s": "",
				"TRAIN_V_RMSE_eV_per_atom": virial,
				"VAL_V_RMSE_eV_per_atom": row.get("virial_rmse_per_atom", ""),
				"frames_total": count_unique_training_frames(run_dir),
				"source_result_root": str(summary.parent),
				"source_run_dir": str(run_dir),
			}


def sorted_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
	"""Sort rows by validation energy RMSE.

	Args:
		rows: Rows to sort.

	Returns:
		Rows sorted by ``VAL_E_RMSE_eV_per_atom``.
	"""
	return sorted(rows, key=lambda row: float(row["VAL_E_RMSE_eV_per_atom"]))


def write_markdown_summary(all_rows: List[Dict[str, str]]) -> None:
	"""Write a short markdown companion note.

	Args:
		all_rows: Full sorted reference table rows.
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
virial metrics, frame counts when available, and scratch provenance paths.

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
	"""Build and save all compact benchmark reference tables."""
	REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
	rows_by_name = comparison_rows()
	historical_tf_rows(rows_by_name)
	all_rows = sorted_rows(rows_by_name.values())
	write_tsv(REFERENCE_ROOT / "PT_TF_VALIDATION_REFERENCE_20260523.tsv", all_rows)
	write_tsv(REFERENCE_ROOT / "PT_VALIDATION_REFERENCE_20260523.tsv", [row for row in all_rows if row["framework"] == "PT"])
	write_tsv(REFERENCE_ROOT / "TF_VALIDATION_REFERENCE_20260523.tsv", [row for row in all_rows if row["framework"] == "TF"])
	write_markdown_summary(all_rows)
	print(f"Wrote {len(all_rows)} rows to {REFERENCE_ROOT}")


if __name__ == "__main__":
	main()
