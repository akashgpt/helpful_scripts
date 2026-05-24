#!/usr/bin/env python3
"""Plot stitched DeePMD checkpoint-restart training curves.

This script is a reusable template for restart-chain diagnostics. It expects
each run directory to contain a cumulative ``lcurve.out`` and, optionally,
``CHAIN_HISTORY.tsv`` from the self-resubmitting restart scripts.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


SCRATCH_ROOT = Path(
	"/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/"
	"v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/"
	"global_batch_experiments_20260517"
)
RESTART_TEST_ROOT = SCRATCH_ROOT / "runs" / "restart_tests_20260523"
INTERRUPT_RESTART_ROOT = SCRATCH_ROOT / "runs" / "interrupt_restart_chain_20260524"
REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference_results"
SCRATCH_PLOT_ROOT = SCRATCH_ROOT / "training_loss_plots_10x_20260521" / "CONTINUATIONS"

OUTPUT_BASENAME = "TF_PT_4GPU_RESTART_STITCHED_TRAINING_EVOLUTION"
OUTPUT_PNG = REFERENCE_ROOT / f"{OUTPUT_BASENAME}_ROLLING_MEDIAN_MEAN_20260523.png"
OUTPUT_TSV = REFERENCE_ROOT / f"{OUTPUT_BASENAME}_SUMMARY_20260523.tsv"
OUTPUT_MD = REFERENCE_ROOT / f"{OUTPUT_BASENAME}_ANALYSIS_20260523.md"


@dataclass(frozen=True)
class RunSpec:
	"""Metadata for one checkpoint-restart training curve."""

	label: str
	run_dir: Path
	color: str
	line_style: str
	marker_minutes: int
	note: str


RUNS: Tuple[RunSpec, ...] = (
	RunSpec(
		"TF 4gpu linear 10k -> 110k",
		RESTART_TEST_ROOT / "tf4g_10k_restart_plus100k",
		"tab:blue",
		"-",
		15,
		"TF continuation from the 10k linear run",
	),
	RunSpec(
		"TF 4gpu none 100k -> 200k",
		INTERRUPT_RESTART_ROOT / "tf4g_100k_none_10min_chain_healthgate_second_latest_ckpt",
		"tab:cyan",
		"-",
		10,
		"TF none continuation with health gate and second-latest restart checkpoint",
	),
	RunSpec(
		"PT 4gpu none 100k -> 200k",
		INTERRUPT_RESTART_ROOT / "pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2",
		"tab:red",
		"-",
		10,
		"PT none continuation with health gate and second-latest restart checkpoint",
	),
)

METRICS: Tuple[Tuple[str, str, int], ...] = (
	("total", "TRAIN total loss/RMSE", 1),
	("energy", "TRAIN E RMSE (eV/atom)", 2),
	("force", "TRAIN F RMSE (eV/A)", 3),
	("virial_stress", "TRAIN virial/stress RMSE (eV/atom)", 4),
)


def load_lcurve(path: Path) -> NDArray[np.float64]:
	"""Load a DeePMD lcurve file.

	Args:
		path: Path to ``lcurve.out``.

	Returns:
		Two-dimensional numeric lcurve array.
	"""
	array = np.loadtxt(path)
	if array.ndim == 1:
		array = array.reshape(1, -1)
	return array


def choose_window(n_rows: int) -> int:
	"""Choose an odd rolling window for noisy DeePMD training curves.

	Args:
		n_rows: Number of lcurve records.

	Returns:
		Odd rolling-window length.
	"""
	window = min(501, max(101, n_rows // 20))
	if window % 2 == 0:
		window += 1
	return max(1, min(window, n_rows if n_rows % 2 == 1 else n_rows - 1))


def rolling_stat(values: NDArray[np.float64], window: int, use_median: bool) -> NDArray[np.float64]:
	"""Compute centered rolling median or mean.

	Args:
		values: Metric values.
		window: Rolling-window length.
		use_median: If true, compute median; otherwise compute mean.

	Returns:
		Smoothed values with the same length as ``values``.
	"""
	if window <= 1:
		return values.copy()
	half_window = window // 2
	output = np.empty_like(values, dtype=float)
	for index in range(values.size):
		start = max(0, index - half_window)
		stop = min(values.size, index + half_window + 1)
		chunk = values[start:stop]
		output[index] = np.nanmedian(chunk) if use_median else np.nanmean(chunk)
	return output


def parse_restart_steps(chain_history: Path) -> List[int]:
	"""Parse restart-boundary steps from a chain-history file.

	Args:
		chain_history: Path to ``CHAIN_HISTORY.tsv``.

	Returns:
		Restart-boundary steps.
	"""
	if not chain_history.exists():
		return []
	steps: List[int] = []
	for line in chain_history.read_text(encoding="utf-8", errors="replace").splitlines():
		match = re.search(r"\bstep=([0-9]+)\b", line)
		if match is not None:
			steps.append(int(match.group(1)))
	return sorted(set(steps))


def finite_min_step(steps: NDArray[np.float64], values: NDArray[np.float64]) -> Tuple[int, float]:
	"""Find the finite minimum value and its step.

	Args:
		steps: Training steps.
		values: Metric values.

	Returns:
		Step and value at the finite minimum.
	"""
	mask = np.isfinite(values)
	filtered_steps = steps[mask]
	filtered_values = values[mask]
	min_index = int(np.argmin(filtered_values))
	return int(filtered_steps[min_index]), float(filtered_values[min_index])


def summarize_run(spec: RunSpec, lcurve: NDArray[np.float64]) -> Dict[str, str]:
	"""Summarize one stitched restart curve.

	Args:
		spec: Run metadata.
		lcurve: Numeric lcurve array.

	Returns:
		Summary row.
	"""
	steps = lcurve[:, 0]
	last_window_start = max(0, int(lcurve.shape[0] * 0.9))
	row: Dict[str, str] = {
		"label": spec.label,
		"run_dir": str(spec.run_dir),
		"note": spec.note,
		"first_step": str(int(steps[0])),
		"last_step": str(int(steps[-1])),
		"n_lcurve_rows": str(lcurve.shape[0]),
		"restart_marker_steps": ",".join(str(step) for step in parse_restart_steps(spec.run_dir / "CHAIN_HISTORY.tsv")),
	}
	for metric_name, _, column in METRICS:
		values = lcurve[:, column]
		min_step, min_value = finite_min_step(steps, values)
		row[f"final_{metric_name}"] = f"{values[-1]:.8g}"
		row[f"best_{metric_name}"] = f"{min_value:.8g}"
		row[f"best_{metric_name}_step"] = str(min_step)
		row[f"last10pct_median_{metric_name}"] = f"{np.nanmedian(values[last_window_start:]):.8g}"
	return row


def plot_curves(curves: Dict[RunSpec, NDArray[np.float64]]) -> None:
	"""Plot rolling median/mean stitched restart curves.

	Args:
		curves: Mapping from run metadata to lcurve arrays.
	"""
	fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
	for axis, (metric_name, ylabel, column) in zip(axes.ravel(), METRICS):
		for spec, lcurve in curves.items():
			steps = lcurve[:, 0]
			values = lcurve[:, column]
			window = choose_window(values.size)
			median_values = rolling_stat(values, window, True)
			mean_values = rolling_stat(values, window, False)
			axis.plot(steps, median_values, spec.line_style, color=spec.color, linewidth=1.8, label=f"{spec.label} median")
			axis.plot(steps, mean_values, "--", color=spec.color, linewidth=1.0, alpha=0.75, label=f"{spec.label} mean")
			for marker_step in parse_restart_steps(spec.run_dir / "CHAIN_HISTORY.tsv"):
				if steps[0] <= marker_step <= steps[-1]:
					axis.axvline(marker_step, color=spec.color, alpha=0.75, linewidth=1.0)
		axis.set_title(metric_name)
		axis.set_ylabel(ylabel)
		axis.set_yscale("log")
		axis.grid(True, alpha=0.35)
		axis.legend(fontsize=6.5, ncol=1, framealpha=0.85)
	for axis in axes[-1, :]:
		axis.set_xlabel("training step")
	fig.suptitle("4GPU checkpoint-restart training evolution", fontsize=14)
	fig.tight_layout()
	fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
	plt.close(fig)


def write_summary(rows: List[Dict[str, str]]) -> None:
	"""Write summary TSV and markdown analysis.

	Args:
		rows: Summary rows.
	"""
	with OUTPUT_TSV.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)
	text = f"""# 4GPU Checkpoint Restart Stitched Training Curves

Date: 2026-05-23

Plot:

```text
{OUTPUT_PNG}
```

Summary table:

```text
{OUTPUT_TSV}
```

Recipe:

- Use the cumulative ``lcurve.out`` in each restart-chain run directory.
- Smooth each metric with both rolling median and rolling mean.
- Draw restart-boundary markers from ``CHAIN_HISTORY.tsv`` when present.
- Read this as an optimizer-health/restart-continuity diagnostic; still use
  validation RMSE before ranking continuation checkpoints.
"""
	OUTPUT_MD.write_text(text, encoding="utf-8")


def copy_to_scratch_plot_folder() -> None:
	"""Copy outputs to the scratch plot folder for interactive browsing."""
	SCRATCH_PLOT_ROOT.mkdir(parents=True, exist_ok=True)
	for path in (OUTPUT_PNG, OUTPUT_TSV, OUTPUT_MD):
		(SCRATCH_PLOT_ROOT / path.name).write_bytes(path.read_bytes())


def main() -> None:
	"""Build stitched restart training-curve plot and summaries."""
	REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
	curves: Dict[RunSpec, NDArray[np.float64]] = {}
	rows: List[Dict[str, str]] = []
	for spec in RUNS:
		lcurve = load_lcurve(spec.run_dir / "lcurve.out")
		curves[spec] = lcurve
		rows.append(summarize_run(spec, lcurve))
	plot_curves(curves)
	write_summary(rows)
	copy_to_scratch_plot_folder()


if __name__ == "__main__":
	main()
