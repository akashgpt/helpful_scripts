#!/usr/bin/env python3
"""Plot representative TF DeePMD training-loss evolution across GPU counts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference_results"
OUTPUT_PNG = REFERENCE_ROOT / "TF_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png"
OUTPUT_TSV = REFERENCE_ROOT / "TF_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_SUMMARY_20260523.tsv"
OUTPUT_MD = REFERENCE_ROOT / "TF_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ANALYSIS_20260523.md"

# This is the closest available 1/4/8/16-GPU TF long-training comparison. It is
# intentionally labeled "representative" because the exact PT fixed-100k none
# schedule has only 4GPU and 8GPU TF counterparts in this benchmark tree.
RUNS: Tuple[Tuple[str, Path, str], ...] = (
	("1gpu", SCRATCH_ROOT / "runs/long_steps_10x/reuse_1gpu_10k_10x", "linear, 100k requested"),
	("4gpu", SCRATCH_ROOT / "runs/long_steps_10x/reuse_4gpu_10k_10x", "linear, 100k requested"),
	("8gpu", SCRATCH_ROOT / "runs/long_steps_10x/reuse_8gpu_10k_10x", "linear, 100k requested"),
	("16gpu", SCRATCH_ROOT / "runs/long_steps_10x_continuations/reuse_16gpu_10k_10x_from50k", "linear continuation to 100k"),
)

METRICS: Tuple[Tuple[str, str, int], ...] = (
	("total", "TRAIN total loss/RMSE", 1),
	("energy", "TRAIN E RMSE (eV/atom)", 2),
	("force", "TRAIN F RMSE (eV/A)", 3),
	("virial_stress", "TRAIN virial/stress RMSE (eV/atom)", 4),
)


def load_lcurve(path: Path) -> NDArray[np.float64]:
	"""Load numeric lcurve data.

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
	"""Choose an odd rolling-statistic window.

	Args:
		n_rows: Number of lcurve rows.

	Returns:
		Window size.
	"""
	window = min(501, max(101, n_rows // 20))
	if window % 2 == 0:
		window += 1
	return min(window, n_rows if n_rows % 2 == 1 else n_rows - 1)


def rolling_stat(values: NDArray[np.float64], window: int, use_median: bool) -> NDArray[np.float64]:
	"""Compute centered rolling mean or median.

	Args:
		values: Metric values.
		window: Window size.
		use_median: If true, compute median; otherwise mean.

	Returns:
		Smoothed series.
	"""
	if window <= 1:
		return values.copy()
	half_window = window // 2
	smoothed = np.empty_like(values, dtype=float)
	for index in range(values.size):
		start = max(0, index - half_window)
		stop = min(values.size, index + half_window + 1)
		chunk = values[start:stop]
		smoothed[index] = np.nanmedian(chunk) if use_median else np.nanmean(chunk)
	return smoothed


def finite_min_step(steps: NDArray[np.float64], values: NDArray[np.float64]) -> Tuple[int, float]:
	"""Find the finite minimum in one metric.

	Args:
		steps: Training steps.
		values: Metric values.

	Returns:
		Step and value at the minimum.
	"""
	mask = np.isfinite(values)
	filtered_values = values[mask]
	filtered_steps = steps[mask]
	min_index = int(np.argmin(filtered_values))
	return int(filtered_steps[min_index]), float(filtered_values[min_index])


def summarize_run(name: str, note: str, lcurve: NDArray[np.float64]) -> Dict[str, str]:
	"""Summarize one training curve.

	Args:
		name: Short run label.
		note: Schedule note.
		lcurve: Numeric lcurve array.

	Returns:
		Summary row.
	"""
	steps = lcurve[:, 0]
	last_window_start = max(0, int(lcurve.shape[0] * 0.9))
	row: Dict[str, str] = {
		"run": name,
		"note": note,
		"first_step": str(int(steps[0])),
		"last_step": str(int(steps[-1])),
		"n_lcurve_rows": str(lcurve.shape[0]),
	}
	for metric_name, _, column in METRICS:
		values = lcurve[:, column]
		min_step, min_value = finite_min_step(steps, values)
		row[f"final_{metric_name}"] = f"{values[-1]:.8g}"
		row[f"best_{metric_name}"] = f"{min_value:.8g}"
		row[f"best_{metric_name}_step"] = str(min_step)
		row[f"last10pct_median_{metric_name}"] = f"{np.nanmedian(values[last_window_start:]):.8g}"
	return row


def plot_curves(curves: Dict[str, NDArray[np.float64]]) -> None:
	"""Plot rolling median and mean curves.

	Args:
		curves: Mapping from run label to lcurve data.
	"""
	fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
	for axis, (metric_name, ylabel, column) in zip(axes.ravel(), METRICS):
		for run_name, lcurve in curves.items():
			steps = lcurve[:, 0]
			values = lcurve[:, column]
			window = choose_window(values.size)
			median_values = rolling_stat(values, window, True)
			mean_values = rolling_stat(values, window, False)
			line = axis.plot(steps, median_values, linewidth=1.7, label=f"{run_name} median")[0]
			axis.plot(steps, mean_values, "--", linewidth=1.0, alpha=0.75, color=line.get_color(), label=f"{run_name} mean")
			axis.axhline(median_values[-1], linestyle=":", linewidth=1.2, alpha=0.5, color=line.get_color())
			axis.axhline(mean_values[-1], linestyle="-.", linewidth=1.0, alpha=0.35, color=line.get_color())
		axis.set_title(metric_name)
		axis.set_ylabel(ylabel)
		axis.set_yscale("log")
		axis.grid(True, alpha=0.35)
		axis.legend(fontsize=7, ncol=2, framealpha=0.85)
	for axis in axes[-1, :]:
		axis.set_xlabel("training step")
	fig.suptitle("TF representative long-training evolution: rolling median and mean", fontsize=14)
	fig.tight_layout()
	fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
	plt.close(fig)


def write_summary(rows: List[Dict[str, str]]) -> None:
	"""Write TSV and markdown summaries.

	Args:
		rows: Summary rows.
	"""
	fieldnames = list(rows[0].keys())
	with OUTPUT_TSV.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)

	best_energy = min(rows, key=lambda row: float(row["last10pct_median_energy"]))
	best_force = min(rows, key=lambda row: float(row["last10pct_median_force"]))
	best_total = min(rows, key=lambda row: float(row["last10pct_median_total"]))
	text = f"""# TF Representative Long-Training Evolution

Date: 2026-05-23

Plot:

```text
{OUTPUT_PNG}
```

Summary table:

```text
{OUTPUT_TSV}
```

Important comparability note:

This is the closest available 1/4/8/16-GPU TF long-training comparison. It is
not exactly the same design as the PT fixed-100k-none plot because this
benchmark tree only has true TF `none` 100k cases for 4GPU and 8GPU. The
1/4/8/16 curves here use the `reuse_*_10k_10x` linear-scaling family, with the
16GPU curve taken from the continuation run.

Interpretation:

- By late-window median total loss, the best TF representative curve is
  `{best_total["run"]}`.
- By late-window median energy RMSE, the best TF representative curve is
  `{best_energy["run"]}`.
- By late-window median force RMSE, the best TF representative curve is
  `{best_force["run"]}`.
- The TF curves are much less uniformly healthy than the PT fixed-100k curves:
  the 2/8/16-style large-batch TF history seen elsewhere in the validation
  tables is consistent with optimization instability rather than a simple data
  limitation.
- Use this plot as an optimizer-health diagnostic; validation RMSE remains the
  production-selection metric.
"""
	OUTPUT_MD.write_text(text, encoding="utf-8")


def main() -> None:
	"""Create TF representative training evolution plots and summaries."""
	REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
	curves = {name: load_lcurve(path / "lcurve.out") for name, path, _ in RUNS}
	plot_curves(curves)
	write_summary([
		summarize_run(name, note, curves[name])
		for name, _, note in RUNS
	])
	print(OUTPUT_PNG)
	print(OUTPUT_TSV)
	print(OUTPUT_MD)


if __name__ == "__main__":
	main()
