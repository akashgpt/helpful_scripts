#!/usr/bin/env python3
"""Plot representative TF none-scaling DeePMD training-loss evolution."""

from __future__ import annotations

import csv
import shutil
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
REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference_results"
PLOT_ROOT = SCRATCH_ROOT / "training_loss_plots_10x_20260521" / "TF"

OUTPUT_PNG = REFERENCE_ROOT / "TF_NONE_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png"
OUTPUT_TSV = REFERENCE_ROOT / "TF_NONE_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_SUMMARY_20260523.tsv"
OUTPUT_MD = REFERENCE_ROOT / "TF_NONE_REPRESENTATIVE_LONG_TRAINING_EVOLUTION_ANALYSIS_20260523.md"

RUNS: Tuple[Tuple[str, Path, str, str], ...] = (
	("1gpu baseline", SCRATCH_ROOT / "runs/long_steps_10x/reuse_1gpu_10k_10x", "1GPU baseline; worker scaling choices are equivalent", "tab:blue"),
	("4gpu none", SCRATCH_ROOT / "runs/none_100k/4gpu_100k_none", "none, 100k requested", "tab:green"),
	("8gpu none", SCRATCH_ROOT / "runs/none_100k/8gpu_100k_none", "none, 100k requested", "tab:purple"),
)

METRICS: Tuple[Tuple[str, str, int], ...] = (
	("total", "TRAIN total loss/RMSE", 1),
	("energy", "TRAIN E RMSE (eV/atom)", 2),
	("force", "TRAIN F RMSE (eV/A)", 3),
	("virial_stress", "TRAIN virial/stress RMSE (eV/atom)", 4),
)


def load_lcurve(path: Path) -> NDArray[np.float64]:
	"""Load a DeePMD ``lcurve.out`` file.

	Args:
		path: Lcurve file path.

	Returns:
		Two-dimensional numeric lcurve array.
	"""
	array = np.loadtxt(path)
	if array.ndim == 1:
		array = array.reshape(1, -1)
	return array


def choose_window(n_rows: int) -> int:
	"""Choose an odd rolling-window size.

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
	"""Compute a centered rolling statistic.

	Args:
		values: Metric values.
		window: Rolling-window size.
		use_median: If true, compute median; otherwise mean.

	Returns:
		Smoothed values.
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
		name: Run label.
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


def plot_curves(curves: Dict[str, Tuple[NDArray[np.float64], str]]) -> None:
	"""Plot rolling median and mean curves.

	Args:
		curves: Mapping from run label to lcurve and color.
	"""
	fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
	for axis, (metric_name, ylabel, column) in zip(axes.ravel(), METRICS):
		for run_name, (lcurve, color) in curves.items():
			steps = lcurve[:, 0]
			values = lcurve[:, column]
			window = choose_window(values.size)
			median_values = rolling_stat(values, window, True)
			mean_values = rolling_stat(values, window, False)
			axis.plot(steps, median_values, linewidth=1.8, color=color, label=f"{run_name} median")
			axis.plot(steps, mean_values, "--", linewidth=1.0, alpha=0.75, color=color, label=f"{run_name} mean")
			axis.axhline(median_values[-1], linestyle=":", linewidth=1.2, alpha=0.5, color=color)
			axis.axhline(mean_values[-1], linestyle="-.", linewidth=1.0, alpha=0.35, color=color)
		axis.set_title(metric_name)
		axis.set_ylabel(ylabel)
		axis.set_yscale("log")
		axis.grid(True, alpha=0.35)
		axis.legend(fontsize=7, ncol=2, framealpha=0.85)
	for axis in axes[-1, :]:
		axis.set_xlabel("training step")
	fig.suptitle("TF representative none-scaling training evolution", fontsize=14)
	fig.tight_layout()
	fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
	plt.close(fig)


def write_summary(rows: List[Dict[str, str]]) -> None:
	"""Write TSV and markdown summaries.

	Args:
		rows: Per-run summary rows.
	"""
	with OUTPUT_TSV.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)

	text = f"""# TF Representative None-Scaling Training Evolution

Date: 2026-05-23

Plot:

```text
{OUTPUT_PNG}
```

Summary table:

```text
{OUTPUT_TSV}
```

Comparability note:

This is the TF-only counterpart to the representative long-training plot, but
restricted to available `none`-schedule curves. For 1GPU, worker-scaling
choices are equivalent because the worker-count factor is 1, so the
`reuse_1gpu_10k_10x` curve is used as the shared 1GPU baseline. The available
TF `none` 100k curves with lcurve data are 4GPU and 8GPU; a 16GPU `none`
folder exists in the scratch tree, but no `lcurve.out` was available for this
plot.

Interpretation:

- The 4GPU `none` curve is the stable-looking TF production-style curve used
  in the explicit TF-vs-PT none comparison.
- The 8GPU `none` curve is included to show that `none` is not a universal
  cure at higher GPU count; the present stability observation is mainly about
  the 4GPU TF case.
- Validation RMSE remains the model-selection criterion; this plot is an
  optimizer-health diagnostic.
"""
	OUTPUT_MD.write_text(text, encoding="utf-8")


def copy_to_scratch_plot_folder() -> None:
	"""Copy generated files into the scratch TF plot folder."""
	PLOT_ROOT.mkdir(parents=True, exist_ok=True)
	for path in (OUTPUT_PNG, OUTPUT_TSV, OUTPUT_MD):
		shutil.copy2(path, PLOT_ROOT / path.name)


def main() -> None:
	"""Create TF representative none-scaling training evolution plot."""
	REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
	curves = {
		name: (load_lcurve(path / "lcurve.out"), color)
		for name, path, _, color in RUNS
	}
	plot_curves(curves)
	write_summary([
		summarize_run(name, note, curves[name][0])
		for name, _, note, _ in RUNS
	])
	copy_to_scratch_plot_folder()
	print(OUTPUT_PNG)
	print(OUTPUT_TSV)
	print(OUTPUT_MD)
	print(PLOT_ROOT)


if __name__ == "__main__":
	main()
