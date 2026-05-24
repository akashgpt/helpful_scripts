#!/usr/bin/env python3
"""Plot PT fixed-100k DeePMD training-loss evolution across GPU counts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
OUTPUT_PNG = REFERENCE_ROOT / "PT_100K_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png"
OUTPUT_TSV = REFERENCE_ROOT / "PT_100K_TRAINING_EVOLUTION_SUMMARY_20260523.tsv"
OUTPUT_MD = REFERENCE_ROOT / "PT_100K_TRAINING_EVOLUTION_ANALYSIS_20260523.md"

RUNS: Tuple[Tuple[str, Path], ...] = (
	("1gpu", SCRATCH_ROOT / "runs/pt_none_100k/1gpu_100k_none_pt"),
	("4gpu", SCRATCH_ROOT / "runs/pt_none_100k/4gpu_100k_none_pt"),
	("8gpu", SCRATCH_ROOT / "runs/pt_none_100k/8gpu_100k_none_pt"),
	("16gpu", SCRATCH_ROOT / "runs/pt_none_100k/16gpu_100k_none_pt"),
)

METRICS: Tuple[Tuple[str, str, int], ...] = (
	("total", "TRAIN total loss/RMSE", 1),
	("energy", "TRAIN E RMSE (eV/atom)", 2),
	("force", "TRAIN F RMSE (eV/A)", 3),
	("virial_stress", "TRAIN virial/stress RMSE (eV/atom)", 4),
)


def load_lcurve(path: Path) -> NDArray[np.float64]:
	"""Load numeric data from a DeePMD ``lcurve.out`` file.

	Args:
		path: Lcurve file path.

	Returns:
		Two-dimensional numeric array.
	"""
	array = np.loadtxt(path)
	if array.ndim == 1:
		array = array.reshape(1, -1)
	return array


def choose_window(n_rows: int) -> int:
	"""Choose a smoothing window for rolling statistics.

	Args:
		n_rows: Number of lcurve rows.

	Returns:
		Odd window size.
	"""
	window = min(501, max(101, n_rows // 20))
	if window % 2 == 0:
		window += 1
	return min(window, n_rows if n_rows % 2 == 1 else n_rows - 1)


def rolling_stat(values: NDArray[np.float64], window: int, use_median: bool) -> NDArray[np.float64]:
	"""Compute centered rolling mean or median.

	Args:
		values: Series to smooth.
		window: Window size.
		use_median: If true, compute median; otherwise compute mean.

	Returns:
		Smoothed series.
	"""
	if window <= 1:
		return values.copy()
	half = window // 2
	output = np.empty_like(values, dtype=float)
	for index in range(values.size):
		start = max(0, index - half)
		stop = min(values.size, index + half + 1)
		chunk = values[start:stop]
		output[index] = np.nanmedian(chunk) if use_median else np.nanmean(chunk)
	return output


def finite_min_step(steps: NDArray[np.float64], values: NDArray[np.float64]) -> Tuple[int, float]:
	"""Find the finite minimum in a metric series.

	Args:
		steps: Step values.
		values: Metric values.

	Returns:
		Step and value at the finite minimum.
	"""
	mask = np.isfinite(values)
	filtered_values = values[mask]
	filtered_steps = steps[mask]
	min_index = int(np.argmin(filtered_values))
	return int(filtered_steps[min_index]), float(filtered_values[min_index])


def summarize_run(name: str, lcurve: NDArray[np.float64]) -> Dict[str, str]:
	"""Summarize one PT training curve.

	Args:
		name: Short run label.
		lcurve: Lcurve numeric array.

	Returns:
		Summary row.
	"""
	steps = lcurve[:, 0]
	last_window_start = max(0, int(lcurve.shape[0] * 0.9))
	row: Dict[str, str] = {
		"run": name,
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
	"""Plot rolling median and mean training curves.

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
	fig.suptitle("PT fixed-100k training evolution: rolling median and mean", fontsize=14)
	fig.tight_layout()
	fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
	plt.close(fig)


def write_summary(rows: List[Dict[str, str]]) -> None:
	"""Write TSV and markdown summaries.

	Args:
		rows: Per-run summary rows.
	"""
	fieldnames = list(rows[0].keys())
	with OUTPUT_TSV.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)

	best_energy = min(rows, key=lambda row: float(row["last10pct_median_energy"]))
	best_force = min(rows, key=lambda row: float(row["last10pct_median_force"]))
	best_total = min(rows, key=lambda row: float(row["last10pct_median_total"]))
	text = f"""# PT Fixed-100k Training Evolution

Date: 2026-05-23

Plot:

```text
{OUTPUT_PNG}
```

Summary table:

```text
{OUTPUT_TSV}
```

Interpretation:

- The four PT fixed-100k runs all show a broadly decreasing smoothed training
  total loss, energy RMSE, force RMSE, and virial/stress proxy, with noisy
  minibatch-scale oscillations suppressed by rolling mean/median lines.
- By late-window median total loss, the best curve is `{best_total["run"]}`.
- By late-window median energy RMSE, the best curve is `{best_energy["run"]}`.
- By late-window median force RMSE, the best curve is `{best_force["run"]}`.
- The 4/8/16 GPU curves converge to similar late-training quality, consistent
  with the validation table where these PT 100k-none models occupy a tight
  validation-energy band.
- Training metrics alone still do not fully determine validation ranking; use
  this plot as an optimizer-health view, not as a replacement for validation
  RMSE.
"""
	OUTPUT_MD.write_text(text, encoding="utf-8")


def main() -> None:
	"""Create PT fixed-100k training evolution plots and summaries."""
	REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
	curves = {name: load_lcurve(path / "lcurve.out") for name, path in RUNS}
	plot_curves(curves)
	write_summary([summarize_run(name, lcurve) for name, lcurve in curves.items()])
	print(OUTPUT_PNG)
	print(OUTPUT_TSV)
	print(OUTPUT_MD)


if __name__ == "__main__":
	main()
