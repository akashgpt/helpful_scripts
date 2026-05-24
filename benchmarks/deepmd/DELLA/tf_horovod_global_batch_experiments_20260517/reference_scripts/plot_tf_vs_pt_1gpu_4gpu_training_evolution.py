#!/usr/bin/env python3
"""Plot 1GPU/4GPU TF-vs-PT training-loss evolution."""

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
PLOT_ROOT = SCRATCH_ROOT / "training_loss_plots_10x_20260521" / "TF_vs_PT"

OUTPUT_PNG = REFERENCE_ROOT / "TF_VS_PT_1GPU_4GPU_TRAINING_EVOLUTION_ROLLING_MEDIAN_MEAN_20260523.png"
OUTPUT_TSV = REFERENCE_ROOT / "TF_VS_PT_1GPU_4GPU_TRAINING_EVOLUTION_SUMMARY_20260523.tsv"
OUTPUT_MD = REFERENCE_ROOT / "TF_VS_PT_1GPU_4GPU_TRAINING_EVOLUTION_ANALYSIS_20260523.md"

RUNS: Tuple[Tuple[str, Path, str, str, str], ...] = (
	("TF", "1gpu", SCRATCH_ROOT / "runs/long_steps_10x/reuse_1gpu_10k_10x", "linear, 100k requested", "tab:blue"),
	("TF", "4gpu", SCRATCH_ROOT / "runs/none_100k/4gpu_100k_none", "none, 100k requested", "tab:cyan"),
	("PT", "1gpu", SCRATCH_ROOT / "runs/pt_none_100k/1gpu_100k_none_pt", "none, 100k requested", "tab:orange"),
	("PT", "4gpu", SCRATCH_ROOT / "runs/pt_none_100k/4gpu_100k_none_pt", "none, 100k requested", "tab:red"),
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
		path: Lcurve path.

	Returns:
		Two-dimensional numeric lcurve array.
	"""
	array = np.loadtxt(path)
	if array.ndim == 1:
		array = array.reshape(1, -1)
	return array


def choose_window(n_rows: int) -> int:
	"""Choose an odd rolling window.

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
	"""Compute centered rolling median or mean.

	Args:
		values: Metric values.
		window: Rolling window size.
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
	"""Find the finite minimum of a metric.

	Args:
		steps: Training steps.
		values: Metric values.

	Returns:
		Step and value at the finite minimum.
	"""
	mask = np.isfinite(values)
	filtered_values = values[mask]
	filtered_steps = steps[mask]
	min_index = int(np.argmin(filtered_values))
	return int(filtered_steps[min_index]), float(filtered_values[min_index])


def run_label(framework: str, gpu_label: str) -> str:
	"""Create a compact legend label.

	Args:
		framework: Framework name.
		gpu_label: GPU label.

	Returns:
		Legend label.
	"""
	return f"{framework} {gpu_label}"


def summarize_run(framework: str, gpu_label: str, note: str, lcurve: NDArray[np.float64]) -> Dict[str, str]:
	"""Summarize one training curve.

	Args:
		framework: TF or PT.
		gpu_label: GPU label.
		note: Schedule note.
		lcurve: Numeric lcurve array.

	Returns:
		Summary row.
	"""
	steps = lcurve[:, 0]
	last_window_start = max(0, int(lcurve.shape[0] * 0.9))
	row: Dict[str, str] = {
		"framework": framework,
		"run": gpu_label,
		"label": run_label(framework, gpu_label),
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
	"""Plot rolling median/mean curves for all runs.

	Args:
		curves: Mapping from label to lcurve and color.
	"""
	fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
	for axis, (metric_name, ylabel, column) in zip(axes.ravel(), METRICS):
		for label, (lcurve, color) in curves.items():
			steps = lcurve[:, 0]
			values = lcurve[:, column]
			window = choose_window(values.size)
			axis.plot(steps, rolling_stat(values, window, True), "-", color=color, linewidth=1.8, label=f"{label} median")
			axis.plot(steps, rolling_stat(values, window, False), "--", color=color, linewidth=1.0, alpha=0.75, label=f"{label} mean")
		axis.set_title(metric_name)
		axis.set_ylabel(ylabel)
		axis.set_yscale("log")
		axis.grid(True, alpha=0.35)
		axis.legend(fontsize=7, ncol=2, framealpha=0.85)
	for axis in axes[-1, :]:
		axis.set_xlabel("training step")
	fig.suptitle("TF vs PT training evolution: 1GPU and 4GPU", fontsize=14)
	fig.tight_layout()
	fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
	plt.close(fig)


def write_summary(rows: List[Dict[str, str]]) -> None:
	"""Write TSV and markdown analysis outputs.

	Args:
		rows: Summary rows.
	"""
	with OUTPUT_TSV.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)

	best_total = min(rows, key=lambda row: float(row["last10pct_median_total"]))
	best_energy = min(rows, key=lambda row: float(row["last10pct_median_energy"]))
	best_force = min(rows, key=lambda row: float(row["last10pct_median_force"]))
	text = f"""# TF vs PT 1GPU/4GPU Training Evolution

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

The PT 1GPU/4GPU curves are fixed-100k `none` runs. The TF 4GPU curve is the
fixed-100k `none` run, but the available TF 1GPU long curve is the
`reuse_1gpu_10k_10x` linear-scaling run. This plot is therefore best read as a
framework optimizer-health comparison, not as a perfectly controlled schedule
comparison.

Interpretation:

- Best late-window median total loss: `{best_total["label"]}`.
- Best late-window median energy RMSE: `{best_energy["label"]}`.
- Best late-window median force RMSE: `{best_force["label"]}`.
- TF 4GPU and PT 4GPU are the cleanest practical comparison here; both are
  100k-step `none` cases and both train smoothly, with PT 4GPU having a lower
  late-window median force while TF 4GPU has the best validation force/energy
  among these particular rows.
- The TF 1GPU curve is reasonable but schedule-mismatched relative to PT 1GPU,
  so avoid over-reading TF-vs-PT 1GPU differences from this plot alone.
"""
	OUTPUT_MD.write_text(text, encoding="utf-8")


def copy_to_scratch_plot_folder() -> None:
	"""Copy aggregate outputs into the scratch plot folder."""
	PLOT_ROOT.mkdir(parents=True, exist_ok=True)
	for path in (OUTPUT_PNG, OUTPUT_TSV, OUTPUT_MD):
		(PLOT_ROOT / path.name).write_bytes(path.read_bytes())


def main() -> None:
	"""Create TF-vs-PT 1GPU/4GPU training evolution plot and summaries."""
	REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
	curves: Dict[str, Tuple[NDArray[np.float64], str]] = {}
	rows: List[Dict[str, str]] = []
	for framework, gpu_label, run_dir, note, color in RUNS:
		lcurve = load_lcurve(run_dir / "lcurve.out")
		label = run_label(framework, gpu_label)
		curves[label] = (lcurve, color)
		rows.append(summarize_run(framework, gpu_label, note, lcurve))
	plot_curves(curves)
	write_summary(rows)
	copy_to_scratch_plot_folder()
	print(OUTPUT_PNG)
	print(OUTPUT_TSV)
	print(OUTPUT_MD)
	print(PLOT_ROOT)


if __name__ == "__main__":
	main()
