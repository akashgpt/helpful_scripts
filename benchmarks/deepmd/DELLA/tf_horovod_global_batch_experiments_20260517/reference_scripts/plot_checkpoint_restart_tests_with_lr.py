#!/usr/bin/env python3
"""Regenerate checkpoint-restart training/LR figures for the 2026-05-24 tests."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


ROOT = Path(
	"/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/"
	"v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/"
	"global_batch_experiments_20260517"
)
PLOT_DIR = ROOT / "training_loss_plots_10x_20260521" / "CHECKPOINT_RESTART_TESTS_20260524"


@dataclass(frozen=True)
class RunSpec:
	"""Metadata for one restart-test training curve."""

	label: str
	run_dir: Path
	color: str
	line_style: str = "-"


RUNS = (
	RunSpec("TF uninterrupted 4gpu_100k_none", ROOT / "runs/none_100k/4gpu_100k_none", "black", "--"),
	RunSpec("TF true 15m restart, latest ckpt", ROOT / "runs/interrupt_restart_chain_20260523/tf4g_100k_none_15min_chain", "tab:orange"),
	RunSpec("TF save_freq=10 restart chain", ROOT / "runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_savefreq10", "tab:brown"),
	RunSpec("TF second-latest ckpt restart chain", ROOT / "runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_second_latest_ckpt", "tab:green"),
	RunSpec("TF healthgate restart chain", ROOT / "runs/interrupt_restart_chain_20260524/tf4g_100k_none_15min_chain_healthgate", "tab:purple"),
	RunSpec("TF 10m healthgate + second-latest ckpt", ROOT / "runs/interrupt_restart_chain_20260524/tf4g_100k_none_10min_chain_healthgate_second_latest_ckpt", "tab:cyan"),
	RunSpec("TF final Ngpu template 15m test", ROOT / "runs/template_restart_tests_20260524/tf4g_100k_none_final_template_15min_hvd", "tab:blue"),
	RunSpec("PT true 10m restart chain", ROOT / "runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain", "tab:red"),
	RunSpec("PT old healthgate v1 (selector bug)", ROOT / "runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt", "tab:pink"),
	RunSpec("PT healthgate + second-latest v2", ROOT / "runs/interrupt_restart_chain_20260524/pt4g_100k_none_10min_chain_healthgate_second_latest_ckpt_v2", "crimson"),
)

METRICS = (
	("total", "TRAIN total loss/RMSE", 1, True),
	("energy", "TRAIN E RMSE (eV/atom)", 2, True),
	("force", "TRAIN F RMSE (eV/A)", 3, True),
	("virial", "TRAIN virial/stress RMSE", 4, True),
	("lr", "learning rate", 5, True),
)


def load_lcurve(path: Path) -> NDArray[np.float64]:
	"""Load numeric lcurve rows.

	Args:
		path: Path to ``lcurve.out``.

	Returns:
		Numeric lcurve array.
	"""
	array = np.loadtxt(path)
	if array.ndim == 1:
		array = array.reshape(1, -1)
	return array


def rolling_mean(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
	"""Compute a centered rolling mean.

	Args:
		values: Input values.
		window: Rolling window length.

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
		output[index] = np.nanmean(values[start:stop])
	return output


def choose_window(n_rows: int) -> int:
	"""Choose a stable rolling window for a curve length.

	Args:
		n_rows: Number of lcurve rows.

	Returns:
		Odd rolling-window length.
	"""
	window = min(401, max(51, n_rows // 30))
	if window % 2 == 0:
		window += 1
	return max(1, min(window, n_rows if n_rows % 2 else n_rows - 1))


def restart_steps(run_dir: Path) -> list[int]:
	"""Parse restart step markers from ``CHAIN_HISTORY.tsv``.

	Args:
		run_dir: Run directory.

	Returns:
		Sorted restart-boundary steps.
	"""
	history = run_dir / "CHAIN_HISTORY.tsv"
	if not history.exists():
		return []
	steps = []
	for line in history.read_text(encoding="utf-8", errors="replace").splitlines():
		match = re.search(r"\bstep=([0-9]+)\b", line)
		if match:
			steps.append(int(match.group(1)))
	return sorted(set(steps))


def summarize(spec: RunSpec, lcurve: NDArray[np.float64]) -> dict[str, str]:
	"""Summarize one plotted run.

	Args:
		spec: Run metadata.
		lcurve: Numeric lcurve data.

	Returns:
		Summary row for TSV output.
	"""
	steps = lcurve[:, 0]
	row = {
		"label": spec.label,
		"status": "plotted",
		"final_step": str(int(steps[-1])),
		"restart_marker_steps": ",".join(str(step) for step in restart_steps(spec.run_dir)),
		"lcurve": str(spec.run_dir / "lcurve.out"),
	}
	for name, _, column, _ in METRICS:
		values = lcurve[:, column]
		row[f"final_{name}"] = f"{values[-1]:.8g}"
		row[f"min_{name}"] = f"{np.nanmin(values):.8g}"
		row[f"min_{name}_step"] = str(int(steps[int(np.nanargmin(values))]))
	return row


def plot(curves: dict[RunSpec, NDArray[np.float64]], output: Path, xlim: tuple[int, int] | None = None) -> None:
	"""Plot restart-test curves.

	Args:
		curves: Curves to plot.
		output: Output PNG path.
		xlim: Optional x-axis limits.
	"""
	fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
	axes_flat = axes.ravel()
	for axis, (name, ylabel, column, log_scale) in zip(axes_flat, METRICS):
		for spec, lcurve in curves.items():
			steps = lcurve[:, 0]
			values = lcurve[:, column]
			window = choose_window(values.size)
			smoothed = rolling_mean(values, window)
			axis.plot(steps, smoothed, spec.line_style, color=spec.color, linewidth=1.55, label=spec.label)
			for step in restart_steps(spec.run_dir):
				if steps[0] <= step <= steps[-1]:
					axis.axvline(step, color=spec.color, linewidth=1.0, alpha=0.75)
		axis.set_title(name)
		axis.set_ylabel(ylabel)
		if log_scale:
			axis.set_yscale("log")
		if xlim is not None:
			axis.set_xlim(*xlim)
		axis.grid(True, alpha=0.3)
		axis.legend(fontsize=6.4, framealpha=0.82)
	axes_flat[-1].axis("off")
	for axis in axes[-1, :]:
		axis.set_xlabel("training step")
	title_suffix = " first 20k steps" if xlim is not None else ""
	fig.suptitle(f"Checkpoint restart tests: rolling mean training metrics and LR{title_suffix}", fontsize=15)
	fig.tight_layout()
	fig.savefig(output, dpi=180, bbox_inches="tight")
	plt.close(fig)


def write_summary(rows: Iterable[dict[str, str]], output: Path) -> None:
	"""Write plotted-run summary table.

	Args:
		rows: Summary rows.
		output: Output TSV path.
	"""
	rows = list(rows)
	with output.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	"""Regenerate full and first-20k checkpoint-restart figures."""
	PLOT_DIR.mkdir(parents=True, exist_ok=True)
	curves = {}
	rows = []
	for spec in RUNS:
		lcurve_path = spec.run_dir / "lcurve.out"
		if not lcurve_path.exists():
			continue
		lcurve = load_lcurve(lcurve_path)
		curves[spec] = lcurve
		rows.append(summarize(spec, lcurve))
	plot(curves, PLOT_DIR / "CHECKPOINT_RESTART_TESTS_TRAINING_EVOLUTION_ROLLING_MEAN_20260524.png")
	plot(curves, PLOT_DIR / "CHECKPOINT_RESTART_TESTS_TRAINING_EVOLUTION_ROLLING_MEAN_FIRST20K_20260524.png", (0, 20000))
	write_summary(rows, PLOT_DIR / "CHECKPOINT_RESTART_TESTS_PLOTTED_20260524.tsv")


if __name__ == "__main__":
	main()
