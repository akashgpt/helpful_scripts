#!/usr/bin/env python3
"""Plot sliding-window VASP time-step speed from OUTCAR timing lines."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.getuid()}"))

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class TimingSeries:
	"""Store per-step wall-time data parsed from one VASP OUTCAR."""

	label: str
	step_numbers: np.ndarray
	step_seconds: np.ndarray

	@property
	def cumulative_hours(self) -> np.ndarray:
		"""Return cumulative VASP loop wall time in hours."""
		return np.cumsum(self.step_seconds) / 3600.0


@dataclass(frozen=True)
class RunMetadata:
	"""Store compact VASP/MLFF settings for plot labels."""

	ml_mb: str
	ml_mconf: str


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description=(
			"Plot the sliding-window wall time for VASP ionic steps using OUTCAR "
			"LOOP+ timing lines."
		),
	)
	parser.add_argument(
		"paths",
		nargs="+",
		type=Path,
		help="Run directories or OUTCAR files to plot.",
	)
	parser.add_argument(
		"--labels",
		nargs="*",
		default=None,
		help="Optional labels matching the input paths.",
	)
	parser.add_argument(
		"--window",
		type=int,
		default=10,
		help="Sliding-window size in ionic steps.",
	)
	parser.add_argument(
		"--skip-initial-steps",
		type=int,
		default=1,
		help=(
			"Number of initial timing entries to exclude before computing windows. "
			"The default skips the first VASP step to avoid initialization/refit overhead."
		),
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("analysis/step_speed_10step_sliding.png"),
		help="Output PNG path.",
	)
	parser.add_argument(
		"--title",
		default="VASP step speed",
		help="Plot title.",
	)
	return parser.parse_args()


def resolve_outcar(path: Path) -> Path:
	"""Resolve an input path to an OUTCAR file.

	Args:
		path: Either a run directory or an OUTCAR path.

	Returns:
		Path to OUTCAR.

	Raises:
		FileNotFoundError: If no OUTCAR can be found.
	"""
	if path.is_dir():
		outcar_path = path / "OUTCAR"
	else:
		outcar_path = path
	if not outcar_path.is_file():
		raise FileNotFoundError(f"OUTCAR not found: {outcar_path}")
	return outcar_path


def read_first_tag_value(paths: list[Path], tag: str) -> str:
	"""Read the first value for a VASP tag from OUTCAR, ML_LOGFILE, or INCAR-like files.

	Args:
		paths: Files to scan in priority order.
		tag: VASP tag name, such as ``ML_MB``.

	Returns:
		Parsed value, or ``NA`` if the tag is not present.
	"""
	assignment_pattern = re.compile(rf"\b{re.escape(tag)}\s*=\s*([^\s!#;]+)")
	logfile_pattern = re.compile(rf":\s*([^\s]+)(?:\s+\([^)]*\))?\s+{re.escape(tag)}\b")
	for path in paths:
		if not path.is_file():
			continue
		with path.open(errors="ignore") as handle:
			for line in handle:
				assignment_match = assignment_pattern.search(line)
				if assignment_match:
					return assignment_match.group(1)
				logfile_match = logfile_pattern.search(line)
				if logfile_match:
					return logfile_match.group(1)
	return "NA"


def parse_run_metadata(outcar_path: Path) -> RunMetadata:
	"""Parse compact run metadata for annotating timing plots.

	Args:
		outcar_path: OUTCAR path.

	Returns:
		MLFF settings suitable for plot labels. Missing tags are ``NA``.
	"""
	run_dir = outcar_path.parent
	search_paths = [outcar_path, run_dir / "ML_LOGFILE", run_dir / "INCAR"]
	return RunMetadata(
		ml_mb=read_first_tag_value(search_paths, "ML_MB"),
		ml_mconf=read_first_tag_value(search_paths, "ML_MCONF"),
	)


def make_visible_label(label: str, metadata: RunMetadata) -> str:
	"""Build a Matplotlib-visible legend label with MLFF metadata.

	Args:
		label: User-facing run label.
		metadata: Parsed run metadata.

	Returns:
		Legend label. Leading underscores are prefixed so Matplotlib does not hide them.
	"""
	visible_label = label if not label.startswith("_") else f"run {label}"
	return f"{visible_label} | ML_MB={metadata.ml_mb}, ML_MCONF={metadata.ml_mconf}"


def parse_loop_seconds(outcar_path: Path, label: str) -> TimingSeries:
	"""Parse per-ionic-step real times from VASP OUTCAR timing lines.

	Args:
		outcar_path: Path to OUTCAR.
		label: Plot label for this series.

	Returns:
		Timing series with one wall time per parsed VASP step.

	Raises:
		ValueError: If no timing lines are present.
	"""
	loop_plus_pattern = re.compile(
		r"LOOP\+:\s+cpu time\s+[-+0-9.Ee]+:\s+real time\s+([-+0-9.Ee]+)"
	)
	loop_pattern = re.compile(
		r"\bLOOP:\s+cpu time\s+[-+0-9.Ee]+:\s+real time\s+([-+0-9.Ee]+)"
	)

	loop_plus_seconds: list[float] = []
	loop_seconds: list[float] = []
	with outcar_path.open(errors="ignore") as handle:
		for line in handle:
			loop_plus_match = loop_plus_pattern.search(line)
			if loop_plus_match:
				loop_plus_seconds.append(float(loop_plus_match.group(1)))
				continue
			loop_match = loop_pattern.search(line)
			if loop_match:
				loop_seconds.append(float(loop_match.group(1)))

	step_seconds = loop_plus_seconds if loop_plus_seconds else loop_seconds
	if not step_seconds:
		raise ValueError(f"No LOOP+/LOOP timing lines found in {outcar_path}")

	steps = np.arange(1, len(step_seconds) + 1, dtype=int)
	return TimingSeries(
		label=label,
		step_numbers=steps,
		step_seconds=np.asarray(step_seconds, dtype=float),
	)


def sliding_window_sum(values: np.ndarray, window: int) -> np.ndarray:
	"""Return sliding-window sums for a one-dimensional array.

	Args:
		values: Per-step values.
		window: Window size.

	Returns:
		Array of sums over each full sliding window.

	Raises:
		ValueError: If the window is invalid for the data length.
	"""
	if window <= 0:
		raise ValueError("Window size must be positive.")
	if len(values) < window:
		raise ValueError(
			f"Need at least {window} steps for a {window}-step sliding window; got {len(values)}."
		)
	kernel = np.ones(window, dtype=float)
	return np.convolve(values, kernel, mode="valid")


def trim_initial_steps(series: TimingSeries, skip_initial_steps: int) -> TimingSeries:
	"""Drop early timing entries before computing sliding windows.

	Args:
		series: Full timing series.
		skip_initial_steps: Number of initial entries to drop.

	Returns:
		Trimmed timing series. Step numbers retain their original OUTCAR order.

	Raises:
		ValueError: If all timing entries would be removed.
	"""
	if skip_initial_steps < 0:
		raise ValueError("--skip-initial-steps must be non-negative.")
	if skip_initial_steps == 0:
		return series
	if skip_initial_steps >= len(series.step_seconds):
		raise ValueError(
			f"Cannot skip {skip_initial_steps} steps from {series.label}; "
			f"only {len(series.step_seconds)} timing entries were parsed."
		)
	return TimingSeries(
		label=series.label,
		step_numbers=series.step_numbers[skip_initial_steps:],
		step_seconds=series.step_seconds[skip_initial_steps:],
	)


def print_summary(series: TimingSeries, window: int) -> None:
	"""Print a concise timing summary for one series."""
	window_seconds = sliding_window_sum(series.step_seconds, window)
	print(f"== {series.label} ==")
	print(f"steps={len(series.step_seconds)}")
	print(f"total_loop_wall_hours={series.cumulative_hours[-1]:.3f}")
	print(f"last_{window}_steps_seconds={window_seconds[-1]:.2f}")
	print(f"median_{window}_step_window_seconds={np.median(window_seconds):.2f}")
	print(f"p90_{window}_step_window_seconds={np.percentile(window_seconds, 90):.2f}")
	print(f"last_step_seconds={series.step_seconds[-1]:.2f}")


def plot_timing_series(series_list: list[TimingSeries], window: int, output: Path, title: str) -> None:
	"""Plot sliding-window wall time for several timing series.

	Args:
		series_list: Parsed timing series.
		window: Number of steps per sliding window.
		output: Output PNG path.
		title: Plot title.
	"""
	fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=False)
	colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(series_list), 1)))

	for color, series in zip(colors, series_list):
		window_seconds = sliding_window_sum(series.step_seconds, window)
		window_end_indices = np.arange(window - 1, len(series.step_seconds), dtype=int)
		window_end_hours = series.cumulative_hours[window_end_indices]
		window_end_steps = series.step_numbers[window_end_indices]
		steps_per_hour = window * 3600.0 / window_seconds

		axes[0].plot(
			window_end_hours,
			window_seconds,
			lw=1.7,
			color=color,
			label=series.label,
		)
		axes[1].plot(
			window_end_steps,
			steps_per_hour,
			lw=1.7,
			color=color,
			label=series.label,
		)

	axes[0].set_title(f"{title}: sliding {window}-step wall time")
	axes[0].set_xlabel("cumulative VASP loop wall time (hours)")
	axes[0].set_ylabel(f"seconds per {window} steps")
	axes[0].set_yscale("symlog", linthresh=10.0)
	axes[0].grid(True, alpha=0.25)
	axes[0].legend(frameon=False)

	axes[1].set_title("Equivalent instantaneous speed")
	axes[1].set_xlabel("ionic step")
	axes[1].set_ylabel("steps / hour")
	axes[1].grid(True, alpha=0.25)
	axes[1].legend(frameon=False)

	fig.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=220)
	plt.close(fig)


def main() -> None:
	"""Run the command-line interface."""
	args = parse_args()
	if args.labels is not None and len(args.labels) != len(args.paths):
		raise ValueError("--labels must have the same number of entries as paths.")

	series_list: list[TimingSeries] = []
	for index, path in enumerate(args.paths):
		outcar_path = resolve_outcar(path)
		base_label = args.labels[index] if args.labels else outcar_path.parent.name
		label = make_visible_label(base_label, parse_run_metadata(outcar_path))
		series = parse_loop_seconds(outcar_path, label)
		series = trim_initial_steps(series, args.skip_initial_steps)
		series_list.append(series)
		print_summary(series, args.window)

	plot_timing_series(series_list, args.window, args.output, args.title)
	print(f"Wrote {args.output}")


if __name__ == "__main__":
	main()
