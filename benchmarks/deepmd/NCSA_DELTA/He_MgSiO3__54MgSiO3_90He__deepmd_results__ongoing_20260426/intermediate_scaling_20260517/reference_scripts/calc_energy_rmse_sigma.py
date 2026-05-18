#!/usr/bin/env python3
"""Compute bootstrap uncertainty for held-out DeePMD energy RMSE values."""

from __future__ import annotations

import math
import random
from pathlib import Path


VALIDATION_ROOT = Path(
	"/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/"
	"deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/"
	"He_MgSiO3__54MgSiO3_90He/training_bench/"
	"validation_71MgSiO3_5He__v1_i_train_test__20260517"
)

MODEL_INFO = {
	"balanced_10x": ("26.80M", "46h 07m"),
	"big": ("23.91M", "22h 53m"),
	"fit_deep2x": ("5.28M", "12h 47m"),
	"fit_deep10x": ("26.77M", "28h 52m"),
	"base": ("2.67M", "10h 58m"),
}


def read_test_residual_groups(model_name: str) -> dict[str, list[float]]:
	"""Read frame-level per-atom energy residuals grouped by test system.

	Args:
		model_name: Short model identifier under the validation results folder.

	Returns:
		Signed residuals in eV/atom by system id, computed as predicted minus reference.
	"""
	residual_groups: dict[str, list[float]] = {}
	model_root = VALIDATION_ROOT / "results" / model_name
	for path in sorted(model_root.glob("v1_i2_*_test/dp_test.e_peratom.out")):
		system_id = path.parent.name
		residual_groups[system_id] = []
		with path.open("r", encoding="utf-8") as handle:
			for line in handle:
				if not line.strip() or line.startswith("#"):
					continue
				fields = line.split()
				reference = float(fields[0])
				predicted = float(fields[1])
				residual_groups[system_id].append(predicted - reference)
	return residual_groups


def rmse(values: list[float]) -> float:
	"""Return the root mean square of a list of signed errors."""
	return math.sqrt(sum(value * value for value in values) / len(values))


def weighted_system_rmse(residual_groups: dict[str, list[float]]) -> float:
	"""Return nframe-weighted mean of system-level RMSEs."""
	n_total = sum(len(values) for values in residual_groups.values())
	return sum(len(values) * rmse(values) for values in residual_groups.values()) / n_total


def percentile(values: list[float], percentile_value: float) -> float:
	"""Return a linearly interpolated percentile from sorted or unsorted values.

	Args:
		values: Numeric values.
		percentile_value: Percentile in the inclusive range [0, 100].

	Returns:
		Interpolated percentile value.
	"""
	sorted_values = sorted(values)
	position = (len(sorted_values) - 1) * percentile_value / 100.0
	lower_index = math.floor(position)
	upper_index = math.ceil(position)
	if lower_index == upper_index:
		return sorted_values[int(position)]
	lower_value = sorted_values[lower_index]
	upper_value = sorted_values[upper_index]
	return lower_value + (upper_value - lower_value) * (position - lower_index)


def bootstrap_weighted_system_rmse_intervals(
	residual_groups: dict[str, list[float]],
	n_bootstrap: int = 50_000,
	seed: int = 1729,
) -> tuple[float, float, float, float]:
	"""Estimate asymmetric bootstrap intervals for weighted system RMSE.

	Args:
		residual_groups: Signed frame-level errors grouped by system.
		n_bootstrap: Number of bootstrap resamples.
		seed: Random seed for deterministic output.

	Returns:
		Lower/upper percentile bounds for 1 sigma and 3 sigma intervals.
	"""
	random_generator = random.Random(seed)
	n_total = sum(len(values) for values in residual_groups.values())
	bootstrap_values: list[float] = []
	for _ in range(n_bootstrap):
		weighted_sum = 0.0
		for residuals in residual_groups.values():
			n_values = len(residuals)
			squared_sum = 0.0
			for _ in range(n_values):
				value = residuals[random_generator.randrange(n_values)]
				squared_sum += value * value
			weighted_sum += n_values * math.sqrt(squared_sum / n_values)
		bootstrap_values.append(weighted_sum / n_total)
	return (
		percentile(bootstrap_values, 15.865525393145708),
		percentile(bootstrap_values, 84.1344746068543),
		percentile(bootstrap_values, 0.13498980316300932),
		percentile(bootstrap_values, 99.86501019683699),
	)


def main() -> None:
	"""Print ranked energy RMSE table with bootstrap uncertainty."""
	rows: list[tuple[float, str, str, float, float, float, float, str, int]] = []
	for model_name, (param_count, gpu_time) in MODEL_INFO.items():
		residual_groups = read_test_residual_groups(model_name)
		rmse_value = weighted_system_rmse(residual_groups)
		low_1sigma, high_1sigma, low_3sigma, high_3sigma = bootstrap_weighted_system_rmse_intervals(
			residual_groups
		)
		nframes = sum(len(values) for values in residual_groups.values())
		rows.append(
			(
				rmse_value,
				model_name,
				param_count,
				rmse_value - low_1sigma,
				high_1sigma - rmse_value,
				rmse_value - low_3sigma,
				high_3sigma - rmse_value,
				gpu_time,
				nframes,
			)
		)

	print("rank\tmodel\tparams\tE RMSE/atom\t-1sigma\t+1sigma\t-3sigma\t+3sigma\ttraining GPU time\tnframes")
	for rank, row in enumerate(sorted(rows), start=1):
		rmse_value, model_name, param_count, low_1, high_1, low_2, high_2, gpu_time, nframes = row
		print(
			f"{rank}\t{model_name}\t{param_count}\t{rmse_value:.6f}\t{low_1:.6f}\t"
			f"{high_1:.6f}\t{low_2:.6f}\t{high_2:.6f}\t{gpu_time}\t{nframes}"
		)


if __name__ == "__main__":
	main()
