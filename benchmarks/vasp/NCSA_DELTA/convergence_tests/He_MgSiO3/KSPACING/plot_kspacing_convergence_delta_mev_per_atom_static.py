"""Plot static KSPACING convergence with k-point mesh and runtime panels."""

from __future__ import annotations

import csv
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BENCHMARK_DIR = Path(os.environ.get("KSPACING_BENCHMARK_DIR", Path(__file__).resolve().parent))
INPUT_CSV = BENCHMARK_DIR / "kspacing_convergence_delta_mev_per_atom_static.csv"
OUTPUT_PNG = Path(
	os.environ.get(
		"KSPACING_OUTPUT_PNG",
		BENCHMARK_DIR / "kspacing_convergence_delta_mev_per_atom_static.png",
	)
)

KPOINT_MESH_PATTERN = re.compile(r"generate k-points for:\s*(?P<nkx>\d+)\s+(?P<nky>\d+)\s+(?P<nkz>\d+)")
ELAPSED_PATTERN = re.compile(r"Elapsed time \(sec\):\s*(?P<seconds>[0-9.]+)")
RESOURCE_PATTERN = re.compile(
	r"running\s+(?P<mpi_ranks>\d+)\s+mpi-ranks,\s+with\s+(?P<threads>\d+)\s+threads/rank"
)
GPU_PATTERN = re.compile(r"Offloading initialized\s+\.\.\.\s+(?P<gpus>\d+)\s+GPUs detected")
PARTITION_PATTERN = re.compile(r"#SBATCH\s+--partition=(?P<partition>\S+)")
ACCELERATOR_MARKERS = {"A100": "^", "H200": "*", "unknown": "o"}


@dataclass(frozen=True)
class StaticKspacingRow:
	"""One completed static KSPACING convergence calculation."""

	family: str
	config: str
	run: str
	kspacing: float
	nkpts: int
	delta_toten_mev_per_atom: float
	delta_internal_mev_per_atom: float
	kpoint_mesh: tuple[int, int, int]
	elapsed_seconds: float
	mpi_ranks: int | None
	threads_per_rank: int | None
	gpus_detected: int | None
	accelerator: str

	@property
	def label(self) -> str:
		"""Return the compact legend label used in the convergence plots."""
		family_label = self.family.replace("_atoms__and_He", " atoms")
		return f"{family_label}, {self.config}"


def read_outcar_metadata(outcar_path: Path) -> tuple[tuple[int, int, int], float, int | None, int | None, int | None]:
	"""Read VASP mesh, elapsed time, and resource metadata from OUTCAR."""
	kpoint_mesh: tuple[int, int, int] | None = None
	elapsed_seconds: float | None = None
	mpi_ranks: int | None = None
	threads_per_rank: int | None = None
	gpus_detected: int | None = None
	with outcar_path.open("r", encoding="utf-8", errors="replace") as handle:
		for line in handle:
			if kpoint_mesh is None and (match := KPOINT_MESH_PATTERN.search(line)):
				kpoint_mesh = (
					int(match.group("nkx")),
					int(match.group("nky")),
					int(match.group("nkz")),
				)
			if elapsed_seconds is None and (match := ELAPSED_PATTERN.search(line)):
				elapsed_seconds = float(match.group("seconds"))
			if mpi_ranks is None and (match := RESOURCE_PATTERN.search(line)):
				mpi_ranks = int(match.group("mpi_ranks"))
				threads_per_rank = int(match.group("threads"))
			if gpus_detected is None and (match := GPU_PATTERN.search(line)):
				gpus_detected = int(match.group("gpus"))
	if kpoint_mesh is None:
		raise ValueError(f"No k-point mesh line found in {outcar_path}")
	if elapsed_seconds is None:
		raise ValueError(f"No elapsed-time line found in {outcar_path}")
	return kpoint_mesh, elapsed_seconds, mpi_ranks, threads_per_rank, gpus_detected


def read_accelerator(run_path: Path) -> str:
	"""Read accelerator class from the local submission script."""
	for script_path in sorted(run_path.glob("sub*.sh")):
		with script_path.open("r", encoding="utf-8", errors="replace") as handle:
			for line in handle:
				match = PARTITION_PATTERN.search(line)
				if match is None:
					continue
				partition = match.group("partition")
				if "H200" in partition:
					return "H200"
				if "A100" in partition:
					return "A100"
	return "unknown"


def load_static_kspacing_rows(csv_path: Path) -> list[StaticKspacingRow]:
	"""Load static KSPACING convergence rows from the saved CSV."""
	rows: list[StaticKspacingRow] = []
	with csv_path.open("r", encoding="utf-8", newline="") as handle:
		for raw_row in csv.DictReader(handle):
			run_path = BENCHMARK_DIR / raw_row["family"] / raw_row["config"] / raw_row["run"]
			kpoint_mesh, elapsed_seconds, mpi_ranks, threads_per_rank, gpus_detected = read_outcar_metadata(
				run_path / "OUTCAR"
			)
			rows.append(
				StaticKspacingRow(
					family=raw_row["family"],
					config=raw_row["config"],
					run=raw_row["run"],
					kspacing=float(raw_row["kspacing"]),
					nkpts=int(raw_row["nkpts"]),
					delta_toten_mev_per_atom=float(raw_row["delta_toten_meV_per_atom"]),
					delta_internal_mev_per_atom=float(raw_row["delta_internal_meV_per_atom"]),
					kpoint_mesh=kpoint_mesh,
					elapsed_seconds=elapsed_seconds,
					mpi_ranks=mpi_ranks,
					threads_per_rank=threads_per_rank,
					gpus_detected=gpus_detected,
					accelerator=read_accelerator(run_path),
				)
			)
	return sorted(rows, key=lambda item: (item.family, item.config, -item.kspacing))


def group_static_rows(rows: Iterable[StaticKspacingRow]) -> OrderedDict[tuple[str, str], list[StaticKspacingRow]]:
	"""Group static rows by simulation family and composition."""
	grouped_rows: OrderedDict[tuple[str, str], list[StaticKspacingRow]] = OrderedDict()
	for row in rows:
		grouped_rows.setdefault((row.family, row.config), []).append(row)
	return grouped_rows


def configure_energy_axis(axis: plt.Axes, title: str) -> None:
	"""Apply shared styling to a static convergence energy axis."""
	axis.set_title(title, fontsize=14)
	axis.set_xlabel("KSPACING", fontsize=11)
	axis.set_ylabel("meV/atom", fontsize=11)
	axis.set_yscale("symlog", linthresh=1.0e-5)
	axis.set_ylim(bottom=0.0)
	axis.grid(True, which="both", alpha=0.25)
	axis.tick_params(axis="both", labelsize=10)
	axis.invert_xaxis()


def add_energy_panels(
	axes: tuple[plt.Axes, plt.Axes],
	grouped_rows: OrderedDict[tuple[str, str], list[StaticKspacingRow]],
) -> None:
	"""Add the two top static KSPACING convergence panels."""
	toten_axis, internal_axis = axes
	for rows in grouped_rows.values():
		ordered_rows = sorted(rows, key=lambda item: item.kspacing, reverse=True)
		kspacing_values = [row.kspacing for row in ordered_rows]
		label = ordered_rows[0].label
		toten_axis.plot(
			kspacing_values,
			[row.delta_toten_mev_per_atom for row in ordered_rows],
			marker="o",
			linewidth=1.8,
			markersize=6.5,
			label=label,
		)
		internal_axis.plot(
			kspacing_values,
			[row.delta_internal_mev_per_atom for row in ordered_rows],
			marker="s",
			linewidth=1.8,
			markersize=6.5,
			label=label,
		)
	configure_energy_axis(toten_axis, r"$|\Delta$ TOTEN| vs smallest valid KSPACING")
	configure_energy_axis(internal_axis, r"$|\Delta$ internal energy| vs smallest valid KSPACING")
	toten_axis.legend(fontsize=8, loc="upper right")
	internal_axis.legend(fontsize=8, loc="upper right")


def add_kpoint_panel(
	axis: plt.Axes,
	grouped_rows: OrderedDict[tuple[str, str], list[StaticKspacingRow]],
) -> None:
	"""Add the k-point mesh panel with staggered overlapping points."""
	unique_meshes = sorted(
		{row.kpoint_mesh for rows in grouped_rows.values() for row in rows},
		key=lambda mesh: (mesh[0] * mesh[1] * mesh[2], mesh),
	)
	mesh_positions = {mesh: index for index, mesh in enumerate(unique_meshes)}
	mesh_labels = [f"{mesh[0]}x{mesh[1]}x{mesh[2]}" for mesh in unique_meshes]
	mesh_nkpts = [mesh[0] * mesh[1] * mesh[2] for mesh in unique_meshes]
	collision_counts: dict[tuple[float, int], int] = {}
	collision_ranks: dict[tuple[float, int], int] = {}
	for rows in grouped_rows.values():
		for row in rows:
			key = (row.kspacing, mesh_positions[row.kpoint_mesh])
			collision_counts[key] = collision_counts.get(key, 0) + 1
	for rows in grouped_rows.values():
		ordered_rows = sorted(rows, key=lambda item: item.kspacing, reverse=True)
		kspacing_values = [row.kspacing for row in ordered_rows]
		positions: list[float] = []
		for row in ordered_rows:
			base_position = mesh_positions[row.kpoint_mesh]
			key = (row.kspacing, base_position)
			rank = collision_ranks.get(key, 0)
			collision_ranks[key] = rank + 1
			count = collision_counts[key]
			offset = (rank - (count - 1) / 2.0) * 0.12 if count > 1 else 0.0
			positions.append(base_position + offset)
		line = axis.plot(kspacing_values, positions, linewidth=1.3, alpha=0.55, label=ordered_rows[0].label)[0]
		axis.scatter(
			kspacing_values,
			positions,
			s=115,
			color=line.get_color(),
			edgecolor="black",
			linewidth=0.85,
			zorder=3,
		)
	axis.set_title("VASP-generated k-point mesh", fontsize=12)
	axis.set_xlabel("KSPACING", fontsize=11)
	axis.set_ylabel("k-point mesh", fontsize=10)
	axis.set_yticks(range(len(mesh_labels)))
	axis.set_yticklabels(mesh_labels, fontsize=9)
	axis.set_ylim(-0.45, len(mesh_labels) - 0.55)
	axis.grid(True, alpha=0.25)
	axis.legend(fontsize=8, loc="upper left")
	axis.tick_params(axis="x", labelsize=10)
	axis.invert_xaxis()
	nkpts_axis = axis.twinx()
	nkpts_axis.set_ylim(axis.get_ylim())
	nkpts_axis.set_yticks(range(len(mesh_nkpts)))
	nkpts_axis.set_yticklabels([str(nkpts) for nkpts in mesh_nkpts], fontsize=9)
	nkpts_axis.set_ylabel("NKPTS", fontsize=10)


def describe_resources(rows: Iterable[StaticKspacingRow]) -> str:
	"""Describe the common detected resource footprint for plot labels."""
	resource_keys = {
		(row.mpi_ranks, row.threads_per_rank, row.gpus_detected)
		for row in rows
		if row.mpi_ranks is not None or row.threads_per_rank is not None or row.gpus_detected is not None
	}
	if len(resource_keys) != 1:
		return "detected resources vary"
	mpi_ranks, threads_per_rank, gpus_detected = next(iter(resource_keys))
	parts: list[str] = []
	if mpi_ranks is not None and threads_per_rank is not None:
		parts.append(f"{mpi_ranks} MPI rank x {threads_per_rank} thread")
	if gpus_detected is not None:
		parts.append(f"{gpus_detected} GPU")
	accelerators = sorted({row.accelerator for row in rows})
	if len(accelerators) == 1 and accelerators[0] != "unknown":
		parts.append(accelerators[0])
	elif any(accelerator != "unknown" for accelerator in accelerators):
		parts.append("mixed A100/H200")
	return ", ".join(parts) if parts else "detected resources unavailable"


def add_runtime_panel(
	axis: plt.Axes,
	grouped_rows: OrderedDict[tuple[str, str], list[StaticKspacingRow]],
) -> None:
	"""Add the runtime panel using VASP elapsed wall time."""
	for rows in grouped_rows.values():
		ordered_rows = sorted(rows, key=lambda item: item.kspacing, reverse=True)
		line = axis.plot(
			[row.kspacing for row in ordered_rows],
			[row.elapsed_seconds / 60.0 for row in ordered_rows],
			linewidth=1.7,
			label=ordered_rows[0].label,
		)[0]
		for row in ordered_rows:
			marker = ACCELERATOR_MARKERS.get(row.accelerator, "o")
			axis.scatter(
				row.kspacing,
				row.elapsed_seconds / 60.0,
				marker=marker,
				s=92 if marker != "*" else 150,
				color=line.get_color(),
				edgecolor="black",
				linewidth=0.8,
				zorder=3,
			)
			if row.accelerator == "H200":
				axis.annotate(
					"H200",
					(row.kspacing, row.elapsed_seconds / 60.0),
					textcoords="offset points",
					xytext=(5, 6),
					fontsize=7,
				)
	all_rows = [row for rows in grouped_rows.values() for row in rows]
	axis.set_title(f"VASP elapsed runtime ({describe_resources(all_rows)})", fontsize=12)
	axis.set_xlabel("KSPACING", fontsize=11)
	axis.set_ylabel("elapsed time (min)", fontsize=10)
	axis.grid(True, alpha=0.25)
	series_legend = axis.legend(fontsize=8, loc="upper left")
	axis.add_artist(series_legend)
	accelerators = [name for name in ("A100", "H200") if any(row.accelerator == name for row in all_rows)]
	if accelerators:
		axis.legend(
			[
				Line2D([0], [0], marker=ACCELERATOR_MARKERS[name], color="none", markerfacecolor="white", markeredgecolor="black", markersize=8)
				for name in accelerators
			],
			accelerators,
			fontsize=8,
			loc="upper right",
			title="GPU",
			title_fontsize=8,
		)
	axis.tick_params(axis="both", labelsize=10)
	axis.invert_xaxis()


def plot_static_kspacing_convergence(rows: list[StaticKspacingRow], output_png: Path) -> None:
	"""Create the static KSPACING convergence PNG."""
	grouped_rows = group_static_rows(rows)
	fig = plt.figure(figsize=(13.8, 10.4), dpi=220, constrained_layout=True)
	grid_spec = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.58, 0.58])
	toten_axis = fig.add_subplot(grid_spec[0, 0])
	internal_axis = fig.add_subplot(grid_spec[0, 1])
	kpoint_axis = fig.add_subplot(grid_spec[1, :])
	runtime_axis = fig.add_subplot(grid_spec[2, :])
	add_energy_panels((toten_axis, internal_axis), grouped_rows)
	add_kpoint_panel(kpoint_axis, grouped_rows)
	add_runtime_panel(runtime_axis, grouped_rows)
	fig.suptitle("Static KSPACING convergence including H200 K=0.20", fontsize=16)
	fig.savefig(output_png)
	plt.close(fig)


def main() -> None:
	"""Run the plotting workflow."""
	static_kspacing_rows = load_static_kspacing_rows(INPUT_CSV)
	plot_static_kspacing_convergence(static_kspacing_rows, OUTPUT_PNG)
	print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
	main()
