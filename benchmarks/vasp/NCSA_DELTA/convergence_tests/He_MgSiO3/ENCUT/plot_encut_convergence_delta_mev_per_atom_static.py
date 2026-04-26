"""Plot static ENCUT convergence with k-point mesh and runtime panels."""

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


BENCHMARK_DIR = Path(os.environ.get("ENCUT_BENCHMARK_DIR", Path(__file__).resolve().parent))
INPUT_CSV = BENCHMARK_DIR / "encut_convergence_delta_mev_per_atom_static.csv"
OUTPUT_PNG = Path(
	os.environ.get(
		"ENCUT_OUTPUT_PNG",
		BENCHMARK_DIR / "encut_convergence_delta_mev_per_atom_static.png",
	)
)

KPOINT_MESH_PATTERN = re.compile(
	r"generate k-points for:\s*(?P<nkx>\d+)\s+(?P<nky>\d+)\s+(?P<nkz>\d+)"
)
ELAPSED_PATTERN = re.compile(r"Elapsed time \(sec\):\s*(?P<seconds>[0-9.]+)")
RESOURCE_PATTERN = re.compile(
	r"running\s+(?P<mpi_ranks>\d+)\s+mpi-ranks,\s+with\s+(?P<threads>\d+)\s+threads/rank"
)
GPU_PATTERN = re.compile(r"Offloading initialized\s+\.\.\.\s+(?P<gpus>\d+)\s+GPUs detected")
PARTITION_PATTERN = re.compile(r"#SBATCH\s+--partition=(?P<partition>\S+)")
ACCELERATOR_MARKERS = {"A100": "^", "H200": "*", "unknown": "o"}


@dataclass(frozen=True)
class ConvergenceRow:
	"""One completed static ENCUT convergence calculation."""

	family: str
	config: str
	encut: int
	n_atoms: int
	nkpts: int
	reference_encut: int
	delta_toten_mev_per_atom: float
	delta_internal_mev_per_atom: float
	run_path: Path
	kpoint_mesh: tuple[int, int, int]
	elapsed_seconds: float
	mpi_ranks: int | None
	threads_per_rank: int | None
	gpus_detected: int | None
	accelerator: str

	@property
	def label(self) -> str:
		"""Return the compact legend label used in convergence plots."""
		family_label = self.family.replace("_atoms__and_He", " atoms")
		return f"{family_label}, {self.config}"

	@property
	def mesh_label(self) -> str:
		"""Return the k-point mesh as an axis label."""
		nkx, nky, nkz = self.kpoint_mesh
		return f"{nkx}x{nky}x{nkz}"


def resolve_run_path(row: dict[str, str]) -> Path:
	"""Resolve a run directory from the CSV row.

	Args:
		row: CSV record containing family, config, encut, and path fields.

	Returns:
		Existing run directory under the benchmark tree.

	Raises:
		FileNotFoundError: If no candidate run directory exists.
	"""
	csv_path = Path(row["path"])
	candidates = [
		csv_path,
		Path(str(csv_path).replace("/setup_MLMD/ENCUT_test/", "/setup_MLMD/benchmarking_tests/ENCUT_test/")),
		BENCHMARK_DIR / row["family"] / row["config"] / f"ENCUT_{int(row['encut']):04d}",
	]
	for candidate in candidates:
		if candidate.is_dir():
			return candidate
	raise FileNotFoundError(f"Could not resolve run directory for {row['family']} {row['config']} ENCUT={row['encut']}")


def read_outcar_metadata(outcar_path: Path) -> tuple[tuple[int, int, int], float, int | None, int | None, int | None]:
	"""Read VASP mesh, elapsed time, and resource metadata from OUTCAR.

	Args:
		outcar_path: Path to a VASP OUTCAR file.

	Returns:
		``(mesh, elapsed_seconds, mpi_ranks, threads_per_rank, gpus_detected)``.

	Raises:
		ValueError: If the OUTCAR lacks mesh or elapsed-time lines.
	"""
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


def load_rows(csv_path: Path) -> list[ConvergenceRow]:
	"""Load ENCUT convergence rows and attach k-point mesh metadata.

	Args:
		csv_path: Static ENCUT convergence CSV.

	Returns:
		Rows sorted by family, config, and ENCUT.
	"""
	rows: list[ConvergenceRow] = []
	with csv_path.open("r", encoding="utf-8", newline="") as handle:
		for raw_row in csv.DictReader(handle):
			run_path = resolve_run_path(raw_row)
			kpoint_mesh, elapsed_seconds, mpi_ranks, threads_per_rank, gpus_detected = read_outcar_metadata(
				run_path / "OUTCAR"
			)
			rows.append(
				ConvergenceRow(
					family=raw_row["family"],
					config=raw_row["config"],
					encut=int(raw_row["encut"]),
					n_atoms=int(raw_row["n_atoms"]),
					nkpts=int(raw_row["nkpts"]),
					reference_encut=int(raw_row["reference_encut"]),
					delta_toten_mev_per_atom=float(raw_row["delta_toten_meV_per_atom"]),
					delta_internal_mev_per_atom=float(raw_row["delta_internal_meV_per_atom"]),
					run_path=run_path,
					kpoint_mesh=kpoint_mesh,
					elapsed_seconds=elapsed_seconds,
					mpi_ranks=mpi_ranks,
					threads_per_rank=threads_per_rank,
					gpus_detected=gpus_detected,
					accelerator=read_accelerator(run_path),
				)
			)
	return sorted(rows, key=lambda item: (item.family, item.config, item.encut))


def group_rows(rows: Iterable[ConvergenceRow]) -> OrderedDict[tuple[str, str], list[ConvergenceRow]]:
	"""Group rows by simulation family and composition.

	Args:
		rows: Convergence rows to group.

	Returns:
		Ordered dictionary keyed by `(family, config)`.
	"""
	grouped: OrderedDict[tuple[str, str], list[ConvergenceRow]] = OrderedDict()
	for row in rows:
		grouped.setdefault((row.family, row.config), []).append(row)
	return grouped


def configure_energy_axis(axis: plt.Axes, title: str) -> None:
	"""Apply shared styling to an energy convergence axis.

	Args:
		axis: Matplotlib axis to style.
		title: Axis title.
	"""
	axis.set_title(title, fontsize=14)
	axis.set_xlabel("ENCUT (eV)", fontsize=11)
	axis.set_ylabel("meV/atom", fontsize=11)
	axis.set_yscale("symlog", linthresh=1.0e-1)
	axis.set_ylim(bottom=0.0)
	axis.grid(True, which="both", alpha=0.25)
	axis.tick_params(axis="both", labelsize=10)


def add_energy_panels(
	axes: tuple[plt.Axes, plt.Axes],
	grouped_rows: OrderedDict[tuple[str, str], list[ConvergenceRow]],
) -> None:
	"""Add the two top ENCUT convergence panels.

	Args:
		axes: `(TOTEN axis, internal-energy axis)` pair.
		grouped_rows: Rows grouped by composition.
	"""
	toten_axis, internal_axis = axes
	for rows in grouped_rows.values():
		ordered_rows = sorted(rows, key=lambda item: item.encut)
		encuts = [row.encut for row in ordered_rows]
		toten_values = [row.delta_toten_mev_per_atom for row in ordered_rows]
		internal_values = [row.delta_internal_mev_per_atom for row in ordered_rows]
		label = ordered_rows[0].label
		toten_axis.plot(encuts, toten_values, marker="o", linewidth=1.8, markersize=6.5, label=label)
		internal_axis.plot(encuts, internal_values, marker="s", linewidth=1.8, markersize=6.5, label=label)

	configure_energy_axis(toten_axis, "|Δ TOTEN| vs highest valid ENCUT")
	configure_energy_axis(internal_axis, "|Δ internal energy| vs highest valid ENCUT")
	toten_axis.legend(fontsize=8, loc="upper right")
	internal_axis.legend(fontsize=8, loc="upper right")


def add_kpoint_panel(
	axis: plt.Axes,
	grouped_rows: OrderedDict[tuple[str, str], list[ConvergenceRow]],
) -> None:
	"""Add the lower panel showing the VASP k-point mesh for each ENCUT.

	Args:
		axis: Matplotlib axis spanning the figure width.
		grouped_rows: Rows grouped by composition.
	"""
	unique_meshes = sorted(
		{row.kpoint_mesh for rows in grouped_rows.values() for row in rows},
		key=lambda mesh: (mesh[0] * mesh[1] * mesh[2], mesh),
	)
	mesh_positions = {mesh: index for index, mesh in enumerate(unique_meshes)}
	mesh_labels = [f"{mesh[0]}x{mesh[1]}x{mesh[2]}" for mesh in unique_meshes]
	mesh_nkpts = [mesh[0] * mesh[1] * mesh[2] for mesh in unique_meshes]

	for rows in grouped_rows.values():
		ordered_rows = sorted(rows, key=lambda item: item.encut)
		encuts = [row.encut for row in ordered_rows]
		positions = [mesh_positions[row.kpoint_mesh] for row in ordered_rows]
		line = axis.plot(
			encuts,
			positions,
			linewidth=1.3,
			alpha=0.55,
			label=ordered_rows[0].label,
		)[0]
		axis.scatter(
			encuts,
			positions,
			s=115,
			color=line.get_color(),
			edgecolor="black",
			linewidth=0.85,
			zorder=3,
		)

	axis.set_title("VASP-generated k-point mesh", fontsize=12)
	axis.set_xlabel("ENCUT (eV)", fontsize=11)
	axis.set_ylabel("k-point mesh", fontsize=10)
	axis.set_yticks(range(len(mesh_labels)))
	axis.set_yticklabels(mesh_labels, fontsize=9)
	axis.grid(True, alpha=0.25)
	axis.legend(fontsize=8, loc="upper left")
	axis.tick_params(axis="x", labelsize=10)

	nkpts_axis = axis.twinx()
	nkpts_axis.set_ylim(axis.get_ylim())
	nkpts_axis.set_yticks(range(len(mesh_nkpts)))
	nkpts_axis.set_yticklabels([str(nkpts) for nkpts in mesh_nkpts], fontsize=9)
	nkpts_axis.set_ylabel("NKPTS", fontsize=10)


def describe_resources(rows: Iterable[ConvergenceRow]) -> str:
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
	grouped_rows: OrderedDict[tuple[str, str], list[ConvergenceRow]],
) -> None:
	"""Add the runtime panel using VASP elapsed wall time."""
	for rows in grouped_rows.values():
		ordered_rows = sorted(rows, key=lambda item: item.encut)
		line = axis.plot(
			[row.encut for row in ordered_rows],
			[row.elapsed_seconds / 60.0 for row in ordered_rows],
			linewidth=1.7,
			label=ordered_rows[0].label,
		)[0]
		for row in ordered_rows:
			marker = ACCELERATOR_MARKERS.get(row.accelerator, "o")
			axis.scatter(
				row.encut,
				row.elapsed_seconds / 60.0,
				marker=marker,
				s=92 if marker != "*" else 150,
				color=line.get_color(),
				edgecolor="black",
				linewidth=0.8,
				zorder=3,
			)
	all_rows = [row for rows in grouped_rows.values() for row in rows]
	axis.set_title(f"VASP elapsed runtime ({describe_resources(all_rows)})", fontsize=12)
	axis.set_xlabel("ENCUT (eV)", fontsize=11)
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


def plot_encut_convergence(rows: list[ConvergenceRow], output_png: Path) -> None:
	"""Create the ENCUT convergence figure.

	Args:
		rows: Completed convergence rows with FFT-grid metadata.
		output_png: Destination PNG path.
	"""
	grouped_rows = group_rows(rows)
	fig = plt.figure(figsize=(13.8, 10.4), dpi=220, constrained_layout=True)
	grid_spec = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.58, 0.58])
	toten_axis = fig.add_subplot(grid_spec[0, 0])
	internal_axis = fig.add_subplot(grid_spec[0, 1])
	kpoint_axis = fig.add_subplot(grid_spec[1, :])
	runtime_axis = fig.add_subplot(grid_spec[2, :])

	add_energy_panels((toten_axis, internal_axis), grouped_rows)
	add_kpoint_panel(kpoint_axis, grouped_rows)
	add_runtime_panel(runtime_axis, grouped_rows)

	fig.suptitle("Static ENCUT convergence at KSPACING=0.40", fontsize=16)
	fig.savefig(output_png)
	plt.close(fig)


def main() -> None:
	"""Run the plotting workflow."""
	rows = load_rows(INPUT_CSV)
	plot_encut_convergence(rows, OUTPUT_PNG)
	print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
	main()
