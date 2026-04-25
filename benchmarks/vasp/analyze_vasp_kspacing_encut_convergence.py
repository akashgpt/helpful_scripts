"""Analyze static VASP KSPACING and ENCUT convergence benchmarks.

The script expects a TSV manifest with these columns:
``run_id, family, case, label, value, run_dir``.  The ``family`` column should
contain ``KSPACING`` or ``ENCUT``.  Each ``run_dir`` should contain ``OUTCAR``
and ``POSCAR`` for completed runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


TOTEN_PATTERN = re.compile(r"free\s+energy\s+TOTEN\s+=\s+(?P<energy>[-+0-9.Ee]+)")
INTERNAL_PATTERN = re.compile(
	r"energy\s+without entropy\s*=\s*(?P<energy>[-+0-9.Ee]+)\s+energy\(sigma->0\)\s*=\s*(?P<sigma>[-+0-9.Ee]+)"
)
KPOINT_MESH_PATTERN = re.compile(
	r"generate k-points for:\s*(?P<nkx>\d+)\s+(?P<nky>\d+)\s+(?P<nkz>\d+)"
)
ELAPSED_PATTERN = re.compile(r"Elapsed time \(sec\):\s*(?P<seconds>[0-9.]+)")
RESOURCE_PATTERN = re.compile(
	r"running\s+(?P<mpi_ranks>\d+)\s+mpi-ranks,\s+with\s+(?P<threads>\d+)\s+threads/rank"
)
NKPTS_PATTERN = re.compile(r"NKPTS =\s*(?P<nkpts>\d+)\s+k-points in BZ")
EDIFF_REACHED_PATTERN = re.compile(r"aborting loop because EDIFF is reached")
SELF_CONSISTENCY_FAILURE_PATTERN = re.compile(r"self-consistency was not achieved")


@dataclass(frozen=True)
class ManifestRow:
	"""One benchmark row from the saved manifest."""

	run_id: int
	family: str
	case_name: str
	label: str
	value: float
	run_dir: Path


@dataclass(frozen=True)
class RunResult:
	"""Parsed data from one completed and electronically converged run."""

	manifest: ManifestRow
	n_atoms: int
	nkpts: int
	kpoint_mesh: tuple[int, int, int]
	toten_ev: float
	internal_ev: float
	sigma0_ev: float
	elapsed_seconds: float
	mpi_ranks: int | None
	threads_per_rank: int | None

	@property
	def case_label(self) -> str:
		"""Return a compact label for plot legends."""
		return self.manifest.case_name.replace("__1", "").replace("__", " ")


@dataclass(frozen=True)
class SkippedRun:
	"""One manifest row excluded from the convergence analysis."""

	manifest: ManifestRow
	reason: str


def display_run_label(label: str) -> str:
	"""Return a run label with obsolete accelerator suffixes removed."""
	return label.removesuffix("_H200").removesuffix("_A100")


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments.

	Returns:
		Parsed argument namespace.
	"""
	parser = argparse.ArgumentParser(
		description="Analyze static VASP KSPACING and ENCUT convergence benchmarks."
	)
	parser.add_argument(
		"--source-dir",
		type=Path,
		required=True,
		help="Benchmark source directory containing the manifest and run folders.",
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		default=None,
		help="Manifest TSV path. Defaults to the first benchmark_manifest*.tsv in --source-dir.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory where CSV, PNG, and Markdown outputs are written. Defaults to --source-dir.",
	)
	parser.add_argument(
		"--system-label",
		default=None,
		help="System label for CSVs and plot titles. Defaults to the source directory name.",
	)
	parser.add_argument(
		"--encut-kspacing",
		default="0.40",
		help="KSPACING label written to the ENCUT CSV.",
	)
	parser.add_argument(
		"--threshold-mev-per-atom",
		type=float,
		default=1.0,
		help="Threshold used in the text summary for converged sweep settings.",
	)
	parser.add_argument(
		"--copy-plots-to-source",
		action="store_true",
		help="Also copy the generated PNG plots into --source-dir.",
	)
	return parser.parse_args()


def resolve_manifest(source_dir: Path, manifest_path: Path | None) -> Path:
	"""Resolve the manifest path for a benchmark source directory.

	Args:
		source_dir: Benchmark source directory.
		manifest_path: Optional user-provided manifest path.

	Returns:
		Resolved manifest path.

	Raises:
		FileNotFoundError: If no unique manifest can be found.
	"""
	if manifest_path is not None:
		return manifest_path
	candidates = sorted(source_dir.glob("benchmark_manifest*.tsv"))
	if len(candidates) != 1:
		raise FileNotFoundError(
			f"Expected exactly one benchmark_manifest*.tsv in {source_dir}, found {len(candidates)}"
		)
	return candidates[0]


def read_manifest(manifest_path: Path) -> list[ManifestRow]:
	"""Read a benchmark manifest.

	Args:
		manifest_path: Path to the manifest TSV file.

	Returns:
		Manifest rows in file order.
	"""
	rows: list[ManifestRow] = []
	with manifest_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		for raw_row in reader:
			rows.append(
				ManifestRow(
					run_id=int(raw_row["run_id"]),
					family=raw_row["family"],
					case_name=raw_row["case"],
					label=raw_row["label"],
					value=float(raw_row["value"]),
					run_dir=Path(raw_row["run_dir"]),
				)
			)
	return rows


def count_atoms(poscar_path: Path) -> int:
	"""Count atoms from a POSCAR file.

	Args:
		poscar_path: Path to the POSCAR file.

	Returns:
		Total number of atoms.
	"""
	lines = poscar_path.read_text(encoding="utf-8", errors="replace").splitlines()
	counts_line = lines[6].split()
	return sum(int(value) for value in counts_line)


def parse_outcar(outcar_path: Path) -> tuple[float, float, float, int, tuple[int, int, int], float, int | None, int | None]:
	"""Parse final energies, k-point data, and runtime from an OUTCAR.

	Args:
		outcar_path: Path to the VASP OUTCAR file.

	Returns:
		``(toten_ev, internal_ev, sigma0_ev, nkpts, mesh, elapsed_seconds, mpi_ranks, threads_per_rank)``.

	Raises:
		ValueError: If any required quantity is missing.
	"""
	toten_ev: float | None = None
	internal_ev: float | None = None
	sigma0_ev: float | None = None
	nkpts: int | None = None
	kpoint_mesh: tuple[int, int, int] | None = None
	elapsed_seconds: float | None = None
	mpi_ranks: int | None = None
	threads_per_rank: int | None = None

	with outcar_path.open("r", encoding="utf-8", errors="replace") as handle:
		for line in handle:
			if match := TOTEN_PATTERN.search(line):
				toten_ev = float(match.group("energy"))
			if match := INTERNAL_PATTERN.search(line):
				internal_ev = float(match.group("energy"))
				sigma0_ev = float(match.group("sigma"))
			if nkpts is None and (match := NKPTS_PATTERN.search(line)):
				nkpts = int(match.group("nkpts"))
			if kpoint_mesh is None and (match := KPOINT_MESH_PATTERN.search(line)):
				kpoint_mesh = (
					int(match.group("nkx")),
					int(match.group("nky")),
					int(match.group("nkz")),
				)
			if match := ELAPSED_PATTERN.search(line):
				elapsed_seconds = float(match.group("seconds"))
			if mpi_ranks is None and (match := RESOURCE_PATTERN.search(line)):
				mpi_ranks = int(match.group("mpi_ranks"))
				threads_per_rank = int(match.group("threads"))

	if toten_ev is None:
		raise ValueError(f"Could not find final TOTEN in {outcar_path}")
	if internal_ev is None or sigma0_ev is None:
		raise ValueError(f"Could not find final internal-energy line in {outcar_path}")
	if nkpts is None:
		raise ValueError(f"Could not find NKPTS in {outcar_path}")
	if kpoint_mesh is None:
		raise ValueError(f"Could not find generated k-point mesh in {outcar_path}")
	if elapsed_seconds is None:
		raise ValueError(f"Could not find elapsed time in {outcar_path}")
	return (
		toten_ev,
		internal_ev,
		sigma0_ev,
		nkpts,
		kpoint_mesh,
		elapsed_seconds,
		mpi_ranks,
		threads_per_rank,
	)


def electronic_convergence_status(outcar_path: Path) -> tuple[bool, str]:
	"""Check whether a VASP OUTCAR reached electronic self-consistency.

	Args:
		outcar_path: Path to the VASP OUTCAR file.

	Returns:
		``(is_converged, reason)`` where reason is suitable for reports.
	"""
	ediff_reached = False
	self_consistency_failed = False
	with outcar_path.open("r", encoding="utf-8", errors="replace") as handle:
		for line in handle:
			if EDIFF_REACHED_PATTERN.search(line):
				ediff_reached = True
			if SELF_CONSISTENCY_FAILURE_PATTERN.search(line):
				self_consistency_failed = True
	if self_consistency_failed:
		return False, "electronic self-consistency not achieved"
	if not ediff_reached:
		return False, "no final EDIFF convergence marker"
	return True, "electronic self-consistency reached"


def load_results(manifest_rows: list[ManifestRow]) -> list[RunResult]:
	"""Load completed and electronically converged benchmark results.

	Args:
		manifest_rows: Manifest rows to inspect.

	Returns:
		Converged run results sorted by family, case, and sweep value.
	"""
	results: list[RunResult] = []
	for manifest_row in manifest_rows:
		outcar_path = manifest_row.run_dir / "OUTCAR"
		poscar_path = manifest_row.run_dir / "POSCAR"
		if not outcar_path.is_file() or not poscar_path.is_file():
			continue
		is_converged, _reason = electronic_convergence_status(outcar_path)
		if not is_converged:
			continue
		toten_ev, internal_ev, sigma0_ev, nkpts, kpoint_mesh, elapsed_seconds, mpi_ranks, threads_per_rank = parse_outcar(
			outcar_path
		)
		results.append(
			RunResult(
				manifest=manifest_row,
				n_atoms=count_atoms(poscar_path),
				nkpts=nkpts,
				kpoint_mesh=kpoint_mesh,
				toten_ev=toten_ev,
				internal_ev=internal_ev,
				sigma0_ev=sigma0_ev,
				elapsed_seconds=elapsed_seconds,
				mpi_ranks=mpi_ranks,
				threads_per_rank=threads_per_rank,
			)
		)
	return sorted(results, key=lambda item: (item.manifest.family, item.manifest.case_name, item.manifest.value))


def find_skipped_rows(manifest_rows: list[ManifestRow]) -> list[SkippedRun]:
	"""Return manifest rows excluded from the analysis.

	Args:
		manifest_rows: Manifest rows to inspect.

	Returns:
		Rows missing required files or lacking electronic convergence.
	"""
	skipped_rows: list[SkippedRun] = []
	for manifest_row in manifest_rows:
		outcar_path = manifest_row.run_dir / "OUTCAR"
		poscar_path = manifest_row.run_dir / "POSCAR"
		if not outcar_path.is_file() or not poscar_path.is_file():
			skipped_rows.append(SkippedRun(manifest=manifest_row, reason="missing OUTCAR or POSCAR"))
			continue
		is_converged, reason = electronic_convergence_status(outcar_path)
		if not is_converged:
			skipped_rows.append(SkippedRun(manifest=manifest_row, reason=reason))
	return skipped_rows


def group_results(results: list[RunResult], family: str) -> OrderedDict[str, list[RunResult]]:
	"""Group results by case name for one sweep family.

	Args:
		results: Parsed benchmark results.
		family: Sweep family, either ``KSPACING`` or ``ENCUT``.

	Returns:
		Ordered mapping from case name to its ordered sweep rows.
	"""
	grouped: OrderedDict[str, list[RunResult]] = OrderedDict()
	for result in results:
		if result.manifest.family != family:
			continue
		grouped.setdefault(result.manifest.case_name, []).append(result)
	for case_name, case_results in grouped.items():
		grouped[case_name] = sorted(case_results, key=lambda item: item.manifest.value)
	return grouped


def mev_per_atom_delta(energy_ev: float, reference_ev: float, n_atoms: int) -> float:
	"""Convert an absolute energy difference to meV/atom.

	Args:
		energy_ev: Energy for the current run.
		reference_ev: Reference energy for the group.
		n_atoms: Number of atoms in the structure.

	Returns:
		Absolute energy delta in meV/atom.
	"""
	return abs(energy_ev - reference_ev) * 1000.0 / float(n_atoms)


def write_kspacing_csv(
	grouped: OrderedDict[str, list[RunResult]],
	output_csv: Path,
	system_label: str,
) -> None:
	"""Write a static KSPACING convergence table.

	Args:
		grouped: KSPACING results grouped by case name.
		output_csv: Destination CSV path.
		system_label: Label written into the CSV ``family`` column.
	"""
	fieldnames = [
		"family",
		"config",
		"run",
		"kspacing",
		"nkpts",
		"reference_run",
		"reference_kspacing",
		"delta_toten_meV_per_atom",
		"delta_internal_meV_per_atom",
	]
	with output_csv.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for case_name, case_results in grouped.items():
			reference = min(case_results, key=lambda item: item.manifest.value)
			for result in case_results:
				writer.writerow(
					{
						"family": system_label,
						"config": case_name,
						"run": display_run_label(result.manifest.label),
						"kspacing": f"{result.manifest.value:.2f}",
						"nkpts": result.nkpts,
						"reference_run": display_run_label(reference.manifest.label),
						"reference_kspacing": f"{reference.manifest.value:.2f}",
						"delta_toten_meV_per_atom": f"{mev_per_atom_delta(result.toten_ev, reference.toten_ev, result.n_atoms):.8f}",
						"delta_internal_meV_per_atom": f"{mev_per_atom_delta(result.internal_ev, reference.internal_ev, result.n_atoms):.8f}",
					}
				)


def write_encut_csv(
	grouped: OrderedDict[str, list[RunResult]],
	output_csv: Path,
	system_label: str,
	encut_kspacing: str,
) -> None:
	"""Write a static ENCUT convergence table.

	Args:
		grouped: ENCUT results grouped by case name.
		output_csv: Destination CSV path.
		system_label: Label written into the CSV ``family`` column.
		encut_kspacing: KSPACING label written into the CSV.
	"""
	fieldnames = [
		"family",
		"config",
		"encut",
		"kspacing",
		"nkpts",
		"n_atoms",
		"reference_encut",
		"toten_eV_per_atom",
		"internal_eV_per_atom",
		"delta_toten_meV_per_atom",
		"delta_internal_meV_per_atom",
		"path",
	]
	with output_csv.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for case_name, case_results in grouped.items():
			reference = max(case_results, key=lambda item: item.manifest.value)
			for result in case_results:
				writer.writerow(
					{
						"family": system_label,
						"config": case_name,
						"encut": int(result.manifest.value),
						"kspacing": encut_kspacing,
						"nkpts": result.nkpts,
						"n_atoms": result.n_atoms,
						"reference_encut": int(reference.manifest.value),
						"toten_eV_per_atom": f"{result.toten_ev / float(result.n_atoms):.10f}",
						"internal_eV_per_atom": f"{result.internal_ev / float(result.n_atoms):.10f}",
						"delta_toten_meV_per_atom": f"{mev_per_atom_delta(result.toten_ev, reference.toten_ev, result.n_atoms):.8f}",
						"delta_internal_meV_per_atom": f"{mev_per_atom_delta(result.internal_ev, reference.internal_ev, result.n_atoms):.8f}",
						"path": str(result.manifest.run_dir),
					}
				)


def describe_resources(results: list[RunResult]) -> str:
	"""Build a shared resource-description string for plot titles.

	Args:
		results: Results included in the plots.

	Returns:
		Human-readable resource summary.
	"""
	resource_counts: dict[tuple[int, int], int] = {}
	for result in results:
		if result.mpi_ranks is None or result.threads_per_rank is None:
			continue
		key = (result.mpi_ranks, result.threads_per_rank)
		resource_counts[key] = resource_counts.get(key, 0) + 1
	if not resource_counts:
		return "detected resources unavailable"
	if len(resource_counts) == 1:
		mpi_ranks, threads_per_rank = next(iter(resource_counts))
		return f"{mpi_ranks} MPI ranks, {threads_per_rank} thread/rank"
	(mpi_ranks, threads_per_rank), dominant_count = max(resource_counts.items(), key=lambda item: item[1])
	parts = [
		f"mostly {mpi_ranks} MPI ranks",
		f"{threads_per_rank} thread/rank",
		f"{len(results) - dominant_count} outlier run(s)",
	]
	return ", ".join(parts)


def padded_energy_limit(values: list[float]) -> float:
	"""Choose a decade upper bound for an energy-delta axis.

	Args:
		values: Data values plotted on the axis.

	Returns:
		Next decade above the maximum plotted value.
	"""
	maximum_value = max(values) if values else 0.0
	if maximum_value <= 0.0:
		return 1.0e-4
	return 10.0 ** math.ceil(math.log10(maximum_value))


def positive_decade_floor(values: list[float]) -> float:
	"""Return the lowest useful positive tick for plotted values.

	Args:
		values: Data values plotted on a non-negative symlog axis.

	Returns:
		The decade of the smallest positive value.
	"""
	positive_values = [value for value in values if value > 0.0]
	if not positive_values:
		return 1.0e-5
	minimum_value = min(positive_values)
	return 10.0 ** math.floor(math.log10(minimum_value))


def energy_ticks(values: list[float]) -> list[float]:
	"""Build compact major ticks for a zero-baseline symlog energy axis.

	Args:
		values: Data values plotted on the axis.

	Returns:
		Major tick values from zero through the useful positive decades.
	"""
	upper_limit = padded_energy_limit(values)
	lowest_tick = positive_decade_floor(values)
	ticks = [0.0]
	current_tick = lowest_tick
	while current_tick <= upper_limit:
		ticks.append(current_tick)
		current_tick *= 10.0
	return ticks


def format_energy_tick(value: float, _position: int) -> str:
	"""Format symlog energy ticks without crowding the zero baseline."""
	if value == 0.0:
		return "0"
	if 1.0e-2 <= value < 1.0e3:
		return f"{value:g}"
	return f"{value:.0e}"


def configure_energy_scale(axis: plt.Axes, values: list[float]) -> None:
	"""Apply compact zero-baseline symlog scaling to an energy axis.

	Args:
		axis: Matplotlib axis to configure.
		values: Data values plotted on the axis.
	"""
	lowest_tick = positive_decade_floor(values)
	axis.set_yscale("symlog", linthresh=lowest_tick)
	axis.set_ylim(0.0, padded_energy_limit(values))
	axis.set_yticks(energy_ticks(values))
	axis.yaxis.set_major_formatter(FuncFormatter(format_energy_tick))


def count_collision_keys(keys: list[tuple[float, float]]) -> dict[tuple[float, float], int]:
	"""Count repeated plot coordinates for visual staggering.

	Args:
		keys: Plot-coordinate keys that may overlap.

	Returns:
		Count of each repeated key.
	"""
	counts: dict[tuple[float, float], int] = {}
	for key in keys:
		counts[key] = counts.get(key, 0) + 1
	return counts


def stagger_offset(
	key: tuple[float, float],
	counts: dict[tuple[float, float], int],
	ranks: dict[tuple[float, float], int],
	step: float,
) -> float:
	"""Return a centered visual offset for repeated plot coordinates.

	Args:
		key: Plot-coordinate key for the current point.
		counts: Total number of points at each key.
		ranks: Running rank counter for points already drawn at each key.
		step: Offset spacing in axis units.

	Returns:
		Centered offset in axis units.
	"""
	count = counts.get(key, 1)
	if count <= 1:
		return 0.0
	rank = ranks.get(key, 0)
	ranks[key] = rank + 1
	return (rank - (count - 1) / 2.0) * step


def plot_kspacing(
	grouped: OrderedDict[str, list[RunResult]],
	output_png: Path,
	system_label: str,
) -> None:
	"""Plot the KSPACING convergence figure.

	Args:
		grouped: KSPACING results grouped by case name.
		output_png: Destination PNG path.
		system_label: System label used in the figure title.
	"""
	all_results = [result for case_results in grouped.values() for result in case_results]
	fig = plt.figure(figsize=(13.8, 10.4), dpi=220, constrained_layout=True)
	grid_spec = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.58, 0.58])
	toten_axis = fig.add_subplot(grid_spec[0, 0])
	internal_axis = fig.add_subplot(grid_spec[0, 1])
	kpoint_axis = fig.add_subplot(grid_spec[1, :])
	runtime_axis = fig.add_subplot(grid_spec[2, :])

	toten_values: list[float] = []
	internal_values: list[float] = []
	unique_meshes = sorted(
		{result.kpoint_mesh for result in all_results},
		key=lambda mesh: (mesh[0] * mesh[1] * mesh[2], mesh),
	)
	mesh_positions = {mesh: index for index, mesh in enumerate(unique_meshes)}
	kpoint_collision_counts = count_collision_keys(
		[(result.manifest.value, float(mesh_positions[result.kpoint_mesh])) for result in all_results]
	)
	kpoint_collision_ranks: dict[tuple[float, float], int] = {}

	for case_results in grouped.values():
		ordered = sorted(case_results, key=lambda item: item.manifest.value, reverse=True)
		reference = min(case_results, key=lambda item: item.manifest.value)
		kspacing_values = [result.manifest.value for result in ordered]
		delta_toten = [
			mev_per_atom_delta(result.toten_ev, reference.toten_ev, result.n_atoms)
			for result in ordered
		]
		delta_internal = [
			mev_per_atom_delta(result.internal_ev, reference.internal_ev, result.n_atoms)
			for result in ordered
		]
		toten_values.extend(delta_toten)
		internal_values.extend(delta_internal)
		label = ordered[0].case_label
		toten_axis.plot(kspacing_values, delta_toten, marker="o", linewidth=1.8, markersize=6.5, label=label)
		internal_axis.plot(kspacing_values, delta_internal, marker="s", linewidth=1.8, markersize=6.5, label=label)

		positions: list[float] = []
		for result in ordered:
			base_position = mesh_positions[result.kpoint_mesh]
			key = (result.manifest.value, float(base_position))
			positions.append(base_position + stagger_offset(key, kpoint_collision_counts, kpoint_collision_ranks, step=0.12))
		kpoint_line = kpoint_axis.plot(kspacing_values, positions, linewidth=1.3, alpha=0.55, label=label)[0]
		for x_value, y_value in zip(kspacing_values, positions):
			kpoint_axis.scatter(
				x_value,
				y_value,
				s=115,
				marker="o",
				color=kpoint_line.get_color(),
				edgecolor="black",
				linewidth=0.85,
				zorder=3,
			)

		runtime_line = runtime_axis.plot(
			kspacing_values,
			[result.elapsed_seconds / 60.0 for result in ordered],
			linewidth=1.7,
			label=label,
		)[0]
		for result in ordered:
			runtime_axis.scatter(
				result.manifest.value,
				result.elapsed_seconds / 60.0,
				marker="o",
				s=90,
				color=runtime_line.get_color(),
				edgecolor="black",
				linewidth=0.8,
				zorder=3,
			)

	toten_axis.set_title(r"$|\Delta$ TOTEN| vs smallest valid KSPACING", fontsize=14)
	internal_axis.set_title(r"$|\Delta$ internal energy| vs smallest valid KSPACING", fontsize=14)
	for axis, values in ((toten_axis, toten_values), (internal_axis, internal_values)):
		axis.set_xlabel("KSPACING", fontsize=11)
		axis.set_ylabel("meV/atom", fontsize=11)
		configure_energy_scale(axis, values)
		axis.grid(True, which="both", alpha=0.25)
		axis.tick_params(axis="both", labelsize=10)
		axis.invert_xaxis()
		axis.legend(fontsize=8, loc="upper right")

	kpoint_axis.set_title("VASP-generated k-point mesh", fontsize=12)
	kpoint_axis.set_xlabel("KSPACING", fontsize=11)
	kpoint_axis.set_ylabel("k-point mesh", fontsize=10)
	kpoint_axis.set_yticks(range(len(unique_meshes)))
	kpoint_axis.set_yticklabels([f"{mesh[0]}x{mesh[1]}x{mesh[2]}" for mesh in unique_meshes], fontsize=9)
	kpoint_axis.set_ylim(-0.45, len(unique_meshes) - 0.55)
	kpoint_axis.grid(True, alpha=0.25)
	kpoint_axis.tick_params(axis="x", labelsize=10)
	kpoint_axis.invert_xaxis()
	kpoint_axis.legend(fontsize=8, loc="upper left")
	nkpts_axis = kpoint_axis.twinx()
	nkpts_axis.set_ylim(kpoint_axis.get_ylim())
	nkpts_axis.set_yticks(range(len(unique_meshes)))
	nkpts_axis.set_yticklabels([str(math.prod(mesh)) for mesh in unique_meshes], fontsize=9)
	nkpts_axis.set_ylabel("Full mesh NKPTS", fontsize=10)

	runtime_axis.set_title(f"VASP elapsed runtime ({describe_resources(all_results)})", fontsize=12)
	runtime_axis.set_xlabel("KSPACING", fontsize=11)
	runtime_axis.set_ylabel("elapsed time (min)", fontsize=10)
	runtime_axis.grid(True, alpha=0.25)
	runtime_axis.tick_params(axis="both", labelsize=10)
	runtime_axis.invert_xaxis()
	runtime_axis.legend(fontsize=8, loc="upper left")

	fig.suptitle(f"{system_label} static KSPACING convergence", fontsize=16)
	fig.savefig(output_png)
	plt.close(fig)


def plot_encut(
	grouped: OrderedDict[str, list[RunResult]],
	output_png: Path,
	system_label: str,
) -> None:
	"""Plot the ENCUT convergence figure.

	Args:
		grouped: ENCUT results grouped by case name.
		output_png: Destination PNG path.
		system_label: System label used in the figure title.
	"""
	all_results = [result for case_results in grouped.values() for result in case_results]
	fig = plt.figure(figsize=(13.8, 10.4), dpi=220, constrained_layout=True)
	grid_spec = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.58, 0.58])
	toten_axis = fig.add_subplot(grid_spec[0, 0])
	internal_axis = fig.add_subplot(grid_spec[0, 1])
	kpoint_axis = fig.add_subplot(grid_spec[1, :])
	runtime_axis = fig.add_subplot(grid_spec[2, :])

	toten_values: list[float] = []
	internal_values: list[float] = []
	unique_meshes = sorted(
		{result.kpoint_mesh for result in all_results},
		key=lambda mesh: (mesh[0] * mesh[1] * mesh[2], mesh),
	)
	mesh_positions = {mesh: index for index, mesh in enumerate(unique_meshes)}
	for case_results in grouped.values():
		ordered = sorted(case_results, key=lambda item: item.manifest.value)
		reference = max(case_results, key=lambda item: item.manifest.value)
		encut_values = [result.manifest.value for result in ordered]
		delta_toten = [mev_per_atom_delta(result.toten_ev, reference.toten_ev, result.n_atoms) for result in ordered]
		delta_internal = [
			mev_per_atom_delta(result.internal_ev, reference.internal_ev, result.n_atoms) for result in ordered
		]
		toten_values.extend(delta_toten)
		internal_values.extend(delta_internal)
		label = ordered[0].case_label
		toten_axis.plot(encut_values, delta_toten, marker="o", linewidth=1.8, markersize=6.5, label=label)
		internal_axis.plot(encut_values, delta_internal, marker="s", linewidth=1.8, markersize=6.5, label=label)

		positions = [mesh_positions[result.kpoint_mesh] for result in ordered]
		line = kpoint_axis.plot(encut_values, positions, linewidth=1.3, alpha=0.55, label=label)[0]
		kpoint_axis.scatter(
			encut_values,
			positions,
			s=115,
			color=line.get_color(),
			edgecolor="black",
			linewidth=0.85,
			zorder=3,
		)

		runtime_line = runtime_axis.plot(
			encut_values,
			[result.elapsed_seconds / 60.0 for result in ordered],
			linewidth=1.7,
			label=label,
		)[0]
		for result in ordered:
			runtime_axis.scatter(
				result.manifest.value,
				result.elapsed_seconds / 60.0,
				s=90,
				color=runtime_line.get_color(),
				edgecolor="black",
				linewidth=0.8,
				zorder=3,
			)

	toten_axis.set_title(r"$|\Delta$ TOTEN| vs highest valid ENCUT", fontsize=14)
	internal_axis.set_title(r"$|\Delta$ internal energy| vs highest valid ENCUT", fontsize=14)
	for axis, values in ((toten_axis, toten_values), (internal_axis, internal_values)):
		axis.set_xlabel("ENCUT (eV)", fontsize=11)
		axis.set_ylabel("meV/atom", fontsize=11)
		configure_energy_scale(axis, values)
		axis.grid(True, which="both", alpha=0.25)
		axis.tick_params(axis="both", labelsize=10)
		axis.legend(fontsize=8, loc="upper right")

	kpoint_axis.set_title("VASP-generated k-point mesh", fontsize=12)
	kpoint_axis.set_xlabel("ENCUT (eV)", fontsize=11)
	kpoint_axis.set_ylabel("k-point mesh", fontsize=10)
	kpoint_axis.set_yticks(range(len(unique_meshes)))
	kpoint_axis.set_yticklabels([f"{mesh[0]}x{mesh[1]}x{mesh[2]}" for mesh in unique_meshes], fontsize=9)
	kpoint_axis.set_ylim(-0.45, len(unique_meshes) - 0.55)
	kpoint_axis.grid(True, alpha=0.25)
	kpoint_axis.tick_params(axis="x", labelsize=10)
	kpoint_axis.legend(fontsize=8, loc="upper left")
	nkpts_axis = kpoint_axis.twinx()
	nkpts_axis.set_ylim(kpoint_axis.get_ylim())
	nkpts_axis.set_yticks(range(len(unique_meshes)))
	nkpts_axis.set_yticklabels([str(math.prod(mesh)) for mesh in unique_meshes], fontsize=9)
	nkpts_axis.set_ylabel("Full mesh NKPTS", fontsize=10)

	runtime_axis.set_title(f"VASP elapsed runtime ({describe_resources(all_results)})", fontsize=12)
	runtime_axis.set_xlabel("ENCUT (eV)", fontsize=11)
	runtime_axis.set_ylabel("elapsed time (min)", fontsize=10)
	runtime_axis.grid(True, alpha=0.25)
	runtime_axis.tick_params(axis="both", labelsize=10)
	runtime_axis.legend(fontsize=8, loc="upper left")

	fig.suptitle(f"{system_label} static ENCUT convergence", fontsize=16)
	fig.savefig(output_png)
	plt.close(fig)


def find_first_converged(
	case_results: list[RunResult],
	reference_energy_selector: str,
	threshold_mev_per_atom: float,
	prefer_highest_value: bool,
) -> RunResult | None:
	"""Find the least expensive setting meeting both energy thresholds.

	Args:
		case_results: One ordered case sweep.
		reference_energy_selector: Either ``"min"`` or ``"max"`` reference.
		threshold_mev_per_atom: Allowed absolute delta threshold.
		prefer_highest_value: When true, search the sweep from largest to smallest value.

	Returns:
		The first converged run, or ``None`` if none satisfy the criterion.
	"""
	if reference_energy_selector == "min":
		reference = min(case_results, key=lambda item: item.manifest.value)
	else:
		reference = max(case_results, key=lambda item: item.manifest.value)
	ordered_results = sorted(case_results, key=lambda item: item.manifest.value, reverse=prefer_highest_value)
	for result in ordered_results:
		delta_toten = mev_per_atom_delta(result.toten_ev, reference.toten_ev, result.n_atoms)
		delta_internal = mev_per_atom_delta(result.internal_ev, reference.internal_ev, result.n_atoms)
		if delta_toten <= threshold_mev_per_atom and delta_internal <= threshold_mev_per_atom:
			return result
	return None


def write_summary(
	output_path: Path,
	source_dir: Path,
	manifest_rows: list[ManifestRow],
	results: list[RunResult],
	skipped_rows: list[SkippedRun],
	kspacing_grouped: OrderedDict[str, list[RunResult]],
	encut_grouped: OrderedDict[str, list[RunResult]],
	system_label: str,
	threshold_mev_per_atom: float,
) -> None:
	"""Write a short Markdown summary of the benchmark analysis.

	Args:
		output_path: Destination Markdown file.
		source_dir: Source benchmark directory.
		manifest_rows: All manifest rows.
		results: All parsed and electronically converged results.
		skipped_rows: Rows excluded from the analysis.
		kspacing_grouped: KSPACING results grouped by case.
		encut_grouped: ENCUT results grouped by case.
		system_label: System label used in the report heading.
		threshold_mev_per_atom: Threshold used for the summary recommendation.
	"""
	lines: list[str] = []
	lines.append(f"# {system_label} static convergence benchmark")
	lines.append("")
	lines.append(f"- Source benchmark directory: `{source_dir}`")
	lines.append(f"- Manifest rows: `{len(manifest_rows)}`")
	lines.append(f"- Converged runs analyzed: `{len(results)}`")
	lines.append(f"- Incomplete or non-converged rows skipped: `{len(skipped_rows)}`")
	lines.append(f"- Shared runtime footprint: `{describe_resources(results)}`")
	if skipped_rows:
		lines.append("- Skipped rows:")
		for skipped_run in skipped_rows:
			manifest_row = skipped_run.manifest
			lines.append(
				f"  - `{manifest_row.family}` / `{manifest_row.case_name}` / `{display_run_label(manifest_row.label)}`: {skipped_run.reason}"
			)
	lines.append("")
	lines.append("## KSPACING observations")
	lines.append("")
	for case_name, case_results in kspacing_grouped.items():
		reference = min(case_results, key=lambda item: item.manifest.value)
		max_toten = max(mev_per_atom_delta(result.toten_ev, reference.toten_ev, result.n_atoms) for result in case_results)
		max_internal = max(
			mev_per_atom_delta(result.internal_ev, reference.internal_ev, result.n_atoms) for result in case_results
		)
		converged = find_first_converged(
			case_results,
			"min",
			threshold_mev_per_atom,
			prefer_highest_value=True,
		)
		converged_text = (
			f"`{display_run_label(converged.manifest.label)}` ({converged.manifest.value:.2f})"
			if converged is not None
			else f"none within {threshold_mev_per_atom:g} meV/atom"
		)
		runtime_minutes = [result.elapsed_seconds / 60.0 for result in case_results]
		lines.append(
			f"- `{case_name}`: reference is `{display_run_label(reference.manifest.label)}`; loosest completed setting within {threshold_mev_per_atom:g} meV/atom in both metrics is {converged_text}; max |dTOTEN| = `{max_toten:.4f}` meV/atom; max |dE_internal| = `{max_internal:.4f}` meV/atom; runtime span = `{min(runtime_minutes):.1f}` to `{max(runtime_minutes):.1f}` min."
		)
	lines.append("")
	lines.append("## ENCUT observations")
	lines.append("")
	for case_name, case_results in encut_grouped.items():
		reference = max(case_results, key=lambda item: item.manifest.value)
		max_toten = max(mev_per_atom_delta(result.toten_ev, reference.toten_ev, result.n_atoms) for result in case_results)
		max_internal = max(
			mev_per_atom_delta(result.internal_ev, reference.internal_ev, result.n_atoms) for result in case_results
		)
		converged = find_first_converged(
			case_results,
			"max",
			threshold_mev_per_atom,
			prefer_highest_value=False,
		)
		converged_text = (
			f"`{display_run_label(converged.manifest.label)}` ({int(converged.manifest.value)} eV)"
			if converged is not None
			else f"none within {threshold_mev_per_atom:g} meV/atom"
		)
		runtime_minutes = [result.elapsed_seconds / 60.0 for result in case_results]
		lines.append(
			f"- `{case_name}`: reference is `{display_run_label(reference.manifest.label)}`; lowest completed setting within {threshold_mev_per_atom:g} meV/atom in both metrics is {converged_text}; max |dTOTEN| = `{max_toten:.4f}` meV/atom; max |dE_internal| = `{max_internal:.4f}` meV/atom; runtime span = `{min(runtime_minutes):.1f}` to `{max(runtime_minutes):.1f}` min."
		)
	lines.append("")
	lines.append("## Runtime notes")
	lines.append("")
	lines.append(
		f"- Mean runtime across all converged runs: `{statistics.mean(result.elapsed_seconds for result in results) / 60.0:.1f}` min."
	)
	lines.append(
		f"- Slowest converged run: `{display_run_label(max(results, key=lambda item: item.elapsed_seconds).manifest.label)}` at `{max(result.elapsed_seconds for result in results) / 60.0:.1f}` min."
	)
	lines.append(
		f"- Fastest converged run: `{display_run_label(min(results, key=lambda item: item.elapsed_seconds).manifest.label)}` at `{min(result.elapsed_seconds for result in results) / 60.0:.1f}` min."
	)
	output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_plots_to_source(output_dir: Path, source_dir: Path) -> None:
	"""Copy generated plot PNGs to the benchmark source directory.

	Args:
		output_dir: Directory containing generated plot PNGs.
		source_dir: Benchmark source directory that should receive plot copies.
	"""
	for filename in (
		"kspacing_convergence_delta_mev_per_atom_static.png",
		"encut_convergence_delta_mev_per_atom_static.png",
	):
		source_path = output_dir / filename
		if source_path.is_file():
			shutil.copy2(source_path, source_dir / filename)


def main() -> None:
	"""Run the static VASP convergence analysis workflow."""
	args = parse_args()
	source_dir: Path = args.source_dir
	output_dir: Path = args.output_dir if args.output_dir is not None else source_dir
	system_label = args.system_label if args.system_label is not None else source_dir.name
	output_dir.mkdir(parents=True, exist_ok=True)

	manifest_path = resolve_manifest(source_dir, args.manifest)
	manifest_rows = read_manifest(manifest_path)
	skipped_rows = find_skipped_rows(manifest_rows)
	results = load_results(manifest_rows)
	if not results:
		raise RuntimeError(f"No electronically converged benchmark runs found under {source_dir}")

	kspacing_grouped = group_results(results, "KSPACING")
	encut_grouped = group_results(results, "ENCUT")

	kspacing_csv = output_dir / "kspacing_convergence_delta_mev_per_atom_static.csv"
	encut_csv = output_dir / "encut_convergence_delta_mev_per_atom_static.csv"
	kspacing_png = output_dir / "kspacing_convergence_delta_mev_per_atom_static.png"
	encut_png = output_dir / "encut_convergence_delta_mev_per_atom_static.png"
	summary_md = output_dir / "SUMMARY.md"

	write_kspacing_csv(kspacing_grouped, kspacing_csv, system_label)
	write_encut_csv(encut_grouped, encut_csv, system_label, args.encut_kspacing)
	plot_kspacing(kspacing_grouped, kspacing_png, system_label)
	plot_encut(encut_grouped, encut_png, system_label)
	write_summary(
		summary_md,
		source_dir,
		manifest_rows,
		results,
		skipped_rows,
		kspacing_grouped,
		encut_grouped,
		system_label,
		args.threshold_mev_per_atom,
	)
	if args.copy_plots_to_source:
		copy_plots_to_source(output_dir, source_dir)

	print(f"Wrote {kspacing_csv}")
	print(f"Wrote {encut_csv}")
	print(f"Wrote {kspacing_png}")
	print(f"Wrote {encut_png}")
	print(f"Wrote {summary_md}")
	if args.copy_plots_to_source:
		print(f"Copied plots to {source_dir}")


if __name__ == "__main__":
	main()
