#!/usr/bin/env python3
"""Summarize all DeePMD datasets found recursively under the current directory.

Scans for every ``deepmd/`` folder, loads metadata and data arrays, and prints
a per-dataset table plus aggregate statistics covering:

- Number of frames and atoms
- Elemental composition (type_map and per-element counts)
- Temperature range (from fparam.npy, converted via kB)
- Energy per atom (from energy.npy)
- Volume per atom and pressure (from box.npy and virial.npy)

Usage:
    cd /path/to/top_level_directory/
    python3 summarize_deepmd.py

Output is printed to stdout and simultaneously saved to
``log.summarize_deepmd`` in the current working directory.
"""

import io
import sys
import numpy as np
from pathlib import Path
from typing import IO, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KB_EV = 8.617333262e-5          # Boltzmann constant [eV/K]
EV_PER_ANG3_TO_GPA = 160.21766 # 1 eV/Ang^3 = 160.21766 GPa


# ---------------------------------------------------------------------------
# Tee helper (mirror all output to a log file)
# ---------------------------------------------------------------------------
class TeeWriter(io.TextIOBase):
	"""Write to both a stream (e.g. stdout) and a log file simultaneously.

	Args:
		stream: Original output stream.
		log_file: Open file handle for the log.
	"""

	def __init__(self, stream: IO[str], log_file: IO[str]) -> None:
		self.stream = stream
		self.log_file = log_file

	def write(self, msg: str) -> int:
		"""Write msg to both the original stream and the log file."""
		self.stream.write(msg)
		self.log_file.write(msg)
		return len(msg)

	def flush(self) -> None:
		"""Flush both the original stream and the log file."""
		self.stream.flush()
		self.log_file.flush()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_deepmd_summary(deepmd_dir: Path) -> Dict:
	"""Load metadata and summary statistics from a single deepmd/ directory.

	Args:
		deepmd_dir: Path to the deepmd/ directory.

	Returns:
		Dictionary with keys: path, nframes, natoms, type_map, atom_counts,
		temperatures, energies_per_atom, volumes_per_atom, pressures_gpa.
		Array-valued entries are 1-D numpy arrays (one element per frame).
		Entries may be None if the corresponding .npy file is absent.
	"""
	types = np.loadtxt(deepmd_dir / "type.raw", dtype=int)
	natoms = len(types)

	with open(deepmd_dir / "type_map.raw") as f:
		type_map = [line.strip() for line in f if line.strip()]

	# Per-element atom counts
	atom_counts = {type_map[i]: int(np.sum(types == i)) for i in range(len(type_map))}

	# Collect data from all set.XXX directories
	set_dirs = sorted(deepmd_dir.glob("set.*"))
	all_fparams: List[np.ndarray] = []
	all_energies: List[np.ndarray] = []
	all_boxes: List[np.ndarray] = []
	all_virials: List[np.ndarray] = []
	total_frames = 0

	for sd in set_dirs:
		coord_file = sd / "coord.npy"
		if not coord_file.exists():
			continue
		coords = np.load(coord_file)
		nf = coords.shape[0]
		total_frames += nf

		# fparam -> temperature
		fp = sd / "fparam.npy"
		if fp.exists():
			all_fparams.append(np.load(fp).flatten())

		# energy
		en = sd / "energy.npy"
		if en.exists():
			all_energies.append(np.load(en).flatten())

		# box (for volume)
		bx = sd / "box.npy"
		if bx.exists():
			all_boxes.append(np.load(bx))

		# virial (for pressure)
		vr = sd / "virial.npy"
		if vr.exists():
			all_virials.append(np.load(vr))

	# Temperatures [K]
	temperatures = None
	if all_fparams:
		fparams = np.concatenate(all_fparams)
		temperatures = fparams / KB_EV

	# Energies per atom [eV/atom]
	energies_per_atom = None
	if all_energies:
		energies = np.concatenate(all_energies)
		energies_per_atom = energies / natoms

	# Volumes per atom [Ang^3/atom] and pressures [GPa]
	volumes_per_atom = None
	pressures_gpa = None
	if all_boxes:
		boxes = np.concatenate(all_boxes).reshape(-1, 3, 3)
		volumes = np.abs(np.linalg.det(boxes))  # Ang^3
		volumes_per_atom = volumes / natoms

		if all_virials:
			# virial is stored as (nframes, 9) in eV, row-major 3x3
			# DeePMD virial = stress * volume, so P = Tr(virial) / (3 * V)
			virials = np.concatenate(all_virials).reshape(-1, 3, 3)
			trace_virial = virials[:, 0, 0] + virials[:, 1, 1] + virials[:, 2, 2]
			pressures_eV_ang3 = trace_virial / (3.0 * volumes)
			pressures_gpa = pressures_eV_ang3 * EV_PER_ANG3_TO_GPA

	return {
		"path": deepmd_dir,
		"nframes": total_frames,
		"natoms": natoms,
		"type_map": type_map,
		"atom_counts": atom_counts,
		"temperatures": temperatures,
		"energies_per_atom": energies_per_atom,
		"volumes_per_atom": volumes_per_atom,
		"pressures_gpa": pressures_gpa,
	}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_range(arr: Optional[np.ndarray], fmt: str = ".1f") -> str:
	"""Format min–max range of an array, or 'N/A' if None.

	Args:
		arr: 1-D numpy array or None.
		fmt: Format specifier for the numbers.

	Returns:
		String like "1234.5 – 6789.0" or "N/A".
	"""
	if arr is None or len(arr) == 0:
		return "N/A"
	lo, hi = arr.min(), arr.max()
	return f"{lo:{fmt}} – {hi:{fmt}}"


def fmt_mean_std(arr: Optional[np.ndarray], fmt: str = ".1f") -> str:
	"""Format mean +/- std of an array, or 'N/A' if None.

	Args:
		arr: 1-D numpy array or None.
		fmt: Format specifier for the numbers.

	Returns:
		String like "1234.5 +/- 56.7" or "N/A".
	"""
	if arr is None or len(arr) == 0:
		return "N/A"
	return f"{arr.mean():{fmt}} +/- {arr.std():{fmt}}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
	"""Find all deepmd/ directories and print summary statistics."""
	# Set up tee logging
	log_path = Path("log.summarize_deepmd")
	log_file = open(log_path, "w")
	assert sys.__stdout__ is not None and sys.__stderr__ is not None
	sys.stdout = TeeWriter(sys.__stdout__, log_file)
	sys.stderr = TeeWriter(sys.__stderr__, log_file)

	# Find all deepmd/ directories
	root = Path(".")
	deepmd_dirs = sorted(root.rglob("deepmd"))
	deepmd_dirs = [d for d in deepmd_dirs if d.is_dir() and (d / "type.raw").exists()]

	if not deepmd_dirs:
		print("ERROR: No deepmd/ directories found under current working directory.",
			  file=sys.stderr)
		sys.exit(1)

	print(f"Scanning {len(deepmd_dirs)} deepmd/ directories under {root.resolve()}\n")

	# Load all datasets
	summaries: List[Dict] = []
	for dd in deepmd_dirs:
		try:
			s = load_deepmd_summary(dd)
			summaries.append(s)
		except Exception as e:
			print(f"WARNING: Failed to load {dd}: {e}", file=sys.stderr)

	# -----------------------------------------------------------------------
	# Per-dataset table
	# -----------------------------------------------------------------------
	print("=" * 100)
	print("PER-DATASET SUMMARY")
	print("=" * 100)

	# Header
	header = (f"{'#':>3}  {'Path':<55} {'Frames':>7} {'Atoms':>6} "
			  f"{'T range (K)':>22} {'P range (GPa)':>24}")
	print(header)
	print("-" * len(header))

	for i, s in enumerate(summaries, 1):
		# Show path relative to root for readability
		rel_path = str(s["path"])
		if len(rel_path) > 53:
			rel_path = "..." + rel_path[-50:]
		t_range = fmt_range(s["temperatures"], ".0f")
		p_range = fmt_range(s["pressures_gpa"], ".1f")
		print(f"{i:>3}  {rel_path:<55} {s['nframes']:>7} {s['natoms']:>6} "
			  f"{t_range:>22} {p_range:>24}")

	# -----------------------------------------------------------------------
	# Composition breakdown
	# -----------------------------------------------------------------------
	print(f"\n{'=' * 80}")
	print("COMPOSITION BREAKDOWN")
	print("=" * 80)

	# Group datasets by (type_map tuple, natoms) to find unique compositions
	comp_groups: Dict[Tuple, List[Dict]] = {}
	for s in summaries:
		key = (tuple(s["type_map"]), s["natoms"], tuple(sorted(s["atom_counts"].items())))
		comp_groups.setdefault(key, []).append(s)

	for (tm, na, ac_items), group in comp_groups.items():
		ac = dict(ac_items)
		formula = "  ".join(f"{elem}: {ac[elem]}" for elem in tm)
		total_frames = sum(s["nframes"] for s in group)
		print(f"\n  {formula}   (natoms={na}, {len(group)} dataset(s), {total_frames} total frames)")

	# -----------------------------------------------------------------------
	# Aggregate statistics
	# -----------------------------------------------------------------------
	print(f"\n{'=' * 80}")
	print("AGGREGATE STATISTICS")
	print("=" * 80)

	total_frames = sum(s["nframes"] for s in summaries)
	total_datasets = len(summaries)
	all_natoms = sorted(set(s["natoms"] for s in summaries))

	print(f"\n  Total datasets:        {total_datasets}")
	print(f"  Total frames:          {total_frames}")
	print(f"  Unique atom counts:    {all_natoms}")

	# Concatenate per-frame arrays across all datasets for global stats
	all_temps = [s["temperatures"] for s in summaries if s["temperatures"] is not None]
	all_energies = [s["energies_per_atom"] for s in summaries if s["energies_per_atom"] is not None]
	all_volumes = [s["volumes_per_atom"] for s in summaries if s["volumes_per_atom"] is not None]
	all_pressures = [s["pressures_gpa"] for s in summaries if s["pressures_gpa"] is not None]

	if all_temps:
		temps = np.concatenate(all_temps)
		print(f"\n  Temperature (K):")
		print(f"    Range:    {fmt_range(temps, '.0f')}")
		print(f"    Mean:     {fmt_mean_std(temps, '.0f')}")

	if all_energies:
		ens = np.concatenate(all_energies)
		print(f"\n  Energy per atom (eV/atom):")
		print(f"    Range:    {fmt_range(ens, '.4f')}")
		print(f"    Mean:     {fmt_mean_std(ens, '.4f')}")

	if all_volumes:
		vols = np.concatenate(all_volumes)
		print(f"\n  Volume per atom (Ang^3/atom):")
		print(f"    Range:    {fmt_range(vols, '.3f')}")
		print(f"    Mean:     {fmt_mean_std(vols, '.3f')}")

	if all_pressures:
		pres = np.concatenate(all_pressures)
		print(f"\n  Pressure (GPa):")
		print(f"    Range:    {fmt_range(pres, '.1f')}")
		print(f"    Mean:     {fmt_mean_std(pres, '.1f')}")

	# -----------------------------------------------------------------------
	# Temperature–Pressure distribution (frame counts in bins)
	# -----------------------------------------------------------------------
	if all_temps and all_pressures:
		print(f"\n{'=' * 80}")
		print("TEMPERATURE–PRESSURE DISTRIBUTION (frame counts)")
		print("=" * 80)

		temps_all = np.concatenate(all_temps)
		pres_all = np.concatenate(all_pressures)

		# Define bin edges based on data range
		t_min, t_max = temps_all.min(), temps_all.max()
		p_min, p_max = pres_all.min(), pres_all.max()

		# Round to nice boundaries
		t_edges = np.arange(
			np.floor(t_min / 1000) * 1000,
			np.ceil(t_max / 1000) * 1000 + 1001,
			1000,
		)
		p_edges = np.arange(
			np.floor(p_min / 50) * 50,
			np.ceil(p_max / 50) * 50 + 51,
			50,
		)

		# Limit bin count to keep the table readable
		if len(t_edges) > 20:
			t_edges = np.linspace(t_min, t_max, 12)
		if len(p_edges) > 30:
			p_edges = np.linspace(p_min, p_max, 16)

		hist, _, _ = np.histogram2d(temps_all, pres_all, bins=[t_edges, p_edges])

		# Print as a table: rows = T bins, columns = P bins
		# Column headers (pressure bin centers)
		p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
		t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

		col_w = 8
		header = f"{'T \\ P (GPa)':>14}"
		for pc in p_centers:
			header += f"{pc:>{col_w}.0f}"
		print(f"\n{header}")
		print("-" * len(header))

		for ti in range(len(t_centers)):
			row = f"{t_centers[ti]:>12.0f} K"
			for pi in range(len(p_centers)):
				count = int(hist[ti, pi])
				row += f"{count:>{col_w}}" if count > 0 else f"{'·':>{col_w}}"
			print(row)

	print(f"\n{'=' * 80}")
	print(f"Log saved to: {log_path.resolve()}")
	print("=" * 80)

	log_file.close()


if __name__ == "__main__":
	main()
