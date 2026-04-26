#!/usr/bin/env python3
"""Tile an orthogonal LAMMPS `atomic`-style data file by (nx, ny, nz).

The input file must have sections in the usual order: header metadata,
box bounds, optional `Atom Type Labels`, `Masses`, and `Atoms  # atomic`.
The output is written to `-o` and keeps types/masses identical, with
atoms replicated on a grid and wrapped into the new (scaled) box.

Why we use a small standalone tiler instead of LAMMPS `replicate`:
the benchmark harness supplies the data file directly to `read_data`,
and we want a concrete `conf_<N>.lmp` artifact on disk so timings and
reproducibility don't depend on runtime `replicate` commands.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def parse_conf(path: Path) -> dict:
	"""Parse an orthogonal atomic-style LAMMPS data file.

	Args:
		path: Path to the input data file.

	Returns:
		Dict with keys: header_comment, n_atoms, n_types, box (3 tuples of
		(lo, hi)), masses (list of "line" strings), atoms (list of
		(id, type, x, y, z)). Any `Atom Type Labels` block is ignored
		(commented or raw) because KOKKOS cannot consume it anyway.
	"""
	lines: List[str] = path.read_text().splitlines()
	i: int = 0
	header_comment: str = lines[0] if lines and lines[0].startswith("#") else ""

	n_atoms: int = 0
	n_types: int = 0
	box: List[Tuple[float, float]] = []

	# Header: read until we hit a named section.
	section_headers = {"Atoms", "Masses", "Atom Type Labels", "Velocities"}
	while i < len(lines):
		raw = lines[i].strip()
		if raw in section_headers or raw.startswith("Atoms ") or raw.startswith("Masses"):
			break
		if raw.endswith("atoms"):
			n_atoms = int(raw.split()[0])
		elif raw.endswith("atom types"):
			n_types = int(raw.split()[0])
		elif raw.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
			parts = raw.split()
			box.append((float(parts[0]), float(parts[1])))
		i += 1

	masses: List[str] = []
	atoms: List[Tuple[int, int, float, float, float]] = []

	while i < len(lines):
		raw = lines[i].strip()
		if raw.startswith("Masses"):
			i += 1
			while i < len(lines) and not lines[i].strip():
				i += 1
			while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("Atoms", "Atom Type Labels", "Velocities")):
				masses.append(lines[i].rstrip())
				i += 1
			continue
		if raw.startswith("Atom Type Labels"):
			# Skip — not portable to KOKKOS; types are already numeric below.
			i += 1
			while i < len(lines) and not lines[i].strip():
				i += 1
			while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("Atoms", "Masses", "Velocities")):
				i += 1
			continue
		if raw.startswith("Atoms"):
			i += 1
			while i < len(lines) and not lines[i].strip():
				i += 1
			for _ in range(n_atoms):
				parts = lines[i].split()
				atoms.append((int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
				i += 1
			continue
		i += 1

	return {
		"header_comment": header_comment,
		"n_atoms": n_atoms,
		"n_types": n_types,
		"box": box,
		"masses": masses,
		"atoms": atoms,
	}


def tile(conf: dict, nx: int, ny: int, nz: int) -> dict:
	"""Replicate a parsed conf (nx, ny, nz) times along each axis.

	Args:
		conf: Parsed dict from `parse_conf`.
		nx: Number of replicas along x.
		ny: Number of replicas along y.
		nz: Number of replicas along z.

	Returns:
		A new parsed-conf dict with scaled box and replicated atoms.
	"""
	lx = conf["box"][0][1] - conf["box"][0][0]
	ly = conf["box"][1][1] - conf["box"][1][0]
	lz = conf["box"][2][1] - conf["box"][2][0]

	new_box = [
		(conf["box"][0][0], conf["box"][0][0] + lx * nx),
		(conf["box"][1][0], conf["box"][1][0] + ly * ny),
		(conf["box"][2][0], conf["box"][2][0] + lz * nz),
	]

	new_atoms: List[Tuple[int, int, float, float, float]] = []
	next_id = 1
	for iz in range(nz):
		for iy in range(ny):
			for ix in range(nx):
				dx, dy, dz = ix * lx, iy * ly, iz * lz
				for (_aid, atype, x, y, z) in conf["atoms"]:
					new_atoms.append((next_id, atype, x + dx, y + dy, z + dz))
					next_id += 1

	return {
		"header_comment": conf["header_comment"] + f"  [tiled {nx}x{ny}x{nz}]",
		"n_atoms": len(new_atoms),
		"n_types": conf["n_types"],
		"box": new_box,
		"masses": conf["masses"],
		"atoms": new_atoms,
	}


def write_conf(conf: dict, path: Path) -> None:
	"""Write a parsed-conf dict back out as a LAMMPS atomic-style data file.

	Args:
		conf: Parsed-conf dict (possibly from `tile`).
		path: Output path.
	"""
	out: List[str] = []
	out.append(conf["header_comment"] if conf["header_comment"] else "# tiled LAMMPS data file")
	out.append("")
	out.append(f"{conf['n_atoms']} atoms")
	out.append(f"{conf['n_types']} atom types")
	out.append("")
	out.append(f"{conf['box'][0][0]:.6f} {conf['box'][0][1]:.6f} xlo xhi")
	out.append(f"{conf['box'][1][0]:.6f} {conf['box'][1][1]:.6f} ylo yhi")
	out.append(f"{conf['box'][2][0]:.6f} {conf['box'][2][1]:.6f} zlo zhi")
	out.append("")
	out.append("Masses")
	out.append("")
	out.extend(conf["masses"])
	out.append("")
	out.append("Atoms  # atomic")
	out.append("")
	for (aid, atype, x, y, z) in conf["atoms"]:
		out.append(f"{aid} {atype} {x:.6f} {y:.6f} {z:.6f}")
	out.append("")
	path.write_text("\n".join(out))


def main() -> None:
	"""CLI entry point for the tiler."""
	parser = argparse.ArgumentParser(description="Tile an orthogonal LAMMPS atomic data file.")
	parser.add_argument("-i", "--input", type=Path, required=True, help="Input conf.lmp path.")
	parser.add_argument("-o", "--output", type=Path, required=True, help="Output conf.lmp path.")
	parser.add_argument("--nx", type=int, required=True)
	parser.add_argument("--ny", type=int, required=True)
	parser.add_argument("--nz", type=int, required=True)
	args = parser.parse_args()

	conf = parse_conf(args.input)
	tiled = tile(conf, args.nx, args.ny, args.nz)
	write_conf(tiled, args.output)
	print(f"Wrote {args.output}: N={tiled['n_atoms']} atoms, box={tiled['box']}")


if __name__ == "__main__":
	main()
