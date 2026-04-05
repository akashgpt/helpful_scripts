#!/usr/bin/env python3
"""Extract all frames from DeePMD datasets (npy format) into numbered POSCAR folders.

Run this script from a top-level directory.  It recursively searches for every
``deepmd/`` folder (at any depth) and unpacks each one.  Numbered folders
1/, 2/, 3/, ... are created *alongside* the respective ``deepmd/`` folder,
each containing a POSCAR, POTCAR, INCAR, and a to_RUN marker file.

DeePMD data layout expected inside each deepmd/ folder:
    deepmd/
        type.raw          – atom type indices (one per atom)
        type_map.raw      – element names (one per line)
        set.000/
            coord.npy     – (nframes, natoms*3) atomic coordinates [Angstrom]
            box.npy       – (nframes, 9) cell vectors flattened row-major [Angstrom]
            fparam.npy    – (nframes,) frame parameters (kB*T in eV)
        set.001/          – (optional additional sets)
            ...

Usage:
    cd /path/to/top_level_directory/
    python3 extract_deepmd_to_poscars.py [--setup-dir /path/to/setup_dir/]

    --setup-dir defaults to the directory containing this script, so if your
    input files live alongside the script you can just run:
        python3 extract_deepmd_to_poscars.py

    The setup directory must contain the following VASP input files
    (the script will error out if any are missing):
        setup_dir/
            POTCAR__Mg        – elemental POTCARs; concatenated into a combined POTCAR.
            POTCAR__Si          One POTCAR__X file is required for each element in
            POTCAR__O           the deepmd type_map.raw.
            POTCAR__Fe
            POTCAR__H
            ...
            INCAR             – INCAR template. Placeholders __TEBEG__ and __SIGMA__
                                are replaced with temperature (nearest {round_to_T} K from
                                fparam) and kB*TEBEG (eV). Requires fparam.npy in the
                                deepmd data; skipped with a warning if absent.
            KPOINTS           – copied as-is into each folder.
        round_to_T is set to 50 K by default (see below import statements), so temperatures derived from fparam are rounded to the nearest 50 K. Adjust this value in the script if you want a different rounding.
"""

import argparse
import io
import shutil
import sys
import numpy as np
from pathlib import Path
from typing import IO, List, Optional, Tuple

########################################################################
round_to_T = 50  # K; used for rounding temperatures derived from fparam
########################################################################

class TeeWriter(io.TextIOBase):
	"""Write to both a stream (e.g. stdout) and a log file simultaneously.

	Args:
		stream: Original output stream (stdout or stderr).
		log_file: Open file handle to write log output to.
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


def load_deepmd_data(
    deepmd_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    """Load all frames from a DeePMD dataset directory.

    Args:
        deepmd_dir: Path to the deepmd/ directory.

    Returns:
        coords: (total_frames, natoms, 3) array of Cartesian coordinates [Angstrom].
        boxes: (total_frames, 3, 3) array of cell vectors [Angstrom].
        types: (natoms,) array of integer atom type indices.
        type_map: List of element symbols, indexed by type index.
        fparams: (total_frames,) array of frame parameters (kB*T in eV), or None.
    """
    deepmd_path = Path(deepmd_dir)

    # Load type info
    types = np.loadtxt(deepmd_path / "type.raw", dtype=int)
    natoms = len(types)

    with open(deepmd_path / "type_map.raw") as f:
        type_map = [line.strip() for line in f if line.strip()]

    # Collect frames from all set.XXX directories (sorted)
    set_dirs = sorted(deepmd_path.glob("set.*"))
    if not set_dirs:
        print(f"ERROR: No set.* directories found in {deepmd_dir}", file=sys.stderr)
        sys.exit(1)

    all_coords = []
    all_boxes = []
    all_fparams = []
    has_fparam = False

    for set_dir in set_dirs:
        coord_file = set_dir / "coord.npy"
        box_file = set_dir / "box.npy"
        fparam_file = set_dir / "fparam.npy"

        if not coord_file.exists() or not box_file.exists():
            print(f"WARNING: Skipping {set_dir.name} — missing coord.npy or box.npy", file=sys.stderr)
            continue

        coords = np.load(coord_file)   # (nframes, natoms*3)
        boxes = np.load(box_file)       # (nframes, 9)

        all_coords.append(coords)
        all_boxes.append(boxes)

        if fparam_file.exists():
            has_fparam = True
            all_fparams.append(np.load(fparam_file).flatten())

    coords = np.concatenate(all_coords, axis=0)   # (total_frames, natoms*3)
    boxes = np.concatenate(all_boxes, axis=0)      # (total_frames, 9)
    fparams = np.concatenate(all_fparams, axis=0) if has_fparam else None

    # Reshape
    nframes = coords.shape[0]
    coords = coords.reshape(nframes, natoms, 3)
    boxes = boxes.reshape(nframes, 3, 3)

    return coords, boxes, types, type_map, fparams


def write_poscar(
    filepath: str,
    coords: np.ndarray,
    box: np.ndarray,
    types: np.ndarray,
    type_map: List[str],
    frame_idx: int,
) -> None:
    """Write a single frame as a VASP POSCAR file.

    Args:
        filepath: Output file path.
        coords: (natoms, 3) Cartesian coordinates [Angstrom].
        box: (3, 3) cell vectors, each row is a lattice vector [Angstrom].
        types: (natoms,) integer type indices.
        type_map: Element names indexed by type index.
        frame_idx: Frame number (for the comment line).
    """
    # Sort atoms by type (VASP convention: group by species)
    sort_idx = np.argsort(types, kind="stable")
    sorted_types = types[sort_idx]
    sorted_coords = coords[sort_idx]

    # Determine species order and counts
    unique_types = []
    counts = []
    for t in sorted_types:
        if not unique_types or unique_types[-1] != t:
            unique_types.append(t)
            counts.append(1)
        else:
            counts[-1] += 1

    species_names = [type_map[t] for t in unique_types]

    with open(filepath, "w") as f:
        f.write(f"Frame {frame_idx} extracted from DeePMD dataset\n")
        f.write("1.0\n")

        # Lattice vectors (rows of box)
        for i in range(3):
            f.write(f"  {box[i, 0]:20.14f}  {box[i, 1]:20.14f}  {box[i, 2]:20.14f}\n")

        # Species names and counts
        f.write("  " + "  ".join(species_names) + "\n")
        f.write("  " + "  ".join(str(c) for c in counts) + "\n")

        # Cartesian coordinates
        f.write("Cartesian\n")
        for i in range(len(sorted_coords)):
            f.write(f"  {sorted_coords[i, 0]:20.14f}  {sorted_coords[i, 1]:20.14f}  {sorted_coords[i, 2]:20.14f}\n")


KB_EV = 8.617333262e-5  # Boltzmann constant in eV/K


def fparam_to_tebeg(fparam_eV: float) -> int:
    """Convert fparam (kB*T in eV) to temperature rounded to the nearest {round_to_T} K.

    Args:
        fparam_eV: Frame parameter value (kB*T in eV).

    Returns:
        Temperature in K, rounded to the nearest multiple of {round_to_T}.
    """
    T_K = fparam_eV / KB_EV
    return int(round(T_K / round_to_T) * round_to_T)


def write_incar(incar_template: str, output_path: Path, tebeg: int) -> None:
    """Write an INCAR file with __TEBEG__ and __SIGMA__ substituted.

    Args:
        incar_template: Contents of the INCAR template file.
        output_path: Path to write the filled INCAR.
        tebeg: Temperature in K (nearest {round_to_T} K multiple).
    """
    sigma = tebeg * KB_EV
    content = incar_template.replace("__TEBEG__", str(tebeg))
    content = content.replace("__SIGMA__", f"{sigma:.6f}")
    with open(output_path, "w") as f:
        f.write(content)


def build_potcar(type_map: List[str], potcar_dir: Path, output_path: Path) -> None:
    """Build a combined POTCAR by concatenating elemental POTCAR files.

    Concatenates POTCAR__X files from potcar_dir in the order given by type_map.

    Args:
        type_map: List of element symbols in the order they appear in the POSCAR.
        potcar_dir: Directory containing POTCAR__X files.
        output_path: Path to write the combined POTCAR.
    """
    with open(output_path, "w") as outf:
        for element in type_map:
            potcar_file = potcar_dir / f"POTCAR__{element}"
            if not potcar_file.exists():
                print(f"ERROR: {potcar_file} not found", file=sys.stderr)
                sys.exit(1)
            with open(potcar_file) as inf:
                content = inf.read()
                outf.write(content)
                # Ensure newline between concatenated POTCAR blocks
                if content and not content.endswith("\n"):
                    outf.write("\n")


def find_deepmd_dirs(root: Path) -> List[Path]:
    """Recursively find all directories named 'deepmd' under root.

    Args:
        root: Top-level directory to search.

    Returns:
        Sorted list of Path objects pointing to deepmd/ directories.
    """
    return sorted(root.rglob("deepmd"))


def process_one_deepmd(
    deepmd_dir: Path,
    setup_dir: Path,
) -> int:
    """Unpack a single deepmd/ directory into numbered POSCAR folders alongside it.

    Numbered folders (1/, 2/, ...) are created in the *parent* of deepmd_dir.
    POTCAR__X, INCAR, and KPOINTS files are picked up from setup_dir
    automatically (whichever exist).

    Args:
        deepmd_dir: Path to the deepmd/ directory.
        setup_dir: Directory containing POTCAR__X, INCAR, KPOINTS files.

    Returns:
        Number of frames extracted.
    """
    parent = deepmd_dir.parent

    print(f"\nLoading DeePMD data from {deepmd_dir.resolve()} ...")
    coords, boxes, types, type_map, fparams = load_deepmd_data(str(deepmd_dir))
    nframes = coords.shape[0]
    natoms = coords.shape[1]

    print(f"  Found {nframes} frame(s), {natoms} atoms/frame")
    print(f"  Type map: {type_map}")
    print(f"  Atom type counts: {dict(zip(type_map, [int(np.sum(types == i)) for i in range(len(type_map))]))}")

    # Validate that all required VASP input files exist in setup_dir
    missing_files: List[str] = []

    # Check for POTCAR__X files matching the type_map
    potcar_files = [setup_dir / f"POTCAR__{elem}" for elem in type_map]
    missing_potcars = [pf.name for pf in potcar_files if not pf.exists()]
    if missing_potcars:
        missing_files.extend(missing_potcars)
    else:
        print(f"  POTCAR files: {' + '.join(f'POTCAR__{e}' for e in type_map)}")

    # Check for INCAR template
    incar_path = setup_dir / "INCAR"
    if not incar_path.is_file():
        missing_files.append("INCAR")
    else:
        print(f"  INCAR template: {incar_path.resolve()}")

    # Check for KPOINTS
    kpoints_path = setup_dir / "KPOINTS"
    if not kpoints_path.is_file():
        missing_files.append("KPOINTS")
    else:
        print(f"  KPOINTS file: {kpoints_path.resolve()}")

    if missing_files:
        print(f"ERROR: Missing required VASP input files in setup_dir "
              f"({setup_dir.resolve()}):\n  " +
              "\n  ".join(missing_files), file=sys.stderr)
        sys.exit(1)

    incar_template = incar_path.read_text()

    # Compute per-frame temperatures from fparam
    tebegs: Optional[List[int]] = None
    if fparams is not None:
        tebegs = [fparam_to_tebeg(fp) for fp in fparams]
        unique_temps = sorted(set(tebegs))
        print(f"  Temperatures (nearest {round_to_T} K): {unique_temps}")
    else:
        if incar_template is not None:
            print("  WARNING: INCAR found but no fparam.npy in this deepmd/; "
                  "INCAR will NOT be written.", file=sys.stderr)

    # Build the combined POTCAR once (same for all frames in this deepmd)
    combined_potcar = parent / "_combined_POTCAR.tmp"
    build_potcar(type_map, setup_dir, combined_potcar)

    # Create numbered folders alongside the deepmd/ directory
    for i in range(nframes):
        folder = parent / str(i + 1)  # 1-indexed
        folder.mkdir(exist_ok=True)

        write_poscar(
            filepath=str(folder / "POSCAR"),
            coords=coords[i],
            box=boxes[i],
            types=types,
            type_map=type_map,
            frame_idx=i + 1,
        )

        shutil.copy2(str(combined_potcar), str(folder / "POTCAR"))

        if tebegs is not None:
            write_incar(incar_template, folder / "INCAR", tebegs[i])

        shutil.copy2(str(kpoints_path), str(folder / "KPOINTS"))

        # Create empty marker file
        (folder / "to_RUN").touch()

    # Clean up temp file
    combined_potcar.unlink()

    extras = ["POTCAR"]
    if tebegs:
        extras.append("INCAR")
    extras.append("KPOINTS")
    extras.append("to_RUN")
    extras_msg = ", ".join(extras)
    print(f"  -> Created {nframes} folders (1 to {nframes}) in {parent.resolve()}/")
    print(f"     Each contains: POSCAR, {extras_msg}")

    return nframes


def main() -> None:
    """Find all deepmd/ directories and extract frames into numbered POSCAR folders."""
    # Tee all output (stdout + stderr) to a log file in the current directory
    log_path = Path("log.extract_deepmd_to_poscars")
    log_file = open(log_path, "w")
    assert sys.__stdout__ is not None and sys.__stderr__ is not None
    sys.stdout = TeeWriter(sys.__stdout__, log_file)
    sys.stderr = TeeWriter(sys.__stderr__, log_file)

    parser = argparse.ArgumentParser(
        description="Recursively find all deepmd/ directories and extract "
                    "DeePMD npy frames into numbered POSCAR folders alongside each."
    )
    script_dir = str(Path(__file__).resolve().parent)
    parser.add_argument(
        "--setup-dir",
        type=str,
        default=script_dir,
        help="Directory containing input files: POTCAR__X (elemental POTCARs), "
             "INCAR (template with __TEBEG__/__SIGMA__ placeholders), and/or "
             "KPOINTS. Only files that exist in the directory are used. "
             f"Defaults to the script's directory ({script_dir}).",
    )
    args = parser.parse_args()

    # Validate setup directory
    setup_dir = Path(args.setup_dir)
    if not setup_dir.is_dir():
        print(f"ERROR: Setup directory not found: {setup_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Setup directory: {setup_dir.resolve()}")

    # Recursively find all deepmd/ directories
    root = Path(".")
    deepmd_dirs = find_deepmd_dirs(root)

    if not deepmd_dirs:
        print("ERROR: No 'deepmd/' directories found anywhere under the "
              "current working directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(deepmd_dirs)} deepmd/ director{'y' if len(deepmd_dirs) == 1 else 'ies'}:")
    for d in deepmd_dirs:
        print(f"  {d.resolve()}")

    # Process each deepmd/ directory
    total_frames = 0
    for deepmd_dir in deepmd_dirs:
        if not deepmd_dir.is_dir():
            continue
        nframes = process_one_deepmd(deepmd_dir, setup_dir)
        total_frames += nframes

    print(f"\n{'='*60}")
    print(f"Done. Processed {len(deepmd_dirs)} deepmd/ director{'y' if len(deepmd_dirs) == 1 else 'ies'}, "
          f"{total_frames} total frames.")


if __name__ == "__main__":
    main()
