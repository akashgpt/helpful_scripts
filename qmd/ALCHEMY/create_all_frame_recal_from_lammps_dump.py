#!/usr/bin/env python3
"""Create ALCHEMY-style VASP recalculation inputs for every LAMMPS dump frame.

This wrapper mirrors the relevant part of ALCHEMY Level 4:

1. Run ASAP on ``npt.dump`` to create ``ASAP-desc.xyz``.
2. Create an index file containing every frame in the dump, instead of using
	ASAP/FPS to select only a subset.
3. Run ALCHEMY's ``extract_deepmd.py`` inside ``pre/`` to build ``deepmd/``.
4. Correct ``deepmd/type_map.raw`` from ``conf.lmp`` atom type labels.
5. Run ALCHEMY's ``recal_dpdata_v3.py`` to build ``pre/recal/<frame>/POSCAR``
	and copy VASP inputs.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_MLDP_DIR = (
	"/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev/"
	"TRAIN_MLMD_scripts/ANALYSIS/mldp"
)
DEFAULT_ENV_COMMAND = "module load anaconda3/2025.12; conda activate ALCHEMY_env"
DEFAULT_ASAP_ENV_COMMAND = "module load anaconda3/2024.6; conda activate asap"
KB_J_PER_K = 1.3806504e-23
EV_TO_J = 1.602176634e-19


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments.

	Args:
		argv: Optional argument vector. If None, argparse reads from sys.argv.

	Returns:
		Parsed command-line arguments.
	"""
	parser = argparse.ArgumentParser(
		description=(
			"Use ALCHEMY's ASAP -> extract_deepmd -> recal_dpdata_v3 route to "
			"create VASP recalculation inputs for every frame in a LAMMPS dump."
		)
	)
	parser.add_argument(
		"lmp_dirs",
		nargs="+",
		type=Path,
		help="LAMMPS pressure/run directory containing npt.dump and conf.lmp.",
	)
	parser.add_argument(
		"--dump-name",
		default="npt.dump",
		help="LAMMPS dump filename inside each lmp_dir. Default: npt.dump.",
	)
	parser.add_argument(
		"--conf-name",
		default="conf.lmp",
		help="LAMMPS data filename used to read Atom Type Labels. Default: conf.lmp.",
	)
	parser.add_argument(
		"--vasp-input-dir",
		type=Path,
		help=(
			"Directory containing ALCHEMY-style INCAR.template, POTCAR, and optionally KPOINTS. "
			"If omitted, the script searches for setup_DIR/DFT first, then a sibling VASP directory."
		),
	)
	parser.add_argument(
		"--mldp-dir",
		type=Path,
		help=(
			"ALCHEMY mldp directory. Default order: --mldp-dir, $ALCHEMY__main__MLDP, "
			"MLDP_SCRIPTS from parameter file, MY_MLMD_SCRIPTS/TRAIN_MLMD_scripts/ANALYSIS/mldp."
		),
	)
	parser.add_argument(
		"--parameter-file",
		type=Path,
		help=(
			"TRAIN_MLMD_parameters.txt used to read DEEPMD_ENV_* and ASAP_ENV_*. "
			"If omitted, the script searches upward from each lmp_dir."
		),
	)
	parser.add_argument(
		"--env-cluster",
		choices=("primary", "secondary"),
		default="primary",
		help="Use PRIMARY or SECONDARY environment commands from TRAIN_MLMD_parameters.txt. Default: primary.",
	)
	parser.add_argument(
		"--asap-env",
		help="Override shell command used before running asap. Default: ASAP_ENV_* from parameter file.",
	)
	parser.add_argument(
		"--deepmd-env",
		help="Override shell command used before running extract_deepmd/recal_dpdata. Default: DEEPMD_ENV_* from parameter file.",
	)
	parser.add_argument(
		"--recal-temp",
		default="auto",
		help="Temperature in K for extract_deepmd fparam. Use 'auto' to read TEMP from in.lammps_npt_eq.",
	)
	parser.add_argument(
		"--lammps-input-name",
		default="in.lammps_npt_eq",
		help=(
			"LAMMPS input filename used when --recal-temp auto is set. If missing, "
			"the script searches for in.lammps* in the LAMMPS run directory."
		),
	)
	parser.add_argument(
		"--crossover",
		action="store_true",
		help="Pass --crossover to asap gen_desc, matching SOAP_CROSSOVER_FRAME_SELECTION=1.",
	)
	parser.add_argument(
		"--backup-existing-pre",
		action="store_true",
		help="Move an existing pre/ directory to old_pre__YYYYmmdd_HHMMSS before rebuilding.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print commands without running them or writing generated outputs.",
	)
	return parser.parse_args(argv)


def find_parameter_file(lmp_dir: Path, explicit_parameter_file: Path | None) -> Path | None:
	"""Find the relevant TRAIN_MLMD_parameters.txt file.

	Args:
		lmp_dir: LAMMPS pressure/run directory.
		explicit_parameter_file: User-provided parameter-file path, if any.

	Returns:
		Path to the parameter file, or None if one cannot be found.
	"""
	if explicit_parameter_file is not None:
		return explicit_parameter_file.resolve()
	for candidate_dir in (lmp_dir, *lmp_dir.parents):
		candidate = candidate_dir / "TRAIN_MLMD_parameters.txt"
		if candidate.is_file():
			return candidate
	return None


def read_parameter_value(parameter_file: Path, key: str) -> str | None:
	"""Read a one-line ALCHEMY parameter value from the following line.

	ALCHEMY parameter files store a key/comment line followed by the active value
	on the next line. Blank lines are skipped.

	Args:
		parameter_file: Path to TRAIN_MLMD_parameters.txt.
		key: Parameter key to read.

	Returns:
		The raw value line with surrounding whitespace removed, or None if absent.
	"""
	lines = parameter_file.read_text(encoding="utf-8", errors="replace").splitlines()
	for line_index, line in enumerate(lines):
		if not line.strip().startswith(key):
			continue
		for value_line in lines[line_index + 1:]:
			value = value_line.strip()
			if value:
				return value
		return None
	return None


def choose_parameter_env(
	parameter_file: Path | None,
	env_cluster: str,
	deepmd_override: str | None,
	asap_override: str | None,
) -> tuple[str, str]:
	"""Choose DeepMD and ASAP activation commands.

	Args:
		parameter_file: TRAIN_MLMD_parameters.txt path, if available.
		env_cluster: Either ``primary`` or ``secondary``.
		deepmd_override: Explicit DeepMD environment command.
		asap_override: Explicit ASAP environment command.

	Returns:
		DeepMD environment command and ASAP environment command.
	"""
	if deepmd_override is not None and asap_override is not None:
		return deepmd_override, asap_override

	deepmd_env = deepmd_override
	asap_env = asap_override
	if parameter_file is not None:
		suffix = "PRIMARY_CLUSTER" if env_cluster == "primary" else "SECONDARY_CLUSTER"
		if deepmd_env is None:
			deepmd_env = read_parameter_value(parameter_file, f"DEEPMD_ENV_{suffix}")
		if asap_env is None:
			asap_env = read_parameter_value(parameter_file, f"ASAP_ENV_{suffix}")

	if deepmd_env is None:
		deepmd_env = os.environ.get("DEEPMD_ENV", DEFAULT_ENV_COMMAND)
	if asap_env is None:
		asap_env = os.environ.get("ASAP_ENV", DEFAULT_ASAP_ENV_COMMAND)
	return deepmd_env, asap_env


def choose_mldp_dir(args: argparse.Namespace, parameter_file: Path | None) -> Path:
	"""Choose the ALCHEMY mldp helper directory.

	Args:
		args: Parsed command-line arguments.
		parameter_file: TRAIN_MLMD_parameters.txt path, if available.

	Returns:
		Path to the mldp helper directory.
	"""
	if args.mldp_dir is not None:
		return args.mldp_dir.resolve()

	env_mldp_dir = os.environ.get("ALCHEMY__main__MLDP")
	if env_mldp_dir:
		return Path(env_mldp_dir).resolve()

	if parameter_file is not None:
		mldp_scripts = read_parameter_value(parameter_file, "MLDP_SCRIPTS")
		if mldp_scripts:
			return Path(mldp_scripts).resolve()

		my_mlmd_scripts = read_parameter_value(parameter_file, "MY_MLMD_SCRIPTS")
		if my_mlmd_scripts:
			return (Path(my_mlmd_scripts) / "TRAIN_MLMD_scripts" / "ANALYSIS" / "mldp").resolve()

	return Path(DEFAULT_MLDP_DIR).resolve()


def run_shell_command(command: str, workdir: Path, dry_run: bool = False) -> None:
	"""Run a shell command in a chosen working directory.

	Args:
		command: Shell command to execute.
		workdir: Directory where the command should run.
		dry_run: If True, print the command and do not execute it.

	Raises:
		subprocess.CalledProcessError: If the command exits with a nonzero code.
	"""
	print(f"[{workdir}] $ {command}")
	if dry_run:
		return
	subprocess.run(command, cwd=workdir, shell=True, check=True, executable="/bin/bash")


def count_dump_frames(dump_path: Path) -> int:
	"""Count frames in a LAMMPS dump by counting TIMESTEP markers.

	Args:
		dump_path: Path to a LAMMPS dump file.

	Returns:
		Number of frames in the dump.

	Raises:
		ValueError: If no frames are found.
	"""
	num_frames = 0
	with dump_path.open("r", encoding="utf-8", errors="replace") as dump_file:
		for line in dump_file:
			if line.strip() == "ITEM: TIMESTEP":
				num_frames += 1
	if num_frames == 0:
		raise ValueError(f"No ITEM: TIMESTEP frames found in {dump_path}")
	return num_frames


def write_all_frame_index(index_path: Path, num_frames: int, dry_run: bool = False) -> None:
	"""Write an ALCHEMY-compatible zero-based index file containing all frames.

	Args:
		index_path: Output index file path.
		num_frames: Number of dump frames to include.
		dry_run: If True, do not write the file.
	"""
	print(f"Writing all-frame index: {index_path} ({num_frames} frames)")
	if dry_run:
		return
	with index_path.open("w", encoding="utf-8") as index_file:
		for frame_index in range(num_frames):
			index_file.write(f"{frame_index}\n")


def infer_system_dir(lmp_dir: Path) -> Path:
	"""Infer the system directory from a LAMMPS pressure/run directory.

	Args:
		lmp_dir: LAMMPS pressure/run directory such as ``.../SYSTEM/MD/1_GPa``.

	Returns:
		The inferred ``SYSTEM`` directory.
	"""
	if lmp_dir.parent.name in {"LMP", "MD"}:
		return lmp_dir.parent.parent
	return lmp_dir.parent


def find_default_vasp_input_dir(lmp_dir: Path) -> Path:
	"""Find the ALCHEMY-style VASP input/template directory.

	For validation paths like ``.../H2_NH3/SYSTEM/MD/1_GPa``, this prefers
	``.../H2_NH3/setup_DIR/DFT``. The older per-system ``SYSTEM/VASP`` layout is
	kept as a fallback.

	Args:
		lmp_dir: LAMMPS pressure/run directory.

	Returns:
		Detected VASP input/template directory.

	Raises:
		FileNotFoundError: If no usable input directory can be found.
	"""
	system_dir = infer_system_dir(lmp_dir)
	candidates = [
		system_dir.parent / "setup_DIR" / "DFT",
		system_dir / "VASP",
	]
	for candidate in candidates:
		if candidate.is_dir():
			return candidate
	raise FileNotFoundError(
		"Could not infer VASP input/template directory. Pass --vasp-input-dir explicitly. "
		f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
	)


def input_uses_kspacing(incar_path: Path) -> bool:
	"""Check whether an INCAR or INCAR template uses KSPACING.

	Args:
		incar_path: Path to an INCAR-like file.

	Returns:
		True if a non-comment KSPACING assignment is present.
	"""
	if not incar_path.is_file():
		return False
	with incar_path.open("r", encoding="utf-8", errors="replace") as incar_file:
		for line in incar_file:
			stripped_line = line.strip()
			if not stripped_line or stripped_line.startswith(("#", "!")):
				continue
			if stripped_line.upper().startswith("KSPACING") and "=" in stripped_line:
				return True
	return False


def validate_vasp_inputs(vasp_input_dir: Path) -> None:
	"""Validate the VASP input or template files needed by recal_dpdata_v3.py.

	Args:
		vasp_input_dir: Directory containing VASP inputs or ALCHEMY templates.

	Raises:
		FileNotFoundError: If required files are missing.
	"""
	incar_path = vasp_input_dir / "INCAR"
	incar_template_path = vasp_input_dir / "INCAR.template"
	active_incar_path = incar_template_path if incar_template_path.is_file() else incar_path
	if not active_incar_path.is_file():
		raise FileNotFoundError(f"Required VASP INCAR or INCAR.template is missing in: {vasp_input_dir}")
	potcar_path = vasp_input_dir / "POTCAR"
	if not potcar_path.is_file():
		raise FileNotFoundError(f"Required VASP input is missing: {potcar_path}")
	kpoints_path = vasp_input_dir / "KPOINTS"
	if not kpoints_path.is_file() and not input_uses_kspacing(active_incar_path):
		raise FileNotFoundError(
			f"KPOINTS is missing in {vasp_input_dir}, and {active_incar_path.name} does not define KSPACING."
		)


def find_lammps_input_path(lmp_dir: Path, preferred_name: str) -> Path:
	"""Find the LAMMPS input file used to extract the target temperature.

	Args:
		lmp_dir: LAMMPS pressure/run directory.
		preferred_name: Preferred input filename, usually ``in.lammps_npt_eq``.

	Returns:
		Path to the LAMMPS input file.

	Raises:
		FileNotFoundError: If no ``in.lammps*`` file is found.
	"""
	preferred_path = lmp_dir / preferred_name
	if preferred_path.is_file():
		return preferred_path
	candidates = sorted(lmp_dir.glob("in.lammps*"))
	if candidates:
		return candidates[0]
	raise FileNotFoundError(f"Could not find {preferred_name} or any in.lammps* file in {lmp_dir}")


def read_lammps_temperature(lammps_input_path: Path) -> float:
	"""Read the TEMP variable from a LAMMPS input file.

	Args:
		lammps_input_path: Path to the LAMMPS input file.

	Returns:
		Temperature in K.

	Raises:
		ValueError: If no TEMP variable is found.
	"""
	pattern = re.compile(r"^\s*variable\s+TEMP\s+equal\s+([0-9.eE+-]+)", re.IGNORECASE)
	with lammps_input_path.open("r", encoding="utf-8", errors="replace") as lammps_input:
		for line in lammps_input:
			match = pattern.search(line)
			if match:
				return float(match.group(1))
	raise ValueError(f"Could not find 'variable TEMP equal ...' in {lammps_input_path}")


def format_temperature_label(recal_temp: float) -> str:
	"""Format a temperature for ALCHEMY-style ``<TEMP>K`` input folders.

	Args:
		recal_temp: Temperature in K.

	Returns:
		Compact temperature label without unnecessary decimal zeros.
	"""
	return f"{recal_temp:g}"


def calculate_sigma_kb(recal_temp: float) -> float:
	"""Calculate ALCHEMY's VASP Fermi smearing width for a temperature.

	Args:
		recal_temp: Temperature in K.

	Returns:
		``kB * T`` in eV, using the constants from ALCHEMY Level 4.
	"""
	return KB_J_PER_K * recal_temp / EV_TO_J


def render_incar_template(template_path: Path, target_path: Path, recal_temp: float) -> None:
	"""Render an ALCHEMY INCAR template for one recalculation temperature.

	Args:
		template_path: Source ``INCAR*.template`` path.
		target_path: Rendered INCAR path.
		recal_temp: Temperature in K.
	"""
	temperature_label = format_temperature_label(recal_temp)
	sigma_kb = calculate_sigma_kb(recal_temp)
	contents = template_path.read_text(encoding="utf-8", errors="replace")
	contents = contents.replace("__TEBEG__", temperature_label)
	contents = contents.replace("__TEEND__", temperature_label)
	contents = contents.replace("__SIGMA__", str(sigma_kb))
	target_path.write_text(contents, encoding="utf-8")


def copy_optional_vasp_files(source_dir: Path, target_dir: Path) -> None:
	"""Copy optional VASP helper files into a materialized input directory.

	Args:
		source_dir: Directory containing template and helper files.
		target_dir: Rendered ``<TEMP>K`` VASP input directory.
	"""
	for filename in ("POTCAR", "KPOINTS", "RUN_VASP.sh", "MULTI_RUN_VASP.sh"):
		source_path = source_dir / filename
		if source_path.is_file():
			shutil.copy2(source_path, target_dir / filename)


def materialize_vasp_input_dir(vasp_input_dir: Path, recal_temp: float, dry_run: bool = False) -> Path:
	"""Create the ALCHEMY-style temperature-specific VASP input directory.

	If ``INCAR.template`` exists, the rendered directory is
	``vasp_input_dir/<TEMP>K``. Otherwise, ``vasp_input_dir`` is treated as an
	already-materialized input directory.

	Args:
		vasp_input_dir: Base VASP template/input directory.
		recal_temp: Temperature in K.
		dry_run: If True, report actions without writing files.

	Returns:
		Directory that should be passed to ``recal_dpdata_v3.py -if``.
	"""
	validate_vasp_inputs(vasp_input_dir)
	incar_template_path = vasp_input_dir / "INCAR.template"
	if not incar_template_path.is_file():
		return vasp_input_dir

	temperature_label = format_temperature_label(recal_temp)
	rendered_input_dir = vasp_input_dir / f"{temperature_label}K"
	print(f"Materializing VASP inputs from {vasp_input_dir} -> {rendered_input_dir}")
	print(f"ALCHEMY SIGMA = kB*T = {calculate_sigma_kb(recal_temp)} eV")
	if dry_run:
		return rendered_input_dir

	rendered_input_dir.mkdir(parents=True, exist_ok=True)
	render_incar_template(incar_template_path, rendered_input_dir / "INCAR", recal_temp)
	incar_xtra_template_path = vasp_input_dir / "INCAR_xtra.template"
	if incar_xtra_template_path.is_file():
		render_incar_template(incar_xtra_template_path, rendered_input_dir / "INCAR_xtra", recal_temp)
	copy_optional_vasp_files(vasp_input_dir, rendered_input_dir)
	validate_vasp_inputs(rendered_input_dir)
	return rendered_input_dir


def copy_materialized_inputs_to_recal_folders(
	materialized_input_dir: Path,
	recal_dir: Path,
	dry_run: bool = False,
) -> None:
	"""Copy rendered VASP inputs into the ``pre/recal`` run layout.

	``MULTI_RUN_VASP.sh`` is a parent-level launcher, so it is copied only into
	``pre/recal``. Each numbered frame folder receives the single-frame inputs
	and a ``to_RUN`` marker for batch launch scripts.

	Args:
		materialized_input_dir: Directory with rendered VASP inputs and submission helpers.
		recal_dir: ``pre/recal`` directory created by ``recal_dpdata_v3.py``.
		dry_run: If True, report actions without writing files.
	"""
	if dry_run:
		print(f"Would copy materialized per-frame VASP inputs from {materialized_input_dir} into {recal_dir}/*")
		print(f"Would copy MULTI_RUN_VASP.sh from {materialized_input_dir} into {recal_dir}")
		print(f"Would touch to_RUN in each numbered folder under {recal_dir}")
		return
	if not recal_dir.is_dir():
		raise FileNotFoundError(f"recal_dpdata_v3.py did not create recal directory: {recal_dir}")

	multi_submit_path = materialized_input_dir / "MULTI_RUN_VASP.sh"
	if multi_submit_path.is_file():
		shutil.copy2(multi_submit_path, recal_dir / multi_submit_path.name)

	files_to_copy = [
		path
		for path in sorted(materialized_input_dir.iterdir())
		if path.is_file() and path.name != "MULTI_RUN_VASP.sh"
	]
	num_frame_dirs = 0
	for frame_dir in sorted(recal_dir.iterdir(), key=lambda path: int(path.name) if path.name.isdigit() else path.name):
		if not frame_dir.is_dir() or not frame_dir.name.isdigit():
			continue
		num_frame_dirs += 1
		for source_path in files_to_copy:
			shutil.copy2(source_path, frame_dir / source_path.name)
		(frame_dir / "to_RUN").touch()
	print(
		f"Copied {len(files_to_copy)} per-frame VASP input/helper files and to_RUN markers "
		f"into {num_frame_dirs} numbered folders under {recal_dir}"
	)
	if multi_submit_path.is_file():
		print(f"Copied MULTI_RUN_VASP.sh into parent recal folder: {recal_dir}")


def extract_atom_type_labels(conf_path: Path) -> list[str]:
	"""Extract atom type labels from the LAMMPS data file.

	Args:
		conf_path: Path to ``conf.lmp``.

	Returns:
		Atom type labels ordered by type ID.

	Raises:
		FileNotFoundError: If the data file is missing.
		ValueError: If the Atom Type Labels section is missing or empty.
	"""
	labels: list[str] = []
	in_labels = False
	with conf_path.open("r", encoding="utf-8", errors="replace") as conf_file:
		for line in conf_file:
			stripped = line.strip()
			if stripped == "Atom Type Labels":
				in_labels = True
				continue
			if not in_labels:
				continue
			if not stripped:
				continue
			parts = stripped.split()
			if len(parts) >= 2 and parts[0].isdigit():
				labels.append(parts[1])
				continue
			if labels:
				break
	if not labels:
		raise ValueError(f"No Atom Type Labels found in {conf_path}")
	return labels


def update_type_map(type_map_path: Path, atom_type_labels: Iterable[str], dry_run: bool = False) -> None:
	"""Update DeepMD type_map.raw to match LAMMPS Atom Type Labels.

	Args:
		type_map_path: Path to ``deepmd/type_map.raw``.
		atom_type_labels: Ordered atom labels such as ``["H", "N"]``.
		dry_run: If True, do not write the file.
	"""
	labels = list(atom_type_labels)
	print(f"Writing type map {type_map_path}: {' '.join(labels)}")
	if dry_run:
		return
	type_map_path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def prepare_pre_directory(pre_dir: Path, backup_existing_pre: bool, dry_run: bool = False) -> None:
	"""Create a clean ``pre`` directory for ALCHEMY-style extraction.

	Args:
		pre_dir: Path to the pre directory.
		backup_existing_pre: If True, move an existing pre directory aside.
		dry_run: If True, do not modify the filesystem.

	Raises:
		FileExistsError: If pre exists and backup_existing_pre is False.
	"""
	if not pre_dir.exists():
		print(f"Creating {pre_dir}")
		if not dry_run:
			pre_dir.mkdir(parents=True)
		return
	if not backup_existing_pre:
		raise FileExistsError(
			f"{pre_dir} already exists. Re-run with --backup-existing-pre to move it aside."
		)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	backup_dir = pre_dir.with_name(f"old_pre__{timestamp}")
	print(f"Moving existing {pre_dir} to {backup_dir}")
	if not dry_run:
		shutil.move(str(pre_dir), str(backup_dir))
		pre_dir.mkdir(parents=True)


def build_asap_command(dump_name: str, crossover: bool) -> str:
	"""Build the ALCHEMY-style ASAP descriptor command.

	Args:
		dump_name: Dump filename passed to ASAP.
		crossover: Whether to include mixed-species SOAP terms.

	Returns:
		Shell-safe command string.
	"""
	parts = ["asap", "gen_desc", "--fxyz", dump_name, "soap"]
	if crossover:
		parts.append("--crossover")
	parts.extend(["-e", "-c", "6", "-n", "4", "-l", "4", "-g", "0.44"])
	return " ".join(parts)


def process_lmp_dir(lmp_dir: Path, args: argparse.Namespace) -> None:
	"""Process one LAMMPS directory through the all-frame ALCHEMY recal route.

	Args:
		lmp_dir: Directory containing ``npt.dump`` and ``conf.lmp``.
		args: Parsed command-line arguments.
	"""
	lmp_dir = lmp_dir.resolve()
	dump_path = lmp_dir / args.dump_name
	conf_path = lmp_dir / args.conf_name
	pre_dir = lmp_dir / "pre"
	base_vasp_input_dir = (args.vasp_input_dir or find_default_vasp_input_dir(lmp_dir)).resolve()
	parameter_file = find_parameter_file(lmp_dir, args.parameter_file)
	mldp_dir = choose_mldp_dir(args, parameter_file)
	deepmd_env, asap_env = choose_parameter_env(
		parameter_file,
		args.env_cluster,
		args.deepmd_env,
		args.asap_env,
	)

	if not dump_path.is_file():
		raise FileNotFoundError(f"LAMMPS dump not found: {dump_path}")
	if not conf_path.is_file():
		raise FileNotFoundError(f"LAMMPS conf not found: {conf_path}")
	validate_vasp_inputs(base_vasp_input_dir)

	if args.recal_temp == "auto":
		lammps_input_path = find_lammps_input_path(lmp_dir, args.lammps_input_name)
		recal_temp = read_lammps_temperature(lammps_input_path)
	else:
		lammps_input_path = None
		recal_temp = float(args.recal_temp)
	materialized_vasp_input_dir = materialize_vasp_input_dir(
		base_vasp_input_dir,
		recal_temp,
		dry_run=args.dry_run,
	)

	num_frames = count_dump_frames(dump_path)
	index_path = lmp_dir / f"test-frame-select-all-n-{num_frames}.index"

	print("")
	print("====================================================================")
	print(f"Processing: {lmp_dir}")
	print(f"Frames: {num_frames}")
	print(f"VASP input templates: {base_vasp_input_dir}")
	print(f"Rendered VASP inputs: {materialized_vasp_input_dir}")
	print(f"LAMMPS input: {lammps_input_path if lammps_input_path is not None else 'not used (--recal-temp override)'}")
	print(f"Parameter file: {parameter_file if parameter_file is not None else 'not found, using fallbacks'}")
	print(f"Environment source: {args.env_cluster}")
	print(f"ASAP env: {asap_env}")
	print(f"DeepMD env: {deepmd_env}")
	print(f"MLDP scripts: {mldp_dir}")
	print(f"RECAL_TEMP: {recal_temp:g} K")
	print("====================================================================")

	run_shell_command(
		f"{asap_env}; {build_asap_command(args.dump_name, args.crossover)}",
		workdir=lmp_dir,
		dry_run=args.dry_run,
	)
	if not args.dry_run and not (lmp_dir / "ASAP-desc.xyz").is_file():
		raise FileNotFoundError(f"ASAP did not create {lmp_dir / 'ASAP-desc.xyz'}")

	write_all_frame_index(index_path, num_frames, dry_run=args.dry_run)
	prepare_pre_directory(pre_dir, args.backup_existing_pre, dry_run=args.dry_run)

	extract_command = (
		f"{deepmd_env}; "
		f"python {mldp_dir / 'extract_deepmd.py'} "
		f"-f ../{args.dump_name} -fmt dump -id ../{index_path.name} -st -t {recal_temp:g}"
	)
	run_shell_command(extract_command, workdir=pre_dir, dry_run=args.dry_run)

	type_map_path = pre_dir / "deepmd" / "type_map.raw"
	if not args.dry_run and not type_map_path.is_file():
		raise FileNotFoundError(f"extract_deepmd.py did not create {type_map_path}")
	update_type_map(type_map_path, extract_atom_type_labels(conf_path), dry_run=args.dry_run)

	recal_command = (
		f"{deepmd_env}; "
		f"python {mldp_dir / 'recal_dpdata_v3.py'} "
		f"-d deepmd/ -if {materialized_vasp_input_dir} -sc sbatch -rv 0"
	)
	run_shell_command(recal_command, workdir=pre_dir, dry_run=args.dry_run)
	copy_materialized_inputs_to_recal_folders(
		materialized_vasp_input_dir,
		pre_dir / "recal",
		dry_run=args.dry_run,
	)

	print(f"Done: VASP frame inputs should be under {pre_dir / 'recal'}")


def main(argv: Sequence[str] | None = None) -> int:
	"""Run the all-frame ALCHEMY recal-preparation workflow.

	Args:
		argv: Optional command-line argument vector.

	Returns:
		Process exit code.
	"""
	args = parse_args(argv)
	try:
		for lmp_dir in args.lmp_dirs:
			process_lmp_dir(lmp_dir, args)
	except Exception as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
