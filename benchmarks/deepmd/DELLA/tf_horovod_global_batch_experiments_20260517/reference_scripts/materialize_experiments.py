#!/usr/bin/env python3
"""Create DeePMD TF/Horovod global-batch experiment run directories."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXISTING_ROOT = Path(
	"/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/"
	"sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/"
	"tf_hvd_apptainer300cuda126_bench_20260422"
)
DEFAULT_OUTPUT_ROOT = Path(
	"/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/"
	"sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/"
	"tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517"
)
WORKING_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = WORKING_ROOT / "EXPERIMENT_MATRIX.tsv"
REFERENCE_INPUT = DEFAULT_EXISTING_ROOT / "10k_gpu_scaling_loss_rerun/1gpu/myinput.json"


@dataclass(frozen=True)
class ExperimentCase:
	"""One planned or reused DeePMD training experiment."""

	case_id: str
	group: str
	status: str
	gpus: int
	nodes: int
	steps: int
	decay_steps: int
	scale_by_worker: str
	source_or_output: str


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Materialize the missing DeePMD global-batch experiment runs."
	)
	parser.add_argument(
		"--existing-root",
		type=Path,
		default=DEFAULT_EXISTING_ROOT,
		help="Root containing completed benchmark runs to reuse.",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=DEFAULT_OUTPUT_ROOT,
		help="Scratch root where new experiment directories will be created.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Print actions without creating files.",
	)
	return parser.parse_args()


def read_matrix(matrix_path: Path) -> list[ExperimentCase]:
	"""Read experiment cases from a TSV manifest.

	Args:
		matrix_path: Path to the tab-separated experiment manifest.

	Returns:
		Experiment cases in manifest order.
	"""
	with matrix_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, delimiter="\t")
		return [
			ExperimentCase(
				case_id=str(row["case_id"]),
				group=str(row["group"]),
				status=str(row["status"]),
				gpus=int(row["gpus"]),
				nodes=int(row["nodes"]),
				steps=int(row["steps"]),
				decay_steps=int(row["decay_steps"]),
				scale_by_worker=str(row["scale_by_worker"]),
				source_or_output=str(row["source_or_output"]),
			)
			for row in reader
		]


def load_reference_input(reference_input: Path) -> dict[str, Any]:
	"""Load the DeePMD JSON input used as the base for new runs.

	Args:
		reference_input: Existing validated DeePMD input JSON.

	Returns:
		Parsed JSON object.
	"""
	with reference_input.open("r", encoding="utf-8") as handle:
		data: dict[str, Any] = json.load(handle)
	return data


def build_input(reference_data: dict[str, Any], case: ExperimentCase) -> dict[str, Any]:
	"""Build a per-case DeePMD input with adjusted step and LR settings.

	Args:
		reference_data: Parsed base DeePMD input.
		case: Experiment case to materialize.

	Returns:
		Updated DeePMD input JSON object.
	"""
	data = json.loads(json.dumps(reference_data))
	training = data["training"]
	learning_rate = data["learning_rate"]
	training["numb_steps"] = case.steps
	training["save_freq"] = choose_save_freq(case.steps)
	training["disp_freq"] = 10
	learning_rate["decay_steps"] = case.decay_steps
	learning_rate["scale_by_worker"] = case.scale_by_worker
	return data


def choose_save_freq(steps: int) -> int:
	"""Choose a checkpoint interval that preserves useful validation checkpoints.

	Args:
		steps: Number of DeePMD optimizer steps in the run.

	Returns:
		Checkpoint save interval.
	"""
	if steps <= 1000:
		return max(25, steps // 5)
	if steps <= 5000:
		return 500
	return 1000


def write_json(path: Path, data: dict[str, Any], dry_run: bool) -> None:
	"""Write formatted JSON unless running in dry-run mode.

	Args:
		path: Destination JSON path.
		data: JSON-serializable object.
		dry_run: If true, skip filesystem writes.
	"""
	if dry_run:
		return
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2)
		handle.write("\n")


def build_sbatch(case: ExperimentCase) -> str:
	"""Build the Slurm launcher for one experiment case.

	Args:
		case: Experiment case to materialize.

	Returns:
		Complete sbatch script text.
	"""
	if case.gpus <= 4:
		return build_single_node_sbatch(case)
	return build_multi_node_sbatch(case)


def cpus_per_task(case: ExperimentCase) -> int:
	"""Select CPU allocation per Horovod rank.

	Args:
		case: Experiment case.

	Returns:
		CPUs per task for Slurm.
	"""
	if case.gpus == 1:
		return 8
	return 2


def memory_gb(case: ExperimentCase) -> int:
	"""Select memory per node for Slurm.

	Args:
		case: Experiment case.

	Returns:
		Memory in GB.
	"""
	if case.gpus == 1:
		return 60
	if case.gpus == 2:
		return 80
	return 120


def walltime(case: ExperimentCase) -> str:
	"""Select conservative walltime for Slurm.

	Args:
		case: Experiment case.

	Returns:
		Slurm walltime string.
	"""
	return "00:15:00"


def job_name(case: ExperimentCase) -> str:
	"""Make a compact Slurm job name.

	Args:
		case: Experiment case.

	Returns:
		Slurm job name no longer than typical display widths.
	"""
	return f"dpgb_{case.gpus}g_{case.steps}_{case.scale_by_worker}"[:32]


def build_single_node_sbatch(case: ExperimentCase) -> str:
	"""Build a single-node Slurm launcher.

	Args:
		case: Experiment case with at most four GPUs.

	Returns:
		Complete sbatch script text.
	"""
	cpus = cpus_per_task(case)
	mem = memory_gb(case)
	time_limit = walltime(case)
	name = job_name(case)
	return f"""#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks={case.gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:a100:{case.gpus}
#SBATCH --mem={mem}G
#SBATCH --time={time_limit}

# DeePMD TF/Horovod global-batch experiment.
# Case: {case.case_id}
# Steps: {case.steps}, decay_steps: {case.decay_steps}, scale_by_worker: {case.scale_by_worker}

set -euo pipefail

cd "${{SLURM_SUBMIT_DIR:-$(pwd)}}"

image="/scratch/gpfs/BURROWS/akashgpt/softwares/APPTAINER_REPO/deepmd-kit_3.0.0_cuda126.sif"
nproc={case.gpus}

export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export DP_INFER_BATCH_SIZE=32768
export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=2
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/deepmd-kit
export NCCL_DEBUG=WARN

echo "JOB_START $(date --iso-8601=seconds)"
echo "CASE {case.case_id}"
echo "HOST $(hostname)"
echo "IMAGE ${{image}}"
echo "NPROC ${{nproc}}"
nvidia-smi -L || true

monitor_log="gpu_mem_util_${{SLURM_JOB_ID}}.csv"
(
	echo "timestamp,index,uuid,name,memory.used.MiB,memory.total.MiB,utilization.gpu.percent,utilization.memory.percent,power.draw.W"
	while true; do
		nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits
		sleep 1
	done
) > "${{monitor_log}}" &
monitor_pid=$!
trap 'kill "${{monitor_pid}}" 2>/dev/null || true; wait "${{monitor_pid}}" 2>/dev/null || true' EXIT
echo "GPU_MONITOR_LOG ${{monitor_log}}"

srun --mpi=pmix --ntasks="${{nproc}}" --cpu-bind=cores --kill-on-bad-exit=1 \\
	apptainer exec --nv "${{image}}" env \\
	PYTHONNOUSERSITE="${{PYTHONNOUSERSITE}}" \\
	NCCL_DEBUG="${{NCCL_DEBUG}}" \\
	python -c "import os, socket, horovod.tensorflow as hvd, tensorflow as tf; hvd.init(); print('host', socket.gethostname(), 'hvd_rank', hvd.rank(), 'hvd_size', hvd.size(), 'local_rank', hvd.local_rank(), 'cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES'), 'n_tf_gpus', len(tf.config.list_physical_devices('GPU')))"

srun --mpi=pmix --ntasks="${{nproc}}" --cpu-bind=cores --kill-on-bad-exit=1 \\
	apptainer exec --nv "${{image}}" env \\
	PYTHONNOUSERSITE="${{PYTHONNOUSERSITE}}" \\
	HDF5_USE_FILE_LOCKING="${{HDF5_USE_FILE_LOCKING}}" \\
	DP_INFER_BATCH_SIZE="${{DP_INFER_BATCH_SIZE}}" \\
	OMP_NUM_THREADS="${{OMP_NUM_THREADS}}" \\
	DP_INTRA_OP_PARALLELISM_THREADS="${{DP_INTRA_OP_PARALLELISM_THREADS}}" \\
	DP_INTER_OP_PARALLELISM_THREADS="${{DP_INTER_OP_PARALLELISM_THREADS}}" \\
	TF_FORCE_GPU_ALLOW_GROWTH="${{TF_FORCE_GPU_ALLOW_GROWTH}}" \\
	XLA_FLAGS="${{XLA_FLAGS}}" \\
	NCCL_DEBUG="${{NCCL_DEBUG}}" \\
	dp train --mpi-log=workers --skip-neighbor-stat myinput.json

echo "JOB_END $(date --iso-8601=seconds)"
"""


def build_multi_node_sbatch(case: ExperimentCase) -> str:
	"""Build a multi-node Slurm launcher.

	Args:
		case: Experiment case with more than four GPUs.

	Returns:
		Complete sbatch script text.
	"""
	cpus = cpus_per_task(case)
	mem = memory_gb(case)
	time_limit = walltime(case)
	name = job_name(case)
	return f"""#!/bin/bash
#SBATCH --account=jiedeng
#SBATCH --qos=gpu-test
#SBATCH --job-name={name}
#SBATCH --nodes={case.nodes}
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:a100:4
#SBATCH --mem={mem}G
#SBATCH --time={time_limit}

# DeePMD TF/Horovod multi-node global-batch experiment.
# Case: {case.case_id}
# Nodes x GPUs/node: {case.nodes} x 4 = {case.gpus}
# Steps: {case.steps}, decay_steps: {case.decay_steps}, scale_by_worker: {case.scale_by_worker}

set -euo pipefail

cd "${{SLURM_SUBMIT_DIR:-$(pwd)}}"

image="/scratch/gpfs/BURROWS/akashgpt/softwares/APPTAINER_REPO/deepmd-kit_3.0.0_cuda126.sif"
nproc="${{SLURM_NTASKS:-$((SLURM_NNODES * 4))}}"

export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export DP_INFER_BATCH_SIZE=32768
export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=2
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/deepmd-kit
export NCCL_DEBUG=WARN

echo "JOB_START $(date --iso-8601=seconds)"
echo "CASE {case.case_id}"
echo "NODES ${{SLURM_NNODES}}  NTASKS ${{nproc}}  NODELIST ${{SLURM_JOB_NODELIST}}"
echo "IMAGE ${{image}}"
srun --ntasks-per-node=1 hostname || true
nvidia-smi -L || true

monitor_log="gpu_mem_util_${{SLURM_JOB_ID}}.csv"
(
	echo "timestamp,index,uuid,name,memory.used.MiB,memory.total.MiB,utilization.gpu.percent,utilization.memory.percent,power.draw.W"
	while true; do
		nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits
		sleep 1
	done
) > "${{monitor_log}}" &
monitor_pid=$!
trap 'kill "${{monitor_pid}}" 2>/dev/null || true; wait "${{monitor_pid}}" 2>/dev/null || true' EXIT
echo "GPU_MONITOR_LOG ${{monitor_log}}"

srun --mpi=pmix --ntasks="${{nproc}}" --cpu-bind=cores --kill-on-bad-exit=1 \\
	apptainer exec --nv "${{image}}" env \\
	PYTHONNOUSERSITE="${{PYTHONNOUSERSITE}}" \\
	NCCL_DEBUG="${{NCCL_DEBUG}}" \\
	python -c "import os, socket, horovod.tensorflow as hvd, tensorflow as tf; hvd.init(); print('host', socket.gethostname(), 'hvd_rank', hvd.rank(), 'hvd_size', hvd.size(), 'local_rank', hvd.local_rank(), 'cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES'), 'n_tf_gpus', len(tf.config.list_physical_devices('GPU')))"

srun --mpi=pmix --ntasks="${{nproc}}" --cpu-bind=cores --kill-on-bad-exit=1 \\
	apptainer exec --nv "${{image}}" env \\
	PYTHONNOUSERSITE="${{PYTHONNOUSERSITE}}" \\
	HDF5_USE_FILE_LOCKING="${{HDF5_USE_FILE_LOCKING}}" \\
	DP_INFER_BATCH_SIZE="${{DP_INFER_BATCH_SIZE}}" \\
	OMP_NUM_THREADS="${{OMP_NUM_THREADS}}" \\
	DP_INTRA_OP_PARALLELISM_THREADS="${{DP_INTRA_OP_PARALLELISM_THREADS}}" \\
	DP_INTER_OP_PARALLELISM_THREADS="${{DP_INTER_OP_PARALLELISM_THREADS}}" \\
	TF_FORCE_GPU_ALLOW_GROWTH="${{TF_FORCE_GPU_ALLOW_GROWTH}}" \\
	XLA_FLAGS="${{XLA_FLAGS}}" \\
	NCCL_DEBUG="${{NCCL_DEBUG}}" \\
	dp train --mpi-log=workers --skip-neighbor-stat myinput.json

echo "JOB_END $(date --iso-8601=seconds)"
"""


def run_info(case: ExperimentCase) -> str:
	"""Build a human-readable per-run note.

	Args:
		case: Experiment case to document.

	Returns:
		Markdown text.
	"""
	rank_batches = case.gpus * case.steps
	return f"""# {case.case_id}

Group: `{case.group}`

Purpose: `{case.status}` experiment from the global-batch matrix.

Settings:

```text
gpus = {case.gpus}
nodes = {case.nodes}
numb_steps = {case.steps}
decay_steps = {case.decay_steps}
scale_by_worker = {case.scale_by_worker}
approx_rank_batches = {rank_batches}
```

Submit from this directory with:

```bash
sbatch run_srun_train_mem.sbatch
```
"""


def materialize_case(
	case: ExperimentCase,
	output_root: Path,
	reference_data: dict[str, Any],
	dry_run: bool,
) -> None:
	"""Create files for one new experiment case.

	Args:
		case: Experiment case to create.
		output_root: Scratch root for the experiment bundle.
		reference_data: Parsed reference DeePMD JSON.
		dry_run: If true, print actions without writing.
	"""
	if case.status != "new":
		print(f"reuse\t{case.case_id}\t{case.source_or_output}")
		return

	run_dir = output_root / case.source_or_output
	print(f"create\t{case.case_id}\t{run_dir}")
	if dry_run:
		return

	run_dir.mkdir(parents=True, exist_ok=True)
	write_json(run_dir / "myinput.json", build_input(reference_data, case), dry_run=False)
	(run_dir / "run_srun_train_mem.sbatch").write_text(build_sbatch(case), encoding="utf-8")
	(run_dir / "RUN_INFO.md").write_text(run_info(case), encoding="utf-8")


def write_manifest_files(cases: Iterable[ExperimentCase], output_root: Path, dry_run: bool) -> None:
	"""Write helper manifest and submit scripts into the scratch root.

	Args:
		cases: Experiment cases from the manifest.
		output_root: Scratch root for generated files.
		dry_run: If true, skip filesystem writes.
	"""
	if dry_run:
		return
	output_root.mkdir(parents=True, exist_ok=True)
	matrix_text = MATRIX_PATH.read_text(encoding="utf-8")
	(output_root / "EXPERIMENT_MATRIX.tsv").write_text(matrix_text, encoding="utf-8")
	(output_root / "README.md").write_text((WORKING_ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
	write_submit_script(cases, output_root, "sample_matched")
	write_submit_script(cases, output_root, "lr_sensitivity")
	write_submit_script(cases, output_root, "walltime_matched")
	write_submit_script(cases, output_root, "walltime_and_long_baseline")


def write_submit_script(cases: Iterable[ExperimentCase], output_root: Path, group: str) -> None:
	"""Write a group-specific submission helper.

	Args:
		cases: Experiment cases from the manifest.
		output_root: Scratch root for generated files.
		group: Experiment group to include.
	"""
	lines = [
		"#!/bin/bash",
		"set -euo pipefail",
		"",
		f"# Submit only the {group} cases.",
		"# Review the matrix and queue state before running this.",
		"",
	]
	for case in cases:
		if case.status == "new" and case.group == group:
			lines.append(f"(cd {case.source_or_output} && sbatch run_srun_train_mem.sbatch)")
	lines.append("")
	path = output_root / f"submit_{group}.sh"
	path.write_text("\n".join(lines), encoding="utf-8")
	path.chmod(0o755)


def main() -> None:
	"""Materialize the experiment bundle."""
	args = parse_args()
	cases = read_matrix(MATRIX_PATH)
	reference_input = args.existing_root / "10k_gpu_scaling_loss_rerun/1gpu/myinput.json"
	reference_data = load_reference_input(reference_input)

	print(f"existing_root\t{args.existing_root}")
	print(f"output_root\t{args.output_root}")
	write_manifest_files(cases, args.output_root, args.dry_run)
	for case in cases:
		materialize_case(case, args.output_root, reference_data, args.dry_run)


if __name__ == "__main__":
	main()
