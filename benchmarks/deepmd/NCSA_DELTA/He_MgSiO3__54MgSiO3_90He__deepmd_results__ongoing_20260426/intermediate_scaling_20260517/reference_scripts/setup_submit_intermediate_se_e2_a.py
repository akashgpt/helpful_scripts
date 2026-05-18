#!/usr/bin/env python3
"""Set up and submit intermediate se_e2_a TF training variants."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


TRAINING_ROOT = Path(
	"/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/"
	"deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/"
	"He_MgSiO3__54MgSiO3_90He/training_bench"
)
SHARED_ROOT = TRAINING_ROOT / "shared"
BASE_CONFIG = SHARED_ROOT / "train_se_e2_a.json"


@dataclass(frozen=True)
class Variant:
	"""Description of one intermediate training variant."""

	case_id: str
	shared_json: str
	run_dir: str
	job_name: str
	comment: str
	desc_neuron: list[int]
	fit_neuron: list[int]
	mem_per_cpu: str
	submit_stage: str


VARIANTS = [
	Variant(
		case_id="big2x",
		shared_json="train_se_e2_a_big2x.json",
		run_dir="variant_train_se_e2_a_TF_big2x",
		job_name="se_big2x",
		comment="Training-speed benchmark: se_e2_a width-only big2x, target about 2x params.",
		desc_neuron=[35, 70, 140],
		fit_neuron=[340, 340, 340],
		mem_per_cpu="48G",
		submit_stage="2x",
	),
	Variant(
		case_id="balanced_2x",
		shared_json="train_se_e2_a_balanced_2x.json",
		run_dir="variant_train_se_e2_a_TF_balanced_2x",
		job_name="se_bal2x",
		comment="Training-speed benchmark: se_e2_a balanced 2x, scales width and modest depth together.",
		desc_neuron=[30, 60, 120, 120],
		fit_neuron=[320, 320, 320, 320],
		mem_per_cpu="48G",
		submit_stage="2x",
	),
	Variant(
		case_id="big5x",
		shared_json="train_se_e2_a_big5x.json",
		run_dir="variant_train_se_e2_a_TF_big5x",
		job_name="se_big5x",
		comment="Training-speed benchmark: se_e2_a width-only big5x, target about 5x params.",
		desc_neuron=[56, 112, 224],
		fit_neuron=[540, 540, 540],
		mem_per_cpu="64G",
		submit_stage="5x",
	),
	Variant(
		case_id="balanced_5x",
		shared_json="train_se_e2_a_balanced_5x.json",
		run_dir="variant_train_se_e2_a_TF_balanced_5x",
		job_name="se_bal5x",
		comment="Training-speed benchmark: se_e2_a balanced 5x, scales width and depth together.",
		desc_neuron=[45, 90, 180, 180, 180],
		fit_neuron=[480, 480, 480, 480, 480],
		mem_per_cpu="64G",
		submit_stage="5x",
	),
	Variant(
		case_id="fit_deep5x",
		shared_json="train_se_e2_a_fit_deep5x.json",
		run_dir="variant_train_se_e2_a_TF_fit_deep5x",
		job_name="se_fit5x",
		comment="Training-speed benchmark: se_e2_a fitting-net depth-only 5x.",
		desc_neuron=[25, 50, 100],
		fit_neuron=[240] * 40,
		mem_per_cpu="64G",
		submit_stage="5x",
	),
]


def load_base_config() -> dict:
	"""Load the base se_e2_a training JSON."""
	with BASE_CONFIG.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def write_variant_json(variant: Variant, base_config: dict) -> Path:
	"""Write one shared training JSON for a variant.

	Args:
		variant: Variant to write.
		base_config: Base training config dictionary.

	Returns:
		Path to the written shared JSON.
	"""
	config = json.loads(json.dumps(base_config))
	config["_comment"] = variant.comment
	config["model"]["descriptor"]["neuron"] = variant.desc_neuron
	config["model"]["fitting_net"]["neuron"] = variant.fit_neuron
	output_path = SHARED_ROOT / variant.shared_json
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(config, handle, indent=2)
		handle.write("\n")
	return output_path


def render_submission_script(variant: Variant) -> str:
	"""Render a Slurm submission script for one variant.

	Args:
		variant: Variant to submit.

	Returns:
		Bash submission script text.
	"""
	run_dir = TRAINING_ROOT / variant.run_dir
	return f"""#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name={variant.job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu={variant.mem_per_cpu}
#SBATCH --time=48:00:00

set -e
module purge
module load PrgEnv-gnu
module load gcc-native/13.2
module load cray-mpich
module load cudatoolkit/25.3_12.8
module load fftw/3.3.10-gcc13.3.1
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate ALCHEMY_env
export PYTHONNOUSERSITE=1
export MPICH_GPU_SUPPORT_ENABLED=1

HERE="{run_dir}"
SHARED="$(cd "$HERE/../shared" && pwd)"
cp -f "$SHARED/{variant.shared_json}" "$HERE/input.json"
cd "$HERE"

echo "=== [$(date)] TF training ({variant.case_id}, 1,000,000 steps) ==="
dp --tf train input.json 2>&1 | tee log.train
echo "=== [$(date)] Done ==="
"""


def write_submission_script(variant: Variant) -> Path:
	"""Create a run directory and write its submission script.

	Args:
		variant: Variant to stage.

	Returns:
		Path to the run-directory submission script.
	"""
	run_dir = TRAINING_ROOT / variant.run_dir
	run_dir.mkdir(parents=False, exist_ok=True)
	submission_path = run_dir / "sub.sh"
	if any((run_dir / marker).exists() for marker in ["log.train", "lcurve.out", "checkpoint"]):
		raise RuntimeError(f"Refusing to overwrite active/completed run directory: {run_dir}")
	with submission_path.open("w", encoding="utf-8") as handle:
		handle.write(render_submission_script(variant))
	submission_path.chmod(0o755)
	return submission_path


def write_manifest() -> Path:
	"""Write a small manifest for the staged intermediate variants."""
	manifest_path = TRAINING_ROOT / "INTERMEDIATE_VARIANTS_20260517.tsv"
	with manifest_path.open("w", encoding="utf-8") as handle:
		handle.write("case_id\tstage\trun_dir\tshared_json\tdesc_neuron\tfit_neuron\n")
		for variant in VARIANTS:
			handle.write(
				f"{variant.case_id}\t{variant.submit_stage}\t{variant.run_dir}\t{variant.shared_json}\t"
				f"{variant.desc_neuron}\t{variant.fit_neuron}\n"
			)
	return manifest_path


def submit_job(script_path: Path, dependency: str | None = None) -> str:
	"""Submit one Slurm job and return its job id.

	Args:
		script_path: Submission script path.
		dependency: Optional Slurm dependency string.

	Returns:
		Submitted Slurm job id.
	"""
	command = ["sbatch", "--parsable"]
	if dependency is not None:
		command.append(f"--dependency={dependency}")
	command.append(str(script_path))
	result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=script_path.parent)
	return result.stdout.strip().split(";")[0]


def main() -> None:
	"""Stage all variants, submit 2x jobs first, then dependent 5x jobs."""
	base_config = load_base_config()
	scripts: dict[str, Path] = {}
	for variant in VARIANTS:
		json_path = write_variant_json(variant, base_config)
		script_path = write_submission_script(variant)
		subprocess.run(["bash", "-n", str(script_path)], check=True)
		scripts[variant.case_id] = script_path
		print(f"staged\t{variant.case_id}\t{json_path}\t{script_path}")
	manifest_path = write_manifest()
	print(f"manifest\t{manifest_path}")

	first_stage_ids: list[str] = []
	for variant in VARIANTS:
		if variant.submit_stage == "2x":
			job_id = submit_job(scripts[variant.case_id])
			first_stage_ids.append(job_id)
			print(f"submitted_2x\t{variant.case_id}\t{job_id}")

	dependency = "afterok:" + ":".join(first_stage_ids)
	for variant in VARIANTS:
		if variant.submit_stage == "5x":
			job_id = submit_job(scripts[variant.case_id], dependency=dependency)
			print(f"submitted_5x_dependent\t{variant.case_id}\t{job_id}\t{dependency}")


if __name__ == "__main__":
	main()
