#!/usr/bin/env python3
"""Set up DPA-2 diagnostic training variants on the MgSiOH benchmark data.

The suite is deliberately short enough to diagnose configuration behavior before
spending full 1M-step training time. It writes only live run-directory inputs and
submission scripts; benchmark folders should keep compact summaries of the result.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


TRAINING_BENCH: Path = Path(
	"/work/nvme/bguf/akashgpt/softwares/installing_MLMD_related_stuff/"
	"deepmd-kit__w_plumed/testing__LAMMPS__kokkos_bench/"
	"He_MgSiO3__54MgSiO3_90He/training_bench"
)
SHARED_DIR: Path = TRAINING_BENCH / "shared"
SOURCE_CONFIG: Path = SHARED_DIR / "train_dpa2_v2.json"
NSTEPS: int = 200_000


def load_source_config() -> dict[str, Any]:
	"""Load the previous DPA-2 v2 config as the data/source scaffold."""
	with SOURCE_CONFIG.open(encoding="utf-8") as handle:
		return json.load(handle)


def base_training_block(source: dict[str, Any]) -> dict[str, Any]:
	"""Return a conservative training block using the same MgSiOH systems."""
	training: dict[str, Any] = copy.deepcopy(source["training"])
	training["numb_steps"] = NSTEPS
	training["disp_freq"] = 100
	training["save_freq"] = 1000
	training["save_ckpt"] = "model.ckpt"
	training["gradient_max_norm"] = 5.0
	training["training_data"]["batch_size"] = 1
	training["training_data"].pop("auto_prob", None)
	return training


def current_loss() -> dict[str, Any]:
	"""Return the same energy/force/virial loss used by previous benchmark runs."""
	return {
		"type": "ener",
		"start_pref_e": 0.04,
		"limit_pref_e": 2,
		"start_pref_f": 1000,
		"limit_pref_f": 1.5,
		"start_pref_v": 0.04,
		"limit_pref_v": 2,
	}


def no_virial_loss() -> dict[str, Any]:
	"""Return the official-example-style loss without virial training pressure."""
	return {
		"type": "ener",
		"start_pref_e": 0.02,
		"limit_pref_e": 1,
		"start_pref_f": 1000,
		"limit_pref_f": 1,
		"start_pref_v": 0,
		"limit_pref_v": 0,
	}


def learning_rate(decay_steps: int = 10_000) -> dict[str, Any]:
	"""Return a learning-rate schedule comparable to the previous benchmark."""
	return {
		"type": "exp",
		"start_lr": 0.001,
		"decay_steps": decay_steps,
		"scale_by_worker": "linear",
	}


def current_auto_descriptor() -> dict[str, Any]:
	"""Return the old DPA-2 architecture with auto neighbor selection only."""
	return {
		"type": "dpa2",
		"repinit": {
			"rcut": 6.0,
			"rcut_smth": 0.5,
			"nsel": "auto:1.1",
			"neuron": [25, 50, 100],
			"axis_neuron": 16,
			"activation_function": "tanh",
		},
		"repformer": {
			"rcut": 4.0,
			"rcut_smth": 0.5,
			"nsel": "auto:1.1",
			"nlayers": 6,
			"g1_dim": 64,
			"g2_dim": 32,
			"attn2_hidden": 32,
			"attn2_nhead": 4,
			"attn1_hidden": 64,
			"attn1_nhead": 4,
			"axis_neuron": 4,
			"update_h2": False,
			"update_g1_has_conv": True,
			"update_g1_has_grrg": True,
			"update_g1_has_drrd": True,
			"update_g1_has_attn": True,
			"update_g2_has_g1g1": True,
			"update_g2_has_attn": True,
			"attn2_has_gate": True,
		},
		"seed": 1,
		"add_tebd_to_repinit_out": False,
	}


def doc_medium_descriptor(use_three_body: bool = True) -> dict[str, Any]:
	"""Return an official-medium-style DPA-2 descriptor adapted to this type map."""
	repinit: dict[str, Any] = {
		"tebd_dim": 8,
		"rcut": 6.0,
		"rcut_smth": 0.5,
		"nsel": "auto:1.1",
		"neuron": [25, 50, 100],
		"axis_neuron": 16,
		"activation_function": "tanh",
	}
	if use_three_body:
		repinit.update(
			{
				"three_body_sel": "auto:1.1",
				"three_body_rcut": 4.0,
				"three_body_rcut_smth": 3.5,
				"use_three_body": True,
			}
		)
	return {
		"type": "dpa2",
		"repinit": repinit,
		"repformer": {
			"rcut": 4.0,
			"rcut_smth": 3.5,
			"nsel": "auto:1.1",
			"nlayers": 6,
			"g1_dim": 128,
			"g2_dim": 32,
			"attn2_hidden": 32,
			"attn2_nhead": 4,
			"attn1_hidden": 128,
			"attn1_nhead": 4,
			"axis_neuron": 4,
			"update_h2": False,
			"update_g1_has_conv": True,
			"update_g1_has_grrg": True,
			"update_g1_has_drrd": True,
			"update_g1_has_attn": False,
			"update_g2_has_g1g1": False,
			"update_g2_has_attn": True,
			"update_style": "res_residual",
			"update_residual": 0.01,
			"update_residual_init": "norm",
			"attn2_has_gate": True,
			"use_sqrt_nnei": True,
			"g1_out_conv": True,
			"g1_out_mlp": True,
		},
		"precision": "float64",
		"seed": 1,
		"add_tebd_to_repinit_out": False,
	}


def fitting_net() -> dict[str, Any]:
	"""Return the baseline fitting net used by existing DPA-2 tests."""
	return {
		"neuron": [240, 240, 240],
		"resnet_dt": True,
		"numb_fparam": 1,
		"seed": 1,
		"type": "ener",
		"activation_function": "tanh",
		"precision": "float64",
	}


def make_config(source: dict[str, Any], descriptor: dict[str, Any], loss: dict[str, Any]) -> dict[str, Any]:
	"""Build a full DeePMD input JSON from descriptor and loss choices."""
	return {
		"_comment": "DPA-2 MgSiOH diagnostic suite, 2026-05-18. Short 200k-step run to isolate neighbor selection, official-medium recipe, three-body terms, and virial-loss effects.",
		"model": {
			"type_map": source["model"]["type_map"],
			"descriptor": descriptor,
			"fitting_net": fitting_net(),
			"data_stat_nbatch": 2,
			"data_stat_protect": 0.01,
		},
		"learning_rate": learning_rate(),
		"loss": loss,
		"training": base_training_block(source),
	}


def write_json(path: Path, data: dict[str, Any]) -> None:
	"""Write JSON with stable formatting."""
	path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def submission_script(shared_json: str, job_name: str, description: str) -> str:
	"""Return the Slurm submission script for one diagnostic variant."""
	return f"""#!/bin/bash
#SBATCH --account=bguf-delta-gpu
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --mem-per-cpu=32G
#SBATCH --time=24:00:00

set -e
module purge
module load PrgEnv-gnu
module load gcc-native/13.2
module load cray-mpich
module load cudatoolkit/25.3_12.8
module load fftw/3.3.10-gcc13.3.1
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate ALCHEMY_env__PT
export PYTHONNOUSERSITE=1
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HERE="${{SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}"
SHARED="$(cd "$HERE/../shared" && pwd)"
cp -f "$SHARED/{shared_json}" "$HERE/input.json"
cd "$HERE"

echo "=== [$(date)] PT DPA-2 diagnostic: {description} ==="
dp --pt train input.json 2>&1 | tee log.train
echo "=== [$(date)] Done ==="
"""


def main() -> None:
	"""Create shared inputs, run directories, and a suite manifest."""
	source: dict[str, Any] = load_source_config()
	variants: list[dict[str, str | dict[str, Any]]] = [
		{
			"name": "variant_train_dpa2_PT_current_auto_200k",
			"json": "train_dpa2_current_auto_200k.json",
			"job": "dpa2_cur_auto",
			"description": "old DPA-2 architecture with auto neighbor selection",
			"config": make_config(source, current_auto_descriptor(), current_loss()),
		},
		{
			"name": "variant_train_dpa2_PT_doc_medium_auto_currentloss_200k",
			"json": "train_dpa2_doc_medium_auto_currentloss_200k.json",
			"job": "dpa2_doc_cur",
			"description": "official-medium-style DPA-2 with current energy/force/virial loss",
			"config": make_config(source, doc_medium_descriptor(use_three_body=True), current_loss()),
		},
		{
			"name": "variant_train_dpa2_PT_doc_medium_auto_novirial_200k",
			"json": "train_dpa2_doc_medium_auto_novirial_200k.json",
			"job": "dpa2_doc_nov",
			"description": "official-medium-style DPA-2 with no virial loss",
			"config": make_config(source, doc_medium_descriptor(use_three_body=True), no_virial_loss()),
		},
		{
			"name": "variant_train_dpa2_PT_doc_medium_auto_no3body_200k",
			"json": "train_dpa2_doc_medium_auto_no3body_200k.json",
			"job": "dpa2_doc_no3",
			"description": "official-medium-style DPA-2 without the three-body repinit block",
			"config": make_config(source, doc_medium_descriptor(use_three_body=False), current_loss()),
		},
	]
	manifest_lines: list[str] = [
		"variant\tshared_json\tjob_name\tsteps\tpurpose",
	]
	for variant in variants:
		name: str = str(variant["name"])
		shared_json: str = str(variant["json"])
		job_name: str = str(variant["job"])
		description: str = str(variant["description"])
		config: dict[str, Any] = variant["config"]  # type: ignore[assignment]
		write_json(SHARED_DIR / shared_json, config)
		run_dir: Path = TRAINING_BENCH / name
		run_dir.mkdir(parents=True, exist_ok=True)
		(run_dir / "sub.sh").write_text(submission_script(shared_json, job_name, description), encoding="utf-8")
		write_json(run_dir / "input.json", config)
		manifest_lines.append(f"{name}\t{shared_json}\t{job_name}\t{NSTEPS}\t{description}")
	manifest_path: Path = TRAINING_BENCH / "DPA2_DIAGNOSTIC_SUITE_20260518.tsv"
	manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
	print(f"Wrote {len(variants)} DPA-2 diagnostic variants")
	print(manifest_path)


if __name__ == "__main__":
	main()

