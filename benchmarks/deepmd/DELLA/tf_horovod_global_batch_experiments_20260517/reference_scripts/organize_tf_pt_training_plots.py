from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


BASE = Path("/scratch/gpfs/BURROWS/akashgpt/qmd_data/NH3_H2/sim_data_ML_v3__plumed_test__v2/v7_i34/train__test/tf_hvd_apptainer300cuda126_bench_20260422/global_batch_experiments_20260517")
PLOTS_DIR = BASE / "training_loss_plots_10x_20260521"
PLOTS_MOD = Path("/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev/TRAIN_MLMD_scripts/ANALYSIS/mldp/util/plots_mod.py")


def iter_pt_run_dirs() -> Iterable[Path]:
	for group in ("pt_none_100k", "pt_step_scaled_none"):
		root = BASE / "runs" / group
		if not root.exists():
			continue
		for lcurve in sorted(root.glob("*/lcurve.out")):
			yield lcurve.parent


def symlink_force(target: Path, link: Path) -> None:
	if link.is_symlink() or link.exists():
		link.unlink()
	link.symlink_to(target)


def main() -> None:
	tf_dir = PLOTS_DIR / "TF"
	pt_dir = PLOTS_DIR / "PT"
	tf_dir.mkdir(exist_ok=True)
	pt_dir.mkdir(exist_ok=True)

	for png in sorted(PLOTS_DIR.glob("*.png")):
		target = png.resolve()
		symlink_force(target, tf_dir / png.name)

	env = os.environ.copy()
	env["MPLBACKEND"] = "Agg"
	for run_dir in iter_pt_run_dirs():
		plot = run_dir / "efv_plots.png"
		if not plot.exists():
			subprocess.run(["python", str(PLOTS_MOD)], cwd=run_dir, check=True, env=env)
		symlink_force(plot.resolve(), pt_dir / f"{run_dir.name}.png")

	print(f"TF links: {len(list(tf_dir.glob('*.png')))}")
	print(f"PT links: {len(list(pt_dir.glob('*.png')))}")
	print(tf_dir)
	print(pt_dir)


if __name__ == "__main__":
	main()
