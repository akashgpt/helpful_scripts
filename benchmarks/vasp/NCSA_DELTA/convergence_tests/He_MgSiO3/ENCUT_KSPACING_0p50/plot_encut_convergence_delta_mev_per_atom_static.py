"""Wrapper for the shared He_MgSiO3 ENCUT convergence plotter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SHARED_SCRIPT = Path(__file__).resolve().parents[1] / "plot_encut_convergence_delta_mev_per_atom_static.py"


def load_shared_module() -> object:
	"""Load the shared ENCUT convergence plotting module.

	Returns:
		Imported module object containing ``main``.
	"""
	spec = importlib.util.spec_from_file_location("he_mgsio3_encut_plotter", SHARED_SCRIPT)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load shared plotter from {SHARED_SCRIPT}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)
	return module


def main() -> None:
	"""Run the shared plotter for this ENCUT benchmark directory."""
	module = load_shared_module()
	module.main(["--benchmark-dir", str(Path(__file__).resolve().parent)])


if __name__ == "__main__":
	main()
