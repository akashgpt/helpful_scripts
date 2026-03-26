#!/usr/bin/env python3
"""Compute filtered RMSE metrics from dp_test per-frame output files.

Reads dp_test.e_peratom.out, dp_test.f.out, and dp_test.v_peratom.out
from all system directories found under the current working directory.
Excludes frames where |dE/atom| > energy_upper_cutoff from ALL metric
computations (energy, force, virial), consistent with the filtering
applied by analysis_v3.py during frame selection.

This replaces the old approach of reading aggregate RMSE from log.dp_test,
which included outlier frames and inflated the reported metrics.

Output format matches process_files_v2 in count.sh so that
track_performance.sh grep patterns continue to work.

Usage:
    python3 compute_filtered_rmse.py [--energy_upper_cutoff 10] [--force_upper_cutoff 100]

Authors: akashgpt, Claude
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np


def compute_system_rmse(
    system_dir: str,
    energy_upper_cutoff: float,
    force_upper_cutoff: float,
) -> Optional[tuple[int, float, float, float, int]]:
    """Compute filtered RMSE for a single system directory.

    Reads the three dp_test output files, identifies frames where
    |e_diff_per_atom| > energy_upper_cutoff, and computes RMSE over
    the remaining frames for energy, force, and virial.

    Variable names follow the conventions in analysis_v3.py:
      - e_diff_per_atom: per-frame absolute energy difference per atom
      - rmsd_e, rmsd_f, rmsd_v: root-mean-square deviations
      - nframes, natoms: system dimensions

    Args:
        system_dir: Path to the directory containing dp_test.*.out files.
        energy_upper_cutoff: Per-frame energy cutoff in eV/atom. Frames
            with |e_diff_per_atom| exceeding this are excluded from all
            RMSE computations (same role as args.energy_upper_cutoff in
            analysis_v3.py).
        force_upper_cutoff: Per-frame force cutoff in eV/A (reserved for
            future use; same role as args.force_upper_cutoff in
            analysis_v3.py).

    Returns:
        Tuple of (nframes_kept, rmsd_e_per_atom, rmsd_f, rmsd_v_per_atom,
        nframes_excluded), or None if files are missing or all frames are
        excluded.
    """
    efile = os.path.join(system_dir, "dp_test.e_peratom.out")
    ffile = os.path.join(system_dir, "dp_test.f.out")
    vfile = os.path.join(system_dir, "dp_test.v_peratom.out")

    if not all(os.path.isfile(f) for f in [efile, ffile, vfile]):
        return None

    # ── Energy per atom ──────────────────────────────────────────────
    # Columns: [data_e_per_atom, pred_e_per_atom]
    es_peratom = np.loadtxt(efile)
    if es_peratom.ndim == 1:
        es_peratom = es_peratom.reshape(1, -1)
    nframes = len(es_peratom)

    # Per-frame energy error per atom (matches e_diff_per_atom in analysis_v3.py)
    e_diff_per_atom = es_peratom[:, 0] - es_peratom[:, 1]

    # Apply energy_upper_cutoff filter (same as e_idx1 in analysis_v3.py)
    e_idx1 = np.where(np.abs(e_diff_per_atom) < energy_upper_cutoff)[0]
    nframes_kept = len(e_idx1)
    nframes_excluded = nframes - nframes_kept

    if nframes_kept == 0:
        return None

    # Energy RMSE/Natoms over kept frames
    rmsd_e_per_atom = float(np.sqrt(np.mean(e_diff_per_atom[e_idx1] ** 2)))

    # ── Forces ───────────────────────────────────────────────────────
    # Columns: [data_fx, data_fy, data_fz, pred_fx, pred_fy, pred_fz]
    # Lines: natoms * nframes
    fs_all = np.loadtxt(ffile)
    natoms = len(fs_all) // nframes
    fs_all = fs_all.reshape(nframes, natoms, 6)

    fs_kept = fs_all[e_idx1]
    fs_diff = fs_kept[:, :, 3:6] - fs_kept[:, :, 0:3]  # pred - data
    rmsd_f = float(np.sqrt(np.mean(fs_diff ** 2)))

    # ── Virial per atom ──────────────────────────────────────────────
    # Columns: [9 data virial components, 9 pred virial components]
    vs_peratom = np.loadtxt(vfile)
    if vs_peratom.ndim == 1:
        vs_peratom = vs_peratom.reshape(1, -1)
    vs_kept = vs_peratom[e_idx1]
    vs_diff = vs_kept[:, 9:18] - vs_kept[:, 0:9]
    rmsd_v_per_atom = float(np.sqrt(np.mean(vs_diff ** 2)))

    return nframes_kept, rmsd_e_per_atom, rmsd_f, rmsd_v_per_atom, nframes_excluded


def main() -> None:
    """Find all dp_test outputs, compute filtered RMSE, print summary."""
    parser = argparse.ArgumentParser(
        description="Compute filtered RMSE from dp_test per-frame outputs"
    )
    parser.add_argument(
        "--energy_upper_cutoff",
        "-euc",
        type=float,
        default=10.0,
        help=(
            "Per-frame energy cutoff in eV/atom (default: 10). "
            "Maps to RECAL_CUTOFF_e_high in TRAIN_MLMD_parameters.txt."
        ),
    )
    parser.add_argument(
        "--force_upper_cutoff",
        "-fuc",
        type=float,
        default=100.0,
        help=(
            "Per-frame force cutoff in eV/A (default: 100). "
            "Maps to RECAL_CUTOFF_f_high in TRAIN_MLMD_parameters.txt."
        ),
    )
    args = parser.parse_args()

    # Find all system directories containing dp_test.e_peratom.out
    system_dirs: list[str] = []
    for root, _dirs, files in os.walk("."):
        if "dp_test.e_peratom.out" in files:
            system_dirs.append(root)

    if not system_dirs:
        _print_empty_results()
        return

    # Compute per-system filtered metrics
    all_nframes_kept: list[float] = []
    all_rmsd_e: list[float] = []
    all_rmsd_f: list[float] = []
    all_rmsd_v: list[float] = []
    total_nframes_excluded = 0
    skipped_systems: list[str] = []

    for sdir in sorted(system_dirs):
        try:
            result = compute_system_rmse(
                sdir, args.energy_upper_cutoff, args.force_upper_cutoff
            )
            if result is None:
                skipped_systems.append(sdir)
                continue
            nkept, rmsd_e, rmsd_f, rmsd_v, nexcl = result
            all_nframes_kept.append(float(nkept))
            all_rmsd_e.append(rmsd_e)
            all_rmsd_f.append(rmsd_f)
            all_rmsd_v.append(rmsd_v)
            total_nframes_excluded += nexcl
        except Exception as exc:
            print(f"WARNING: Skipping {sdir} ({exc})", file=sys.stderr)
            skipped_systems.append(sdir)

    if not all_nframes_kept:
        _print_empty_results()
        return

    # ── Frame-weighted aggregation (matches process_files_v2 convention) ──
    nframes_arr = np.array(all_nframes_kept)
    rmsd_e_arr = np.array(all_rmsd_e)
    rmsd_f_arr = np.array(all_rmsd_f)
    rmsd_v_arr = np.array(all_rmsd_v)

    total_nframes_kept = np.sum(nframes_arr)
    weights = nframes_arr / total_nframes_kept

    # Weighted averages
    avg_rmsd_e = float(np.sum(weights * rmsd_e_arr))
    avg_rmsd_f = float(np.sum(weights * rmsd_f_arr))
    avg_rmsd_v = float(np.sum(weights * rmsd_v_arr))

    # Weighted standard deviations
    std_rmsd_e = float(np.sqrt(np.sum(weights * (rmsd_e_arr - avg_rmsd_e) ** 2)))
    std_rmsd_f = float(np.sqrt(np.sum(weights * (rmsd_f_arr - avg_rmsd_f) ** 2)))
    std_rmsd_v = float(np.sqrt(np.sum(weights * (rmsd_v_arr - avg_rmsd_v) ** 2)))

    # Convert energy to meV
    avg_rmsd_e_mev = avg_rmsd_e * 1000
    std_rmsd_e_mev = std_rmsd_e * 1000

    # ── Output (same format as process_files_v2 for backward compatibility) ──
    total_nframes = int(total_nframes_kept) + total_nframes_excluded
    print("")
    print('### "log.dp_test" ###')
    if total_nframes_excluded > 0:
        print(
            f"Total number of test data frames: {total_nframes}"
            f" ({int(total_nframes_kept)} kept, {total_nframes_excluded}"
            f" excluded by energy_upper_cutoff={args.energy_upper_cutoff})"
        )
    else:
        print(f"Total number of test data frames: {total_nframes}")
    print(
        f"Average Energy RMSE/Natoms:"
        f" {avg_rmsd_e_mev:.4f} +/- {std_rmsd_e_mev:.4f} meV"
    )
    print(
        f"Average Force RMSE:"
        f" {avg_rmsd_f:.4f} +/- {std_rmsd_f:.4f} eV/A"
    )
    print(
        f"Average Virial RMSE/Natoms:"
        f" {avg_rmsd_v:.4f} +/- {std_rmsd_v:.4f} eV"
    )
    print("")

    if skipped_systems:
        print(
            f"NOTE: {len(skipped_systems)} system(s) skipped"
            " (missing files or all frames excluded):",
            file=sys.stderr,
        )
        for s in skipped_systems:
            print(f"  {s}", file=sys.stderr)


def _print_empty_results() -> None:
    """Print zero-valued results when no data is found."""
    print("")
    print('### "log.dp_test" ###')
    print("Total number of test data frames: 0")
    print("Average Energy RMSE/Natoms: 0.0000 +/- 0.0000 meV")
    print("Average Force RMSE: 0.0000 +/- 0.0000 eV/A")
    print("Average Virial RMSE/Natoms: 0.0000 +/- 0.0000 eV")
    print("")


if __name__ == "__main__":
    main()
