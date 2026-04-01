#!/usr/bin/env python3

"""
##############################################################################
# plot_lammps_thermo.py
#
# Plot pressure, volume, temperature, and total energy vs time from a LAMMPS
# `log.lammps` file and save the result as a 4x1 figure.
#
# Usage:
#   python plot_lammps_thermo.py path/to/log.lammps
#   python plot_lammps_thermo.py path/to/log.lammps -o custom_plot.png
#
# Example:
#   python plot_lammps_thermo.py sim_data_ML_v4/v8_i2/md/ZONE_8/40H2_40NH3/log.lammps
#   python $HELP_SCRIPTS_ALCHEMY/plot_lammps_thermo.py log.lammps
#
# Notes:
#   - The script looks for thermo tables that contain Step, Temp, Press,
#     Volume, and TotEng.
#   - If the log contains multiple thermo blocks, they are stitched into one
#     continuous trajectory before plotting.
#   - Pressure and volume y-axes use the 0.1th and 99.9th percentiles with
#     sign-aware 20% padding, while energy uses sign-aware 1% padding.
#   - The pressure panel shows bar on the left axis and GPa on the right axis.
#   - The energy panel shows eV on the left axis and kJ/mol on the right axis.
#   - By default, the figure is written next to the log file as
#     `pressure_volume_energy_vs_time.png`.
##############################################################################
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable

# Keep matplotlib cache files in a writable location on the cluster.
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", f"matplotlib_{os.getenv('USER', 'user')}"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NUMERIC_LINE_RE = re.compile(r"^\s*timestep\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$")
BAR_TO_GPA = 1.0e-4
EV_TO_KJMOL = 96.485
DEFAULT_PADDING_FRACTION = 0.20
ENERGY_PADDING_FRACTION = 0.01
EXAMPLE_USAGE = """
Examples:
  python plot_lammps_thermo.py sim_data_ML_v4/v8_i2/md/ZONE_8/40H2_40NH3/log.lammps
  python plot_lammps_thermo.py sim_data_ML_v4/v8_i2/md/ZONE_8/40H2_40NH3/log.lammps -o zone8_thermo.png
"""


@dataclass
class ThermoBlock:
    """Store one thermo table extracted from the LAMMPS log.

    Attributes:
        rows: Parsed thermo rows from one Step... data block.
    """

    rows: list[dict[str, float]]


def bar_to_gpa(pressure_bar: float) -> float:
    """Convert pressure from bar to GPa.

    Args:
        pressure_bar: Pressure value in bar.

    Returns:
        float: Pressure value in GPa.
    """

    return pressure_bar * BAR_TO_GPA


def gpa_to_bar(pressure_gpa: float) -> float:
    """Convert pressure from GPa to bar.

    Args:
        pressure_gpa: Pressure value in GPa.

    Returns:
        float: Pressure value in bar.
    """

    return pressure_gpa / BAR_TO_GPA


def ev_to_kjmol(energy_ev: float) -> float:
    """Convert energy from eV to kJ/mol.

    Args:
        energy_ev: Energy value in eV.

    Returns:
        float: Energy value in kJ/mol.
    """

    return energy_ev * EV_TO_KJMOL


def kjmol_to_ev(energy_kjmol: float) -> float:
    """Convert energy from kJ/mol to eV.

    Args:
        energy_kjmol: Energy value in kJ/mol.

    Returns:
        float: Energy value in eV.
    """

    return energy_kjmol / EV_TO_KJMOL


def expand_lower_limit(value: float, padding_fraction: float) -> float:
    """Expand a lower-axis bound away from the data.

    Args:
        value: Lower percentile value.
        padding_fraction: Fractional padding applied to the percentile bound.

    Returns:
        float: Expanded lower-axis bound.
    """

    lower_scale = 1.0 - padding_fraction
    upper_scale = 1.0 + padding_fraction

    if value > 0.0:
        return lower_scale * value
    if value < 0.0:
        return upper_scale * value
    return value


def expand_upper_limit(value: float, padding_fraction: float) -> float:
    """Expand an upper-axis bound away from the data.

    Args:
        value: Upper percentile value.
        padding_fraction: Fractional padding applied to the percentile bound.

    Returns:
        float: Expanded upper-axis bound.
    """

    lower_scale = 1.0 - padding_fraction
    upper_scale = 1.0 + padding_fraction

    if value > 0.0:
        return upper_scale * value
    if value < 0.0:
        return lower_scale * value
    return value


def compute_axis_limits(values: list[float], padding_fraction: float) -> tuple[float, float]:
    """Compute y-axis limits from percentile-based bounds.

    Args:
        values: Data values to be plotted on one axis.
        padding_fraction: Fractional padding applied to the percentile bounds.

    Returns:
        tuple[float, float]: Lower and upper axis limits.
    """

    lower_percentile, upper_percentile = np.percentile(np.asarray(values, dtype=float), [0.1, 99.9])
    lower_limit = expand_lower_limit(float(lower_percentile), padding_fraction)
    upper_limit = expand_upper_limit(float(upper_percentile), padding_fraction)

    if lower_limit == upper_limit:
        padding = 1.0 if lower_limit == 0.0 else abs(lower_limit) * 0.1
        return lower_limit - padding, upper_limit + padding

    return lower_limit, upper_limit


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the plotting script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Plot pressure, volume, temperature, and total energy vs time from a LAMMPS log.lammps file.",
        epilog=EXAMPLE_USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("logfile", type=Path, help="Path to the LAMMPS log.lammps file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path. Default: <log_dir>/pressure_volume_energy_vs_time.png",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    """Read the full log file into memory.

    Args:
        path: Path to the input log file.

    Returns:
        list[str]: All lines from the file.
    """

    with path.open("r", encoding="utf-8") as handle:
        return handle.readlines()


def extract_timestep(lines: Iterable[str]) -> float:
    """Extract the last numeric timestep value defined in the log.

    Args:
        lines: Iterable of log-file lines.

    Returns:
        float: LAMMPS timestep in ps, consistent with the input script units.

    Raises:
        ValueError: If no numeric timestep entry is found.
    """

    timestep: float | None = None
    for line in lines:
        match = NUMERIC_LINE_RE.match(line)
        if match:
            timestep = float(match.group(1))
    if timestep is None:
        raise ValueError("Could not find a numeric 'timestep' entry in the log file.")
    return timestep


def is_numeric_row(tokens: list[str]) -> bool:
    """Check whether all tokens in a row can be parsed as floats.

    Args:
        tokens: Split fields from one candidate thermo row.

    Returns:
        bool: True if every token is numeric, otherwise False.
    """

    try:
        for token in tokens:
            float(token)
    except ValueError:
        return False
    return True


def parse_thermo_blocks(lines: list[str]) -> list[ThermoBlock]:
    """Parse thermo tables that contain Step, Temp, Press, Volume, and TotEng.

    Args:
        lines: Full LAMMPS log file as a list of lines.

    Returns:
        list[ThermoBlock]: Thermo data blocks found in the log.

    Raises:
        ValueError: If no matching thermo table is found.
    """

    blocks: list[ThermoBlock] = []
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()
        if not (
            stripped.startswith("Step")
            and "Temp" in stripped
            and "Press" in stripped
            and "Volume" in stripped
            and "TotEng" in stripped
        ):
            idx += 1
            continue

        header = stripped.split()
        rows: list[dict[str, float]] = []
        idx += 1

        while idx < len(lines):
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("Loop time of"):
                break

            tokens = stripped.split()
            if len(tokens) != len(header) or not is_numeric_row(tokens):
                break

            rows.append({name: float(value) for name, value in zip(header, tokens)})
            idx += 1

        if rows:
            blocks.append(ThermoBlock(rows=rows))

    if not blocks:
        raise ValueError("No thermo table with Step/Temp/Press/Volume/TotEng columns was found in the log file.")

    return blocks


def stitch_steps(blocks: list[ThermoBlock]) -> tuple[list[dict[str, float]], list[float]]:
    """Build one continuous step axis across multiple thermo blocks.

    Args:
        blocks: Thermo blocks parsed from the log.

    Returns:
        tuple[list[dict[str, float]], list[float]]:
            Combined thermo rows and the stitched-step locations where new
            thermo blocks begin.
    """

    combined: list[dict[str, float]] = []
    boundaries: list[float] = []
    offset = 0.0
    last_step: float | None = None

    for block_index, block in enumerate(blocks):
        first_step = block.rows[0]["Step"]
        # Shift later thermo blocks forward if a new run restarts the step count.
        if last_step is not None and first_step + offset < last_step:
            offset += last_step - (first_step + offset)

        if block_index > 0:
            boundaries.append(first_step + offset)

        for row in block.rows:
            stitched_row = dict(row)
            stitched_row["StitchedStep"] = row["Step"] + offset
            combined.append(stitched_row)

        last_step = combined[-1]["StitchedStep"]

    return combined, boundaries


def plot_thermo(data: list[dict[str, float]], boundaries: list[float], timestep_ps: float, output_path: Path) -> None:
    """Create and save the 4x1 thermo plot.

    Args:
        data: Combined thermo rows with stitched step values.
        boundaries: Step values where a new thermo block starts.
        timestep_ps: Timestep size in ps.
        output_path: Path for the output image.
    """

    time_ps = [row["StitchedStep"] * timestep_ps for row in data]
    pressure = [bar_to_gpa(row["Press"]) for row in data]  # convert to GPa for left axis
    volume = [row["Volume"] for row in data]
    temperature = [row["Temp"] for row in data]
    total_energy = [row["TotEng"] for row in data]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 11), constrained_layout=True)

    # Each entry: (values, ylabel, color, padding_fraction, primary_unit, secondary_convert_fn, secondary_unit)
    # secondary_convert_fn is None when there is no secondary y-axis.
    series = [
        (pressure,     "Pressure (GPa)",           "tab:red",    DEFAULT_PADDING_FRACTION, "GPa", gpa_to_bar,  "bar"),
        (volume,       r"Volume ($\AA^3$)",        "tab:blue",   DEFAULT_PADDING_FRACTION, r"Å³", None,        None),
        (temperature,  "Temperature (K)",          "tab:orange", DEFAULT_PADDING_FRACTION, "K",   None,        None),
        (total_energy, "Total Energy (eV)",        "tab:green",  ENERGY_PADDING_FRACTION,  "eV",  ev_to_kjmol, "kJ/mol"),
    ]

    for ax, (values, ylabel, color, padding_fraction, primary_unit, sec_fn, sec_unit) in zip(axes, series):
        # Compute mean and std over the last 25% of data points for converged statistics.
        n_last = max(1, len(values) // 4)
        tail_values = values[-n_last:]
        mean_val = float(np.mean(tail_values))
        std_val = float(np.std(tail_values))
        legend_label = f"$\\mu$ = {mean_val:.4g} {primary_unit},  $\\sigma$ = {std_val:.4g} {primary_unit}"
        ax.plot(time_ps, values, color=color, linewidth=1.0, label=legend_label)
        # Mark where the log transitions from one thermo block to the next.
        for boundary_step in boundaries:
            ax.axvline(boundary_step * timestep_ps, color="0.6", linestyle="--", linewidth=0.9)
        ax.set_ylim(*compute_axis_limits(values, padding_fraction))
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.3)

    pressure_axis_bar = axes[0].secondary_yaxis("right", functions=(gpa_to_bar, bar_to_gpa))
    pressure_axis_bar.set_ylabel("Pressure (bar)")
    energy_axis_kjmol = axes[3].secondary_yaxis("right", functions=(ev_to_kjmol, kjmol_to_ev))
    energy_axis_kjmol.set_ylabel("Total Energy (kJ/mol)")

    axes[0].set_title("LAMMPS Thermo Data vs Time  ($\\mu$, $\\sigma$ computed over last 25% of steps)")
    axes[-1].set_xlabel("Time (ps)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the CLI workflow: parse, extract, plot, and save."""

    args = parse_args()
    logfile = args.logfile.resolve()

    if not logfile.is_file():
        raise FileNotFoundError(f"Log file not found: {logfile}")

    output_path = args.output or logfile.parent / "pressure_volume_energy_vs_time.png"
    output_path = output_path.resolve()

    lines = read_lines(logfile)
    timestep_ps = extract_timestep(lines)
    blocks = parse_thermo_blocks(lines)
    data, boundaries = stitch_steps(blocks)
    plot_thermo(data, boundaries, timestep_ps, output_path)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
