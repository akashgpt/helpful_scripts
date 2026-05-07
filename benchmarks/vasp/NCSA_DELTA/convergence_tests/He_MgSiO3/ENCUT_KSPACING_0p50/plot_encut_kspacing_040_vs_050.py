"""Plot ENCUT convergence comparison for KSPACING 0.40 and 0.50."""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


BENCHMARK_DIR: Path = Path(__file__).resolve().parent
HE_MGSIO3_DIR: Path = BENCHMARK_DIR.parent
CSV_040: Path = HE_MGSIO3_DIR / "ENCUT" / "encut_convergence_delta_mev_per_atom_static.csv"
CSV_050: Path = BENCHMARK_DIR / "encut_convergence_delta_mev_per_atom_static.csv"
OUTPUT_PNG: Path = BENCHMARK_DIR / "encut_convergence_kspacing_040_vs_050_static.png"


@dataclass(frozen=True)
class EncutRow:
	"""One ENCUT convergence point for a fixed KSPACING."""

	kspacing: str
	family: str
	config: str
	encut: int
	delta_toten_mev_per_atom: float
	delta_internal_mev_per_atom: float

	@property
	def case_label(self) -> str:
		"""Return a compact case label."""
		return f"{self.family}\n{self.config}"


def load_rows(csv_path: Path, kspacing_label: str) -> list[EncutRow]:
	"""Load rows from one static ENCUT convergence CSV.

	Args:
		csv_path: Input CSV path.
		kspacing_label: Label for this fixed-KSPACING sweep.

	Returns:
		Parsed convergence rows.
	"""
	rows: list[EncutRow] = []
	with csv_path.open("r", encoding="utf-8", newline="") as handle:
		for raw_row in csv.DictReader(handle):
			rows.append(
				EncutRow(
					kspacing=kspacing_label,
					family=raw_row["family"],
					config=raw_row["config"],
					encut=int(float(raw_row["encut"])),
					delta_toten_mev_per_atom=float(
						raw_row.get("delta_toten_meV_per_atom", raw_row.get("delta_toten_mev_per_atom", "nan"))
					),
					delta_internal_mev_per_atom=float(
						raw_row.get(
							"delta_internal_meV_per_atom",
							raw_row.get("delta_internal_mev_per_atom", "nan"),
						)
					),
				)
			)
	return rows


def group_by_case(rows: Iterable[EncutRow]) -> OrderedDict[tuple[str, str], list[EncutRow]]:
	"""Group rows by system family and composition."""
	grouped: OrderedDict[tuple[str, str], list[EncutRow]] = OrderedDict()
	for row in rows:
		grouped.setdefault((row.family, row.config), []).append(row)
	return grouped


def configure_axis(axis: plt.Axes, title: str, ylabel: str) -> None:
	"""Apply common energy-axis styling."""
	if title:
		axis.set_title(title, fontsize=12)
	axis.set_xlabel("ENCUT (eV)", fontsize=10)
	axis.set_ylabel(ylabel, fontsize=10)
	axis.set_yscale("symlog", linthresh=1.0e-1)
	axis.grid(True, which="both", alpha=0.25)
	axis.tick_params(axis="both", labelsize=9)


def plot_comparison(rows: list[EncutRow], output_png: Path) -> None:
	"""Create the KSPACING 0.40 vs 0.50 ENCUT comparison plot."""
	grouped = group_by_case(rows)
	fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
	colors: dict[str, str] = {"0.40": "#1f77b4", "0.50": "#d62728"}
	markers: dict[str, str] = {"0.40": "o", "0.50": "s"}
	for column_index, ((_family, _config), case_rows) in enumerate(grouped.items()):
		if column_index >= 4:
			break
		case_label = case_rows[0].case_label
		for kspacing_label in ["0.40", "0.50"]:
			selected = sorted(
				[row for row in case_rows if row.kspacing == kspacing_label],
				key=lambda row: row.encut,
			)
			if not selected:
				continue
			encuts = [row.encut for row in selected]
			axes[0, column_index].plot(
				encuts,
				[row.delta_toten_mev_per_atom for row in selected],
				color=colors[kspacing_label],
				marker=markers[kspacing_label],
				linewidth=1.8,
				label=f"KSPACING={kspacing_label}",
			)
			axes[1, column_index].plot(
				encuts,
				[row.delta_internal_mev_per_atom for row in selected],
				color=colors[kspacing_label],
				marker=markers[kspacing_label],
				linewidth=1.8,
				label=f"KSPACING={kspacing_label}",
			)
		configure_axis(axes[0, column_index], case_label, r"$|\Delta \mathrm{TOTEN}|$ (meV/atom)")
		configure_axis(axes[1, column_index], "", r"$|\Delta$ internal energy| (meV/atom)")
		axes[0, column_index].legend(fontsize=8, loc="upper right")
	for axis in axes.flat:
		axis.set_xticks([400, 500, 600, 800, 1000, 1200])
	fig.suptitle("He_MgSiO3 static ENCUT convergence: KSPACING 0.40 vs 0.50", fontsize=16)
	fig.tight_layout(rect=(0, 0, 1, 0.94))
	fig.savefig(output_png, dpi=220)
	plt.close(fig)


def main() -> None:
	"""Load CSVs and write the comparison plot."""
	rows = load_rows(CSV_040, "0.40") + load_rows(CSV_050, "0.50")
	plot_comparison(rows, OUTPUT_PNG)
	print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
	main()
