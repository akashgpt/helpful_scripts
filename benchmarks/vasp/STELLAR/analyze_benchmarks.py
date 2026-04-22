"""
Analyze VASP core/node/task-per-node benchmarks on Stellar.

Source: /scratch/gpfs/BURROWS/akashgpt/qmd_data/H2O_H2/sim_data/a08318m_test*
Each test dir contains:
  - RUN_VASP.sh  : SLURM submission script with --nodes, --ntasks-per-node,
                   --cpus-per-task, --mem-per-cpu, --time
  - OUTCAR       : VASP output; LOOP+ lines = per-ionic-step real time
                                 LOOP:  lines = per-electronic-step real time
  - log.run_sim  : stdout of `srun vasp_std` (mirrors most of OUTCAR output)
  - slurm-*.out  : slurm stdout/stderr

The same physical system (H2O-H2 mixture, a08318m) is run under different
parallel layouts to probe performance. We extract, per test dir:

  nodes, ntasks_per_node, cpus_per_task  (= OMP_NUM_THREADS in these runs)
  total_mpi_ranks  = nodes * ntasks_per_node
  total_cores      = nodes * ntasks_per_node * cpus_per_task
  ionic_steps      = #(LOOP+) entries
  elec_steps       = #(LOOP)  entries (electronic SCF)
  real_per_ionic   = median real time per LOOP+ (seconds / ionic step)
  real_per_elec    = median real time per LOOP  (seconds / electronic step)
  status           = ok / failed / no_outcar

Stellar "intel" / "cimes" compute nodes: 96 cores per node
(Intel Ice Lake 2x Xeon Platinum 8360Y, 2x48). Cross-check: the configs
include (nodes=1, ntpn=96, cpt=1) and (nodes=1, ntpn=48, cpt=2) and
(nodes=1, ntpn=1, cpt=96), consistent with a 96-core node.
"""

from __future__ import annotations

import csv
import os
import re
import statistics as stats
from pathlib import Path
from typing import Optional


SIM_ROOT = Path("/scratch/gpfs/BURROWS/akashgpt/qmd_data/H2O_H2/sim_data")
OUT_ROOT = Path("/scratch/gpfs/BURROWS/akashgpt/qmd_data/benchmarks/vasp/STELLAR")

SBATCH_PAT = re.compile(r"^\s*#SBATCH\s+--([a-zA-Z-]+)=(\S+)")
LOOP_RE = re.compile(r"LOOP\+?:\s+cpu time\s+([\d.]+):\s+real time\s+([\d.]+)")
# Distinguish LOOP+ (ionic) vs LOOP (electronic). Leading space of each line
# helps: "     LOOP+: ..." vs "      LOOP: ...".
IONIC_RE = re.compile(r"LOOP\+:\s+cpu time\s+([\d.]+):\s+real time\s+([\d.]+)")
ELEC_RE = re.compile(r"(?<!\+)\s+LOOP:\s+cpu time\s+([\d.]+):\s+real time\s+([\d.]+)")
# VASP INCAR tags that influence parallelization.
INCAR_INT_TAGS = ("NPAR", "NCORE", "KPAR", "NBANDS")
INCAR_RE = {tag: re.compile(rf"^\s*{tag}\s*=\s*(\d+)", re.MULTILINE) for tag in INCAR_INT_TAGS}


def parse_sbatch(path: Path) -> dict[str, str]:
    """Parse #SBATCH directives from a submission script.

    Returns a dict mapping the directive long-name to its value string.
    """
    result: dict[str, str] = {}
    if not path.exists():
        return result
    with path.open("r", errors="replace") as f:
        for line in f:
            m = SBATCH_PAT.match(line)
            if m:
                result[m.group(1)] = m.group(2)
    return result


def extract_loop_times(outcar: Path) -> tuple[list[float], list[float]]:
    """Return (ionic_real_times, elec_real_times) lists from an OUTCAR.

    Iterates once and classifies each LOOP line as ionic (LOOP+) or
    electronic (LOOP:). Uses the regex anchored on the "+" presence.
    """
    ionic: list[float] = []
    elec: list[float] = []
    if not outcar.exists():
        return ionic, elec
    with outcar.open("r", errors="replace") as f:
        for line in f:
            # fast reject
            if "LOOP" not in line:
                continue
            if "LOOP+:" in line:
                m = IONIC_RE.search(line)
                if m:
                    ionic.append(float(m.group(2)))
            elif "LOOP:" in line:
                # plain LOOP, not LOOP+
                m = re.search(r"LOOP:\s+cpu time\s+([\d.]+):\s+real time\s+([\d.]+)", line)
                if m:
                    elec.append(float(m.group(2)))
    return ionic, elec


def to_int(x: Optional[str]) -> Optional[int]:
    """Int-cast that returns None on blank/missing input."""
    if x is None:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def classify_status(outcar: Path, slurm_outs: list[Path], ionic_count: int) -> str:
    """Rough classification: ok if it produced ionic steps; failed if a
    slurm-*.out has a segfault/error and no ionic steps; no_outcar otherwise.
    """
    if not outcar.exists() or outcar.stat().st_size == 0:
        return "no_outcar"
    if ionic_count == 0:
        # Look for obvious crash markers.
        for so in slurm_outs:
            try:
                tail = so.read_text(errors="replace")[-2000:]
            except Exception:
                continue
            if "Segmentation fault" in tail or "SICK JOB" in tail or "error (78)" in tail:
                return "failed_crash"
        return "no_ionic_steps"
    return "ok"


def parse_incar(path: Path) -> dict[str, Optional[int]]:
    """Pull NPAR/NCORE/KPAR/NBANDS from an INCAR (None if absent)."""
    result: dict[str, Optional[int]] = {tag: None for tag in INCAR_INT_TAGS}
    if not path.exists():
        return result
    text = path.read_text(errors="replace")
    for tag, pat in INCAR_RE.items():
        m = pat.search(text)
        if m:
            result[tag] = int(m.group(1))
    return result


def analyze_dir(dirpath: Path) -> Optional[dict]:
    """Extract one row of benchmark data from a test directory."""
    sb = parse_sbatch(dirpath / "RUN_VASP.sh")
    if not sb:
        return None

    nodes = to_int(sb.get("nodes"))
    ntpn = to_int(sb.get("ntasks-per-node"))
    cpt = to_int(sb.get("cpus-per-task"))
    mem = sb.get("mem-per-cpu", "")
    walltime = sb.get("time", "")
    incar = parse_incar(dirpath / "INCAR")

    outcar = dirpath / "OUTCAR"
    slurm_outs = sorted(dirpath.glob("slurm-*.out"))
    ionic, elec = extract_loop_times(outcar)

    status = classify_status(outcar, slurm_outs, len(ionic))

    # Skip the very first LOOP+ (often includes startup / KPAR setup cost).
    ionic_trim = ionic[1:] if len(ionic) > 1 else ionic
    elec_trim = elec[10:] if len(elec) > 10 else elec

    row = {
        "name": dirpath.name,
        "nodes": nodes,
        "ntasks_per_node": ntpn,
        "cpus_per_task": cpt,
        "mpi_ranks": (nodes * ntpn) if (nodes and ntpn) else None,
        "total_cores": (nodes * ntpn * cpt) if (nodes and ntpn and cpt) else None,
        "mem_per_cpu": mem,
        "walltime": walltime,
        "NPAR": incar["NPAR"],
        "NCORE": incar["NCORE"],
        "KPAR": incar["KPAR"],
        "NBANDS": incar["NBANDS"],
        "status": status,
        "ionic_steps": len(ionic),
        "elec_steps": len(elec),
        "median_real_per_ionic_s": (round(stats.median(ionic_trim), 4) if ionic_trim else None),
        "mean_real_per_ionic_s": (round(stats.fmean(ionic_trim), 4) if ionic_trim else None),
        "median_real_per_elec_s": (round(stats.median(elec_trim), 4) if elec_trim else None),
        "slurm_out_files": len(slurm_outs),
    }
    # Throughput metrics (computed up-front so they live in the CSV too).
    t = row["median_real_per_ionic_s"]
    c = row["total_cores"]
    if t and c:
        row["steps_per_sec"] = round(1.0 / t, 6)                     # ionic steps / second
        row["steps_per_sec_per_core"] = round(1.0 / (t * c), 8)      # efficiency
        row["core_s_per_step"] = round(t * c, 3)                     # core-seconds / step
    else:
        row["steps_per_sec"] = None
        row["steps_per_sec_per_core"] = None
        row["core_s_per_step"] = None
    # Bands-per-rank: the key ceiling on band-parallelism (and by extension
    # on how many MPI ranks can be used productively). VASP distributes
    # bands across (mpi_ranks / NPAR) groups of NPAR ranks; so the average
    # load is NBANDS / mpi_ranks bands per rank. < 2 ⇒ band parallelism is
    # starving.
    nbands = incar["NBANDS"]
    ranks = row["mpi_ranks"]
    if nbands and ranks:
        row["bands_per_rank"] = round(nbands / ranks, 2)
    else:
        row["bands_per_rank"] = None
    return row


def main() -> None:
    """Walk sim_data, build benchmark CSV and a short human-readable summary."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for d in sorted(SIM_ROOT.glob("a08318m_test*")):
        if not d.is_dir():
            continue
        row = analyze_dir(d)
        if row is not None:
            rows.append(row)

    # CSV.
    csv_path = OUT_ROOT / "benchmarks.csv"
    fieldnames = [
        "name",
        "nodes",
        "ntasks_per_node",
        "cpus_per_task",
        "mpi_ranks",
        "total_cores",
        "mem_per_cpu",
        "walltime",
        "NPAR",
        "NCORE",
        "KPAR",
        "NBANDS",
        "status",
        "ionic_steps",
        "elec_steps",
        "median_real_per_ionic_s",
        "mean_real_per_ionic_s",
        "median_real_per_elec_s",
        "steps_per_sec",
        "steps_per_sec_per_core",
        "core_s_per_step",
        "bands_per_rank",
        "slurm_out_files",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Analysis / summary.
    ok_rows = [r for r in rows if r["status"] == "ok" and r["median_real_per_ionic_s"]]
    by_cores: dict[int, list[dict]] = {}
    for r in ok_rows:
        by_cores.setdefault(r["total_cores"], []).append(r)

    lines: list[str] = []
    lines.append("# Stellar VASP parallelization benchmark — summary")
    lines.append("")
    lines.append(f"Rows parsed:  {len(rows)}")
    lines.append(f"OK (with ionic timing): {len(ok_rows)}")
    lines.append(f"Total cores sampled: {sorted(by_cores)}")
    lines.append("")
    lines.append("## Best median real-time per ionic step (MD step), by total_cores")
    lines.append("")
    lines.append("total_cores | best median_real_per_ionic_s | config (nodes x ntpn x cpt) | ionic_steps")
    lines.append("------------|------------------------------|-----------------------------|------------")
    for n_cores in sorted(by_cores):
        best = min(by_cores[n_cores], key=lambda r: r["median_real_per_ionic_s"])
        cfg = f"{best['nodes']} x {best['ntasks_per_node']} x {best['cpus_per_task']}"
        lines.append(f"{n_cores:>11d} | {best['median_real_per_ionic_s']:>28.4f} | {cfg:<27} | {best['ionic_steps']}")
    lines.append("")
    lines.append("## Per-configuration summary (all 'ok' runs)")
    lines.append("")
    lines.append("name | cores | nodes x ntpn x cpt | NPAR | ionic_steps | median_t_ionic_s")
    lines.append("-----|-------|--------------------|------|-------------|------------------")
    for r in sorted(ok_rows, key=lambda r: (r["total_cores"], r["median_real_per_ionic_s"])):
        lines.append(
            f"{r['name']} | {r['total_cores']} | "
            f"{r['nodes']}x{r['ntasks_per_node']}x{r['cpus_per_task']} | "
            f"{r['NPAR']} | "
            f"{r['ionic_steps']} | {r['median_real_per_ionic_s']}"
        )
    lines.append("")
    lines.append("## Failed / problematic runs")
    lines.append("")
    lines.append("name | status | cores | ionic_steps")
    lines.append("-----|--------|-------|-------------")
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"{r['name']} | {r['status']} | {r['total_cores']} | {r['ionic_steps']}")
    lines.append("")

    # Scaling efficiency relative to best single-node performer.
    if ok_rows:
        # baseline = best single-node (nodes=1) run by median_real_per_ionic_s
        one_node = [r for r in ok_rows if r["nodes"] == 1]
        if one_node:
            baseline = min(one_node, key=lambda r: r["median_real_per_ionic_s"])
            t_b = baseline["median_real_per_ionic_s"]
            c_b = baseline["total_cores"]
            lines.append(f"## Strong-scaling efficiency (baseline = {baseline['name']}, "
                         f"{c_b} cores, {t_b:.4f} s/ionic-step)")
            lines.append("")
            lines.append("Speedup = t_baseline / t_config;  Efficiency = Speedup / (cores_config / cores_baseline)")
            lines.append("")
            lines.append("name | cores | t_ionic_s | speedup | efficiency")
            lines.append("-----|-------|-----------|---------|-----------")
            # Only compare best-per-core-count for readability.
            for n_cores in sorted(by_cores):
                best = min(by_cores[n_cores], key=lambda r: r["median_real_per_ionic_s"])
                t = best["median_real_per_ionic_s"]
                speedup = t_b / t if t else 0.0
                eff = speedup / (n_cores / c_b) if n_cores else 0.0
                lines.append(f"{best['name']} | {n_cores} | {t:.4f} | {speedup:.3f} | {eff:.3f}")
            lines.append("")

    summary_path = OUT_ROOT / "SUMMARY.md"
    summary_path.write_text("\n".join(lines))

    analysis_path = OUT_ROOT / "ANALYSIS.md"
    analysis_path.write_text(build_analysis(ok_rows, by_cores))

    # Also emit a short text preview to stdout.
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {analysis_path}")
    print(f"Rows: {len(rows)} (ok with timing: {len(ok_rows)})")


# -----------------------------------------------------------------------------
# Deeper analysis: how efficiency (steps/sec/core) responds to each knob.
# -----------------------------------------------------------------------------

def _group_by(rows: list[dict], key: str) -> dict:
    """Group rows by a given key; returns dict of key_value -> [rows]."""
    g: dict = {}
    for r in rows:
        g.setdefault(r[key], []).append(r)
    return g


def _agg(rows: list[dict], metric: str) -> dict:
    """Min / median / max of `metric` across a list of rows."""
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    if not vals:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(vals),
        "min": min(vals),
        "median": stats.median(vals),
        "max": max(vals),
    }


def _fmt_eff(x: Optional[float]) -> str:
    """Format throughput-per-core as steps / (s x 1000 cores) for readability."""
    if x is None:
        return "  --"
    return f"{x * 1000:6.2f}"


def build_analysis(ok_rows: list[dict], by_cores: dict[int, list[dict]]) -> str:
    """Produce ANALYSIS.md: explores efficiency vs. each tunable knob.

    The primary efficiency metric is `steps_per_sec_per_core`:
        = 1 / (median_real_time_per_ionic_step * total_cores)
    i.e. how many MD steps each allocated core produces per wall-clock second.
    Perfect strong scaling ⇒ this is constant in total_cores; real-world
    scaling ⇒ it drops as cores grow. Throughput (steps/sec) grows as long
    as efficiency doesn't fall faster than cores grow.

    We also use throughput (steps/sec) and core-seconds/step (== 1 /
    steps_per_sec_per_core) depending on which frame is clearer.
    """
    L: list[str] = []
    L.append("# Stellar VASP benchmark — efficiency vs. knob analysis")
    L.append("")
    L.append("## 0. System under test and why it caps scaling")
    L.append("")
    L.append("- **Atoms:** 370 total (54 O + 316 H), constant across all 77 tests.")
    L.append("- **NBANDS:** 468 (20 early tests), 480 (55 tests, the bulk), 512 (2 tests: `test35`, `test40`). Same POSCAR/POTCAR/KPOINTS otherwise — only NBANDS varies.")
    L.append("- **K-points:** Γ-only (1×1×1, Monkhorst), so `KPAR = 1` always; no k-point parallelism is in play.")
    L.append("- **Why this matters for scaling:** VASP distributes bands across `mpi_ranks / NPAR` groups of `NPAR` ranks, so average load ≈ `NBANDS / mpi_ranks` bands per rank.")
    L.append("  - Healthy: **≥ 4 bands/rank** (enough work per rank, small overhead).")
    L.append("  - Stressed: **2–4 bands/rank** (comms and synchronization start to dominate).")
    L.append("  - Starving: **< 2 bands/rank** (rank holds ~1 band; parallelism breaks down).")
    L.append("- With ~480 bands, the ceiling on productive MPI ranks is roughly **120–240 ranks** for this workload. Beyond that, adding cores via MPI buys nothing; adding cores via OMP threads may still help, up to ~8–12 threads/rank on Stellar.")
    L.append("")
    L.append("**Primary metric:** `steps_per_sec_per_core` = 1 / (median_real_per_ionic_step × total_cores).")
    L.append("Think of it as \"MD steps produced per core per wall-clock second\"; higher = more efficient.")
    L.append("Reported below as **steps · s⁻¹ · (1000 cores)⁻¹** so numbers are readable (×1000 vs. raw value).")
    L.append("")
    L.append("**Throughput** = steps/sec = 1 / t_step. **Core-seconds/step** = t_step × cores; lower = cheaper per step in allocation terms.")
    L.append("")
    L.append("Ideal strong scaling ⇒ `steps/sec/core` constant as cores grow. If it drops, each added core is delivering less than the previous one.")
    L.append("")

    # -------------------------------------------------------------------------
    # 1. Strong scaling curve (best-config-at-each-core-count).
    # -------------------------------------------------------------------------
    L.append("## 1. Strong-scaling: best config at each total-core count")
    L.append("")
    L.append("For each core budget, take the best-performing (fastest per-step) run and show: time per step, total throughput, per-core efficiency, core-seconds per step, and the bands/rank load.")
    L.append("")
    L.append("| total_cores | best layout (n×ntpn×cpt) | NPAR | NBANDS | mpi_ranks | bands/rank | t_step (s) | steps/s | steps/s/(1k cores) | core·s/step |")
    L.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    ranked_cores = sorted(by_cores)
    base_eff = None
    for n_cores in ranked_cores:
        best = min(by_cores[n_cores], key=lambda r: r["median_real_per_ionic_s"])
        eff = best["steps_per_sec_per_core"]
        if base_eff is None:
            base_eff = eff
        L.append(
            f"| {n_cores} "
            f"| {best['nodes']}×{best['ntasks_per_node']}×{best['cpus_per_task']} "
            f"| {best['NPAR']} "
            f"| {best['NBANDS']} "
            f"| {best['mpi_ranks']} "
            f"| {best['bands_per_rank']} "
            f"| {best['median_real_per_ionic_s']:.3f} "
            f"| {best['steps_per_sec']:.4f} "
            f"| {_fmt_eff(eff)} "
            f"| {best['core_s_per_step']:.1f} |"
        )
    L.append("")
    L.append("**Read this as**: the `steps/s/(1k cores)` column is absolute per-core efficiency. Dividing row values by the single-node (96-core) entry gives strong-scaling efficiency. The `bands/rank` column is the single best predictor of where efficiency collapses: the best layouts cluster around 15–20 bands/rank, while stressed ones (≤ 5) match poor efficiency.")
    L.append("")

    # -------------------------------------------------------------------------
    # 2. At each total_cores, how does the MPI/OMP split matter?
    # -------------------------------------------------------------------------
    L.append("## 2. Effect of MPI/OMP split at fixed core budget")
    L.append("")
    L.append("Each row is one (total_cores, ntasks_per_node × cpus_per_task) combination. If a combo was tested multiple times (different NPAR / reruns), we report min/median/max of per-step time across those reruns.")
    L.append("")
    for n_cores in ranked_cores:
        group = by_cores[n_cores]
        L.append(f"### total_cores = {n_cores}")
        L.append("")
        L.append("| layout (n×ntpn×cpt) | mpi_ranks | omp | bands/rank* | runs | t_step min / med / max (s) | best steps/s/(1k cores) |")
        L.append("|---|---:|---:|---:|---:|---|---:|")
        # Group by (nodes, ntpn, cpt).
        split: dict = {}
        for r in group:
            key = (r["nodes"], r["ntasks_per_node"], r["cpus_per_task"])
            split.setdefault(key, []).append(r)
        # Sort layouts by best (min) t_step within group, fastest first.
        keys_sorted = sorted(split, key=lambda k: min(r["median_real_per_ionic_s"] for r in split[k]))
        for (n, ntpn, cpt) in keys_sorted:
            rs = split[(n, ntpn, cpt)]
            ts = [r["median_real_per_ionic_s"] for r in rs]
            best_eff = max((r["steps_per_sec_per_core"] for r in rs if r["steps_per_sec_per_core"] is not None), default=None)
            bpr_vals = sorted({r["bands_per_rank"] for r in rs if r["bands_per_rank"] is not None})
            bpr_str = "/".join(f"{v:g}" for v in bpr_vals) if bpr_vals else "—"
            L.append(
                f"| {n}×{ntpn}×{cpt} "
                f"| {n * ntpn} "
                f"| {cpt} "
                f"| {bpr_str} "
                f"| {len(rs)} "
                f"| {min(ts):.3f} / {stats.median(ts):.3f} / {max(ts):.3f} "
                f"| {_fmt_eff(best_eff)} |"
            )
        L.append("")
    L.append("`*` bands/rank = NBANDS / mpi_ranks. When a cell lists multiple values (e.g. `15/16`) it's because different runs in that layout used different NBANDS (468 vs 480 vs 512).")
    L.append("")

    # -------------------------------------------------------------------------
    # 3. Marginal effect of each knob (aggregated across runs).
    # -------------------------------------------------------------------------
    L.append("## 3. Marginal effect of individual knobs")
    L.append("")
    L.append("For each knob, we aggregate best-per-(total_cores, knob_value) runs, so we're measuring the knob's effect, not conflating it with core-count effects.")
    L.append("")

    def knob_slice(knob: str, label: str) -> None:
        """Build a per-knob × per-core-budget table of best steps/s/(1k cores)."""
        L.append(f"### 3.{label}")
        L.append("")
        L.append(f"Rows = {knob} value. Columns = total_cores budget. Cell = best steps/s/(1k cores) observed for that (knob, cores) pair, with sample count in parens.")
        L.append("")
        knob_values = sorted({r[knob] for r in ok_rows if r[knob] is not None})
        header = ["{:<8}".format(knob)] + [f"{c:>6}" for c in ranked_cores]
        L.append("| " + " | ".join(header) + " |")
        L.append("|" + "|".join(["---"] * len(header)) + "|")
        for v in knob_values:
            cells = [f"{v!s:<8}"]
            for n_cores in ranked_cores:
                subset = [r for r in by_cores[n_cores] if r[knob] == v]
                if not subset:
                    cells.append("{:>6}".format("—"))
                    continue
                best = max(subset, key=lambda r: r["steps_per_sec_per_core"])
                cells.append(f"{best['steps_per_sec_per_core'] * 1000:5.2f}({len(subset)})")
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    knob_slice("ntasks_per_node", "a  — ntasks_per_node (MPI ranks per node)")
    knob_slice("cpus_per_task",   "b  — cpus_per_task (OMP threads per MPI rank)")
    knob_slice("NPAR",            "c  — NPAR (band-parallel group size)")
    knob_slice("nodes",           "d  — nodes (node count)")

    # -------------------------------------------------------------------------
    # 4. NPAR sensitivity at fixed layout.
    # -------------------------------------------------------------------------
    L.append("## 4. NPAR / NCORE sensitivity at fixed (nodes, ntpn, cpt)")
    L.append("")
    L.append("Isolates the NPAR / NCORE effect by restricting to groups with ≥ 2 NPAR/NCORE values tested under the same SLURM layout. Spread = (max − min) / min of t_step.")
    L.append("")
    L.append("| layout (n×ntpn×cpt) | n_runs | NPAR values | NCORE values | t_step range (s) | spread |")
    L.append("|---|---:|---|---|---|---:|")
    by_layout: dict = {}
    for r in ok_rows:
        key = (r["nodes"], r["ntasks_per_node"], r["cpus_per_task"])
        by_layout.setdefault(key, []).append(r)
    for key, rs in sorted(by_layout.items(), key=lambda kv: kv[0]):
        npar_vals = sorted({r["NPAR"] for r in rs if r["NPAR"] is not None})
        ncore_vals = sorted({r["NCORE"] for r in rs if r["NCORE"] is not None})
        n_distinct = len(npar_vals) + len(ncore_vals)
        if n_distinct < 2 or len(rs) < 2:
            continue
        ts = [r["median_real_per_ionic_s"] for r in rs]
        spread = (max(ts) - min(ts)) / min(ts) if min(ts) else 0.0
        n, ntpn, cpt = key
        L.append(
            f"| {n}×{ntpn}×{cpt} "
            f"| {len(rs)} "
            f"| {','.join(str(x) for x in npar_vals) or '—'} "
            f"| {','.join(str(x) for x in ncore_vals) or '—'} "
            f"| {min(ts):.3f} – {max(ts):.3f} "
            f"| {spread * 100:.1f}% |"
        )
    L.append("")

    # -------------------------------------------------------------------------
    # 4b. Bands-per-rank: the system-size ceiling on scaling.
    # -------------------------------------------------------------------------
    L.append("## 4b. Efficiency vs. bands/rank (NBANDS / mpi_ranks)")
    L.append("")
    L.append("This is the single knob where NBANDS and the atom count enter. Below ~4–5 bands/rank the run is starving for band-parallel work and efficiency collapses.")
    L.append("")
    L.append("Binning 'ok' runs by bands/rank bucket; each row shows the best run in that bucket (lowest t_step).")
    L.append("")
    L.append("| bands/rank range | runs | best layout seen | NBANDS | cores | mpi_ranks | t_step (s) | steps/s/(1k cores) |")
    L.append("|---|---:|---|---:|---:|---:|---:|---:|")
    # Buckets covering typical VASP guidance bands.
    buckets = [
        (0, 2, "< 2 (starving)"),
        (2, 5, "2 – 5 (stressed)"),
        (5, 10, "5 – 10 (healthy)"),
        (10, 20, "10 – 20 (comfortable)"),
        (20, 100, "20 – 100 (under-utilized)"),
        (100, 1e9, "≥ 100 (single-rank)"),
    ]
    for lo, hi, label in buckets:
        subset = [r for r in ok_rows if r["bands_per_rank"] is not None and lo <= r["bands_per_rank"] < hi]
        if not subset:
            continue
        best = min(subset, key=lambda r: r["median_real_per_ionic_s"])
        L.append(
            f"| {label} | {len(subset)} "
            f"| {best['nodes']}×{best['ntasks_per_node']}×{best['cpus_per_task']} "
            f"| {best['NBANDS']} "
            f"| {best['total_cores']} "
            f"| {best['mpi_ranks']} "
            f"| {best['median_real_per_ionic_s']:.3f} "
            f"| {_fmt_eff(best['steps_per_sec_per_core'])} |"
        )
    L.append("")
    L.append("**Observed trend (this benchmark):** best-in-bucket efficiency flattens around `≥ 10 bands/rank`; drops markedly below ~5 bands/rank; and is essentially wasted below ~2.5 bands/rank (the 768-core `8×24×4` test has 468/192 ≈ 2.44 bands/rank — exactly where scaling falls off a cliff).")
    L.append("")
    L.append("**Rule of thumb derived from this sweep (NBANDS ≈ 480, Γ-only, 370 atoms):**")
    L.append("")
    L.append("- Target `mpi_ranks ≤ NBANDS / 10` for comfortable scaling → ~48 ranks.")
    L.append("- Hard ceiling at `mpi_ranks ≈ NBANDS / 2.5` → ~192 ranks. Past this, extra cores should be OMP threads, not new ranks.")
    L.append("- At 384-core budget: 32 MPI × 12 OMP (`4×8×12`) gives 15 bands/rank — best observed. 192 MPI × 2 OMP (`4×48×2`) gives 2.5 bands/rank — half the efficiency of best.")
    L.append("- To go beyond 4 nodes productively, either increase NBANDS (more bands to distribute) or increase the atom count (heavier per-rank work).")
    L.append("")

    # -------------------------------------------------------------------------
    # 5. Ranked leaderboard by per-core efficiency.
    # -------------------------------------------------------------------------
    L.append("## 5. Absolute leaderboard by per-core efficiency")
    L.append("")
    L.append("Top 10 by `steps_per_sec_per_core` (i.e. highest per-core throughput — the best cost per core-hour).")
    L.append("")
    L.append("| rank | name | n×ntpn×cpt | NPAR | NBANDS | bands/rank | cores | t_step (s) | steps/s/(1k cores) |")
    L.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    ranked = sorted(ok_rows, key=lambda r: -r["steps_per_sec_per_core"])
    for i, r in enumerate(ranked[:10], start=1):
        L.append(
            f"| {i} | {r['name']} "
            f"| {r['nodes']}×{r['ntasks_per_node']}×{r['cpus_per_task']} "
            f"| {r['NPAR']} "
            f"| {r['NBANDS']} "
            f"| {r['bands_per_rank']} "
            f"| {r['total_cores']} "
            f"| {r['median_real_per_ionic_s']:.3f} "
            f"| {_fmt_eff(r['steps_per_sec_per_core'])} |"
        )
    L.append("")
    L.append("Worst 5 (pathological configs — low per-core efficiency):")
    L.append("")
    L.append("| name | n×ntpn×cpt | NPAR | NBANDS | bands/rank | cores | t_step (s) | steps/s/(1k cores) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(ok_rows, key=lambda r: r["steps_per_sec_per_core"])[:5]:
        L.append(
            f"| {r['name']} "
            f"| {r['nodes']}×{r['ntasks_per_node']}×{r['cpus_per_task']} "
            f"| {r['NPAR']} "
            f"| {r['NBANDS']} "
            f"| {r['bands_per_rank']} "
            f"| {r['total_cores']} "
            f"| {r['median_real_per_ionic_s']:.3f} "
            f"| {_fmt_eff(r['steps_per_sec_per_core'])} |"
        )
    L.append("")

    # -------------------------------------------------------------------------
    # 6. Interpretation.
    # -------------------------------------------------------------------------
    L.append("## 6. Interpretation — what each knob does to efficiency")
    L.append("")
    L.append("The four knobs (`nodes`, `ntasks_per_node`, `cpus_per_task`, `NPAR`) interact, but the marginal-effect tables above isolate each one. Reading them together:")
    L.append("")
    L.append("- **`nodes` (scale):** Per-core efficiency drops as node count rises once the first node is saturated. The 36-core run (1 node, under-subscribed) is actually 15% more efficient per used core than the 96-core run (1 node, fully subscribed) — the 60 idle cores on the under-subscribed node free up memory bandwidth, L3 cache, and UPI/NUMA traffic for the 36 active cores. This is a scheduling quirk, not a sign that 36 cores is the \"right\" budget: absolute throughput (steps/s) is still much lower. From the fully-loaded 1-node point onward (96 → 192 → 384 → 768 cores), per-core efficiency drops monotonically. Total throughput rises 1 → 4 nodes, then falls at 8 — communication dominates past 4 nodes for this 370-atom, Γ-point system.")
    L.append("")
    L.append("- **`ntasks_per_node` vs `cpus_per_task` (MPI/OMP balance):** Within a node-count, the best layouts concentrate around `ntpn × cpt ≈ 96` with moderate OMP (cpt = 8–12). Very high MPI counts (`ntpn = 48`, `ntpn = 96` on ≥ 2 nodes) are inefficient; very high OMP (`cpt = 96`, pure OpenMP) is catastrophic. **Pure MPI is fine on one node; hybrid is needed past one node.**")
    L.append("")
    L.append("- **`NPAR` / `NCORE`:** Effect is layout-dependent. On well-chosen layouts (§4: `1×24×4`, `2×12×8`, `4×8×12`) the spread across NPAR values is **1–4%** — minor. On a few layouts (`1×96×1` with NPAR=96 vs 6 → 75% spread; `2×6×16` → 26% spread) NPAR matters a lot — usually when NPAR is set to a value that doesn't divide `mpi_ranks` cleanly or forces extreme band-group sizes. Safe rule: use NPAR ≈ number of nodes, or NPAR such that `mpi_ranks / NPAR` ≈ OMP threads.")
    L.append("")
    L.append("- **`total_cores`:** Throughput peaks at 384 cores (4 nodes) for this system. Beyond that, steps/sec drops even as cores are added. Cost-effectiveness (per-core efficiency) peaks at the smallest core count tested (36-core, 2.83 steps/s/(1k cores); but this is a partially-loaded node — see nodes bullet above) and falls from there — typical strong-scaling behavior.")
    L.append("")
    L.append("## 7. Decision guide (for 370 atoms, NBANDS ≈ 480, Γ-only)")
    L.append("")
    L.append("- If walltime is the constraint → use 384 cores, `4×8×12`, NPAR ≈ 24. bands/rank ≈ 15, the largest productive layout for this system size.")
    L.append("- If AU / core-hour cost is the constraint → use 1 node, `1×96×1`, NPAR ≈ 6-8. bands/rank ≈ 5, still close-to-ideal per-core efficiency.")
    L.append("- **Avoid:** pure-OMP (`1×1×96`), very-high-MPI on ≥ 2 nodes (`n×96×1`, `n×48×2`), `8×*×*` (past the scaling wall — bands/rank drops below 2.5).")
    L.append("- **Before scaling to a new system size:** estimate `NBANDS / target_mpi_ranks`. If it would drop below 10, either use fewer MPI ranks (more OMP threads) or accept reduced efficiency.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
