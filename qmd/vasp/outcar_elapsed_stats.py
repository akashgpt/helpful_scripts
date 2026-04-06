#!/usr/bin/env python3
"""Summarize VASP OUTCAR elapsed times.

This utility was inspired by
`ALCHEMY__dev/TRAIN_MLMD_scripts/ALCHEMY_timing.sh`, but it is tailored to a
common VASP pattern: the final `Elapsed time (sec):` line is usually near the
end of each `OUTCAR`.

The script therefore scans matching `OUTCAR` files recursively, inspects only
the last N lines of each file, extracts elapsed times, and reports descriptive
statistics.

Usage examples:
    Basic summary for a DIR_X directory:
        python3 $HELP_SCRIPTS_vasp/outcar_elapsed_stats.py DIR_X --last-per-file

    Use a wider tail window and print the slowest runs:
        python3 $HELP_SCRIPTS_vasp/outcar_elapsed_stats.py DIR_X --tail-lines 30 --last-per-file --show-slowest 10

    Group statistics by the first parent-directory level below `BASE_DIR`:
        python3 $HELP_SCRIPTS_vasp/outcar_elapsed_stats.py DIR_X --last-per-file --group-by-parent-depth 1
"""

import argparse
import math
import os
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union


ELAPSED_PATTERN = re.compile(r"Elapsed time \(sec\):\s*([0-9]+(?:\.[0-9]+)?)")
USAGE_EXAMPLES = """Examples:
  python3 $HELP_SCRIPTS_vasp/outcar_elapsed_stats.py DIR_X --last-per-file

  python3 $HELP_SCRIPTS_vasp/outcar_elapsed_stats.py DIR_X --tail-lines 30 --last-per-file --show-slowest 10

  python3 $HELP_SCRIPTS_vasp/outcar_elapsed_stats.py DIR_X --last-per-file --group-by-parent-depth 1
"""


class ElapsedRecord(NamedTuple):
    """Store one parsed elapsed-time record.

    Attributes:
        path: Path to the OUTCAR file that contained the elapsed line.
        seconds: Parsed elapsed time in seconds.
    """

    path: Path
    seconds: float


class SummaryStats(NamedTuple):
    """Store descriptive statistics for elapsed times.

    Attributes:
        count: Number of elapsed values used for the summary.
        mean: Arithmetic mean in seconds.
        stddev: Population standard deviation in seconds.
        minimum: Minimum elapsed time in seconds.
        p10: 10th percentile in seconds.
        median: Median elapsed time in seconds.
        p90: 90th percentile in seconds.
        maximum: Maximum elapsed time in seconds.
        total: Sum of all elapsed times in seconds.
    """

    count: int
    mean: float
    stddev: float
    minimum: float
    p10: float
    median: float
    p90: float
    maximum: float
    total: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional sequence of command-line arguments.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Collect 'Elapsed time (sec):' values from the last lines of OUTCAR "
            "files and report statistics such as mean, standard deviation, "
            "median, and percentiles."
        ),
        epilog=USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "BASE_DIR",
        nargs="?",
        default=".",
        help="Root directory to search recursively for OUTCAR files.",
    )
    parser.add_argument(
        "--include",
        default="OUTCAR",
        help="Filename glob used during recursive search. Default: OUTCAR",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=15,
        help="Number of trailing lines to inspect in each OUTCAR. Default: 15",
    )
    parser.add_argument(
        "--last-per-file",
        action="store_true",
        help=(
            "If multiple elapsed lines appear within the inspected tail region, "
            "keep only the last one from each file."
        ),
    )
    parser.add_argument(
        "--group-by-parent-depth",
        type=int,
        default=0,
        help=(
            "Also report grouped statistics using the first N parent-directory "
            "levels relative to BASE_DIR. Example: N=1 groups by the first "
            "directory below BASE_DIR."
        ),
    )
    parser.add_argument(
        "--show-slowest",
        type=int,
        default=0,
        help="Show the N slowest matched elapsed records.",
    )
    return parser.parse_args(argv)


def iter_outcar_paths(BASE_DIR: Path, include_name: str) -> Iterator[Path]:
    """Yield matching OUTCAR paths below a base directory.

    Args:
        BASE_DIR: Root directory to search.
        include_name: Filename glob used during recursive search.

    Yields:
        Matching file paths.
    """

    for path in BASE_DIR.rglob(include_name):
        if path.is_file():
            yield path


def read_tail_lines(path: Path, tail_lines: int) -> List[str]:
    """Read only the trailing lines from a text file.

    Args:
        path: File path to read.
        tail_lines: Number of trailing lines to keep.

    Returns:
        The trailing lines with newlines stripped.
    """

    # Seek from the end instead of reading the entire file through a deque.
    # Start with an 8 KiB chunk; double if not enough lines.
    file_size = os.path.getsize(path)
    chunk_size = 8192

    with open(path, "rb") as handle:
        if file_size <= chunk_size:
            data = handle.read().decode("utf-8", errors="ignore")
        else:
            while True:
                seek_pos = max(file_size - chunk_size, 0)
                handle.seek(seek_pos)
                data = handle.read().decode("utf-8", errors="ignore")
                # +1 because the first "line" may be partial
                if data.count("\n") >= tail_lines + 1 or seek_pos == 0:
                    break
                chunk_size *= 2

    return data.splitlines()[-tail_lines:]


def collect_records_from_file(
    path: Path,
    tail_lines: int,
    last_per_file: bool,
) -> List[ElapsedRecord]:
    """Collect elapsed-time records from one OUTCAR file.

    Args:
        path: Path to one OUTCAR file.
        tail_lines: Number of trailing lines to inspect.
        last_per_file: Whether to keep only the last match from the tail region.

    Returns:
        Parsed elapsed-time records from the file.
    """

    trailing_lines = read_tail_lines(path, tail_lines)
    matched_values = []  # type: List[float]

    for line in trailing_lines:
        match = ELAPSED_PATTERN.search(line)
        if match is not None:
            matched_values.append(float(match.group(1)))

    if not matched_values:
        return []

    if last_per_file:
        return [ElapsedRecord(path=path, seconds=matched_values[-1])]

    return [ElapsedRecord(path=path, seconds=value) for value in matched_values]


def collect_records(
    outcar_paths: Sequence[Path],
    tail_lines: int,
    last_per_file: bool,
) -> List[ElapsedRecord]:
    """Collect elapsed-time records from matching OUTCAR files.

    Args:
        outcar_paths: Matching OUTCAR file paths.
        tail_lines: Number of trailing lines to inspect.
        last_per_file: Whether to keep only the last match per file.

    Returns:
        Parsed elapsed-time records.
    """

    # Use a process pool to read files in parallel (I/O-bound workload).
    # Cap workers so we don't overwhelm the filesystem on large runs.
    num_workers = min(8, len(outcar_paths)) if outcar_paths else 1
    records = []  # type: List[ElapsedRecord]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(collect_records_from_file, path, tail_lines, last_per_file)
            for path in outcar_paths
        ]
        for future in futures:
            records.extend(future.result())

    return records


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Compute a percentile using linear interpolation.

    Args:
        sorted_values: Values sorted in ascending order.
        fraction: Percentile as a fraction between 0.0 and 1.0.

    Returns:
        The interpolated percentile value.
    """

    if not sorted_values:
        raise ValueError("percentile() requires at least one value.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = fraction * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    if lower_index == upper_index:
        return lower_value

    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def compute_summary(values: Sequence[float]) -> SummaryStats:
    """Compute descriptive statistics for elapsed times.

    Args:
        values: Elapsed times in seconds.

    Returns:
        Computed descriptive statistics.
    """

    if not values:
        raise ValueError("compute_summary() requires at least one value.")

    sorted_values = sorted(values)
    total = math.fsum(sorted_values)
    return SummaryStats(
        count=len(sorted_values),
        mean=total / float(len(sorted_values)),
        stddev=statistics.pstdev(sorted_values),
        minimum=sorted_values[0],
        p10=percentile(sorted_values, 0.10),
        median=statistics.median(sorted_values),
        p90=percentile(sorted_values, 0.90),
        maximum=sorted_values[-1],
        total=total,
    )


def natural_sort_key(text: str) -> List[Tuple[int, Union[int, str]]]:
    """Build a natural-sort key for directory labels.

    Args:
        text: Input text to sort.

    Returns:
        A sort key that keeps numeric chunks in numeric order.
    """

    key = []  # type: List[Tuple[int, Union[int, str]]]
    for chunk in re.split(r"([0-9]+)", text):
        if not chunk:
            continue
        if chunk.isdigit():
            key.append((0, int(chunk)))
        else:
            key.append((1, chunk.lower()))
    return key


def group_records(
    records: Sequence[ElapsedRecord],
    BASE_DIR: Path,
    parent_depth: int,
) -> Dict[str, List[float]]:
    """Group elapsed times by leading parent-directory components.

    Args:
        records: Parsed elapsed-time records.
        BASE_DIR: Root directory used for the search.
        parent_depth: Number of relative parent-directory levels to use.

    Returns:
        Grouped elapsed times keyed by relative parent-directory label.
    """

    grouped = {}  # type: Dict[str, List[float]]
    for record in records:
        relative_parent = record.path.relative_to(BASE_DIR).parent
        if parent_depth <= 0:
            group_label = str(relative_parent)
        else:
            parts = relative_parent.parts[:parent_depth]
            group_label = "/".join(parts) if parts else "."
        grouped.setdefault(group_label, []).append(record.seconds)
    return grouped


def format_seconds(seconds: float) -> str:
    """Format seconds in a human-readable way.

    Args:
        seconds: Time in seconds.

    Returns:
        Human-readable time string.
    """

    if seconds < 60:
        return "{0:.2f} s".format(seconds)
    if seconds < 3600:
        return "{0:.2f} min".format(seconds / 60.0)
    return "{0:.2f} h".format(seconds / 3600.0)


def print_summary(label: str, summary: SummaryStats) -> None:
    """Print one summary block.

    Args:
        label: Summary label.
        summary: Statistics to print.
    """

    print(label)
    print("  count   : {0}".format(summary.count))
    print("  mean    : {0:.3f} s".format(summary.mean))
    print("  stddev  : {0:.3f} s".format(summary.stddev))
    print("  min     : {0:.3f} s".format(summary.minimum))
    print("  p10     : {0:.3f} s".format(summary.p10))
    print("  median  : {0:.3f} s".format(summary.median))
    print("  p90     : {0:.3f} s".format(summary.p90))
    print("  max     : {0:.3f} s".format(summary.maximum))
    print("  total   : {0:.3f} s  ({1})".format(summary.total, format_seconds(summary.total)))


def print_grouped_summaries(
    records: Sequence[ElapsedRecord],
    BASE_DIR: Path,
    parent_depth: int,
) -> None:
    """Print grouped summaries for elapsed times.

    Args:
        records: Parsed elapsed-time records.
        BASE_DIR: Root directory used for the search.
        parent_depth: Number of leading parent-directory levels to group by.
    """

    if parent_depth <= 0:
        return

    grouped = group_records(records, BASE_DIR, parent_depth)
    if not grouped:
        return

    print("")
    print("Grouped statistics by first {0} parent level(s):".format(parent_depth))
    header = "{0:<40} {1:>7} {2:>12} {3:>12} {4:>12} {5:>12} {6:>12}".format(
        "group",
        "n",
        "mean(s)",
        "std(s)",
        "median(s)",
        "min(s)",
        "max(s)",
    )
    print(header)
    print("-" * len(header))

    for group_label in sorted(grouped, key=natural_sort_key):
        summary = compute_summary(grouped[group_label])
        print(
            "{0:<40} {1:>7d} {2:>12.3f} {3:>12.3f} {4:>12.3f} {5:>12.3f} {6:>12.3f}".format(
                group_label,
                summary.count,
                summary.mean,
                summary.stddev,
                summary.median,
                summary.minimum,
                summary.maximum,
            )
        )


def print_slowest_records(records: Sequence[ElapsedRecord], limit: int) -> None:
    """Print the slowest elapsed-time records.

    Args:
        records: Parsed elapsed-time records.
        limit: Number of slowest records to print.
    """

    if limit <= 0 or not records:
        return

    print("")
    print("Slowest {0} record(s):".format(min(limit, len(records))))
    for record in sorted(records, key=lambda item: item.seconds, reverse=True)[:limit]:
        print("  {0:10.3f} s  {1}".format(record.seconds, record.path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: Optional sequence of command-line arguments.

    Returns:
        Process exit code.
    """

    args = parse_args(argv)
    BASE_DIR = Path(args.BASE_DIR).expanduser().resolve()

    if not BASE_DIR.exists():
        print("Error: BASE_DIR does not exist: {0}".format(BASE_DIR), file=sys.stderr)
        return 1
    if not BASE_DIR.is_dir():
        print("Error: BASE_DIR is not a directory: {0}".format(BASE_DIR), file=sys.stderr)
        return 1
    if args.tail_lines <= 0:
        print("Error: --tail-lines must be a positive integer.", file=sys.stderr)
        return 1

    outcar_paths = list(iter_outcar_paths(BASE_DIR, args.include))
    records = collect_records(
        outcar_paths=outcar_paths,
        tail_lines=args.tail_lines,
        last_per_file=args.last_per_file,
    )

    if not records:
        print("No elapsed times found under {0}".format(BASE_DIR))
        print("  OUTCARs found: {0}".format(len(outcar_paths)))
        return 0

    values = [record.seconds for record in records]
    summary = compute_summary(values)
    unique_files = len(set(record.path for record in records))

    print("OUTCAR Elapsed-Time Statistics")
    print("  BASE_DIR     : {0}".format(BASE_DIR))
    print("  include      : {0}".format(args.include))
    print("  OUTCARs found: {0}".format(len(outcar_paths)))
    print("  tail lines   : {0}".format(args.tail_lines))
    print(
        "  counting     : {0}".format(
            "last elapsed value per file"
            if args.last_per_file
            else "every matched elapsed line in the tail window"
        )
    )
    print("  records      : {0}".format(summary.count))
    print("  unique files : {0}".format(unique_files))
    print("")

    print_summary("Global summary:", summary)
    print_grouped_summaries(records, BASE_DIR, args.group_by_parent_depth)
    print_slowest_records(records, args.show_slowest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
