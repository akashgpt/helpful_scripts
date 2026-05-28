#!/usr/bin/env python3
"""Create GIF animations from LAMMPS current snapshots or dump frames.

Examples:
    # Current directory: sample npt.dump every frame and create a GIF.
    python make_lammps_snapshot_gif.py --elements H N

    # Current directory: sample every tenth dump frame.
    python make_lammps_snapshot_gif.py --elements H N -s 10

    # Current directory: make one combined GIF from all immediate subdirectories.
    python make_lammps_snapshot_gif.py -d sub --elements H N
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SNAPSHOT_SCRIPT = SCRIPT_DIR / "plot_lammps_current_snapshot.py"
DEFAULT_MAX_FRAMES = 1000
DEFAULT_PAUSE_SECONDS = 0.8
DEFAULT_OUTPUT_WIDTH = 1400


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument vector. If None, argparse reads from sys.argv.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build a GIF using the existing LAMMPS current-snapshot helper. "
            "By default this samples npt.dump in the selected directory. Use "
            "'-d sub' to build one combined GIF from immediate subdirectories."
        ),
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest="target_dir",
        default=".",
        help=(
            "Directory to process. Default: current directory. Use the literal "
            "'sub' to process all immediate subdirectories of the current directory."
        ),
    )
    parser.add_argument(
        "-f",
        "--file",
        "--structure",
        dest="structure",
        default="npt.dump",
        help="LAMMPS dump or data filename to read. Default: npt.dump.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "lammps-dump-text", "lammps-data"),
        default="auto",
        help="Input format passed to the snapshot helper. Default: auto.",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        help=(
            "Element symbols ordered by LAMMPS type id, e.g. '--elements H N'. "
            "Passed through to the snapshot helper."
        ),
    )
    parser.add_argument(
        "--lammps-data-style",
        default="atomic",
        help="LAMMPS data style for lammps-data files. Default: atomic.",
    )
    parser.add_argument(
        "--dt-ps",
        type=float,
        help="LAMMPS timestep size in ps. Passed through to the snapshot helper.",
    )
    parser.add_argument(
        "--lammps-input",
        type=Path,
        help="LAMMPS input file used by the snapshot helper for timestep/conditions.",
    )
    parser.add_argument(
        "--no-wrap",
        action="store_true",
        help="Do not wrap positions before plotting. Passed through to the snapshot helper.",
    )
    parser.add_argument(
        "-s",
        "--skip",
        type=int,
        default=1,
        help="Use every Nth trajectory frame. Default: 1.",
    )
    parser.add_argument(
        "-p",
        "--pause",
        type=float,
        default=DEFAULT_PAUSE_SECONDS,
        help=f"Pause between GIF frames in seconds. Default: {DEFAULT_PAUSE_SECONDS}.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Maximum number of frames to include. Default: {DEFAULT_MAX_FRAMES}.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_OUTPUT_WIDTH,
        help=f"Resize GIF frames to this pixel width. Default: {DEFAULT_OUTPUT_WIDTH}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output GIF path. Default is placed under analysis/ in the processed directory.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep temporary trajectory PNG frames after writing the GIF.",
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help=(
            "For subdirectory mode, reuse analysis/current_lammps_snapshot.png "
            "when it already exists instead of regenerating it."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate user arguments.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If an argument has an invalid value.
    """
    if args.skip < 1:
        raise ValueError("--skip must be >= 1")
    if args.pause < 0:
        raise ValueError("--pause must be >= 0")
    if args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")
    if args.width < 1:
        raise ValueError("--width must be >= 1")
    if not SNAPSHOT_SCRIPT.is_file():
        raise FileNotFoundError(f"Snapshot helper not found: {SNAPSHOT_SCRIPT}")


def resolve_target(target_dir_text: str) -> tuple[Path, bool]:
    """Resolve the target directory and requested mode.

    Args:
        target_dir_text: User-provided ``-d/--dir`` value.

    Returns:
        Tuple of base directory and whether subdirectory mode was explicitly requested.
    """
    if target_dir_text.strip().lower() == "sub":
        return Path.cwd(), True
    return Path(target_dir_text).expanduser().resolve(), False


def numeric_sort_key(path: Path) -> tuple[int, int | str]:
    """Return a stable sort key that treats numeric folder names numerically.

    Args:
        path: Path to sort.

    Returns:
        Sort key.
    """
    if path.name.isdigit():
        return 0, int(path.name)
    return 1, path.name


def get_immediate_subdirectories(base_dir: Path) -> list[Path]:
    """Return immediate subdirectories in human/numeric order.

    Args:
        base_dir: Parent directory.

    Returns:
        Sorted immediate subdirectories.
    """
    return sorted(
        [path for path in base_dir.iterdir() if path.is_dir()],
        key=numeric_sort_key,
    )


def read_lammps_dump_timesteps(dump_path: Path) -> list[str]:
    """Read all timestep labels from a LAMMPS text dump.

    Args:
        dump_path: LAMMPS dump path.

    Returns:
        Ordered timestep labels.
    """
    timesteps: list[str] = []
    expect_timestep = False
    with dump_path.open("r", encoding="utf-8", errors="replace") as dump_file:
        for line in dump_file:
            if expect_timestep:
                timesteps.append(line.strip())
                expect_timestep = False
                continue
            if line.strip() == "ITEM: TIMESTEP":
                expect_timestep = True
    return timesteps


def choose_sampled_items(items: list[str], skip: int, max_frames: int) -> list[str]:
    """Choose sampled trajectory items with a maximum frame count.

    Args:
        items: Ordered trajectory frame labels.
        skip: Keep every ``skip``-th item.
        max_frames: Maximum number of output frames.

    Returns:
        Sampled item labels.
    """
    sampled = items[::skip]
    if not sampled and items:
        sampled = [items[-1]]
    if len(sampled) <= max_frames:
        return sampled
    if max_frames == 1:
        return [sampled[-1]]
    stride = (len(sampled) - 1) / float(max_frames - 1)
    indices = [round(index * stride) for index in range(max_frames)]
    return [sampled[index] for index in indices]


def find_subdir_structure(subdir: Path, preferred_name: str) -> Path | None:
    """Find the best LAMMPS structure source for one subdirectory snapshot.

    Args:
        subdir: LAMMPS run directory.
        preferred_name: Preferred dump/data filename.

    Returns:
        Structure path, or None when no supported file exists.
    """
    preferred_path = subdir / preferred_name
    if preferred_path.is_file():
        return preferred_path
    for pattern in ("npt.dump", "*.dump", "*.lammpstrj", "conf.lmp", "*.lmp", "*.data"):
        candidates = sorted(subdir.glob(pattern), key=numeric_sort_key)
        if candidates:
            return candidates[0]
    return None


def build_helper_command(
    structure_path: Path,
    output_path: Path,
    title: str,
    args: argparse.Namespace,
    frame_label: str | None = None,
) -> list[str]:
    """Build a snapshot-helper command.

    Args:
        structure_path: LAMMPS dump or data path.
        output_path: PNG output path.
        title: Plot title.
        args: Parsed wrapper arguments.
        frame_label: Optional LAMMPS timestep label.

    Returns:
        Command argument list.
    """
    command = [
        sys.executable,
        str(SNAPSHOT_SCRIPT),
        "-f",
        str(structure_path),
        "--format",
        args.format,
        "--lammps-data-style",
        args.lammps_data_style,
        "--output",
        str(output_path),
        "--title",
        title,
    ]
    if frame_label is not None:
        command.extend(["-t", frame_label])
    else:
        command.extend(["-t", "-1"])
    if args.elements:
        command.append("--elements")
        command.extend(args.elements)
    if args.dt_ps is not None:
        command.extend(["--dt-ps", str(args.dt_ps)])
    if args.lammps_input is not None:
        command.extend(["--lammps-input", str(args.lammps_input)])
    if args.no_wrap:
        command.append("--no-wrap")
    return command


def run_snapshot_helper(
    cwd: Path,
    structure_path: Path,
    output_path: Path,
    title: str,
    args: argparse.Namespace,
    frame_label: str | None = None,
) -> None:
    """Run the existing LAMMPS snapshot helper.

    Args:
        cwd: Working directory for the helper.
        structure_path: LAMMPS dump or data path.
        output_path: PNG output path.
        title: Plot title.
        args: Parsed wrapper arguments.
        frame_label: Optional LAMMPS timestep label.

    Raises:
        subprocess.CalledProcessError: If the helper fails.
    """
    command = build_helper_command(structure_path, output_path, title, args, frame_label)
    subprocess.run(command, cwd=cwd, check=True)


def generate_subdirectory_frames(
    base_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    """Generate or collect one current snapshot per immediate subdirectory.

    Args:
        base_dir: Parent directory whose subdirectories should be processed.
        args: Parsed wrapper arguments.

    Returns:
        Ordered PNG frame paths.
    """
    frame_paths: list[Path] = []
    for subdir in get_immediate_subdirectories(base_dir):
        structure_path = find_subdir_structure(subdir, args.structure)
        if structure_path is None:
            continue
        output_path = subdir / "analysis" / "current_lammps_snapshot.png"
        if not args.use_existing or not output_path.is_file():
            run_snapshot_helper(
                cwd=subdir,
                structure_path=Path(structure_path.name),
                output_path=Path("analysis/current_lammps_snapshot.png"),
                title=f"{base_dir.name} / {subdir.name}",
                args=args,
            )
        frame_paths.append(output_path)
    return frame_paths


def generate_dump_frames(
    base_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    """Generate sampled snapshot PNGs from the current directory's LAMMPS dump.

    Args:
        base_dir: Directory containing the dump.
        args: Parsed wrapper arguments.

    Returns:
        Ordered PNG frame paths.

    Raises:
        FileNotFoundError: If the dump is missing.
        ValueError: If no dump frames are found.
    """
    dump_path = base_dir / args.structure
    if not dump_path.is_file():
        raise FileNotFoundError(f"LAMMPS structure file not found: {dump_path}")
    timesteps = read_lammps_dump_timesteps(dump_path)
    if not timesteps:
        raise ValueError(
            f"No ITEM: TIMESTEP frames found in {dump_path}. "
            "For lammps-data files, use -d sub or make a single PNG with the snapshot helper."
        )
    selected_timesteps = choose_sampled_items(timesteps, args.skip, args.max_frames)
    frame_dir = base_dir / "analysis" / "gif_frames_lammps"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for frame_number, timestep in enumerate(selected_timesteps, start=1):
        output_path = frame_dir / f"lammps_frame_{frame_number:06d}.png"
        run_snapshot_helper(
            cwd=base_dir,
            structure_path=Path(args.structure),
            output_path=output_path.relative_to(base_dir),
            title=f"{base_dir.name} / LAMMPS timestep {timestep}",
            args=args,
            frame_label=timestep,
        )
        frame_paths.append(output_path)
    return frame_paths


def resize_image(image: object, width: int) -> object:
    """Resize a PIL image to the requested width while preserving aspect ratio.

    Args:
        image: PIL Image object.
        width: Target width in pixels.

    Returns:
        Resized PIL Image object.
    """
    if image.width == width:
        return image
    height = max(1, round(image.height * (width / float(image.width))))
    return image.resize((width, height))


def write_gif(
    frame_paths: list[Path],
    output_path: Path,
    pause_seconds: float,
    width: int,
) -> None:
    """Write a GIF from PNG frame paths.

    Args:
        frame_paths: Ordered PNG frames.
        output_path: GIF output path.
        pause_seconds: Pause between frames in seconds.
        width: Resize width in pixels.

    Raises:
        RuntimeError: If Pillow is unavailable.
        ValueError: If no frames are available.
    """
    if not frame_paths:
        raise ValueError("No frames were generated; cannot write GIF.")
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required to write GIFs. Try `python -m pip install pillow`.") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        images.append(resize_image(image, width))
    duration_ms = int(round(pause_seconds * 1000.0))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    for image in images:
        image.close()


def cleanup_frames(frame_paths: list[Path], keep_frames: bool) -> None:
    """Remove temporary trajectory frames when requested.

    Args:
        frame_paths: Generated frame paths.
        keep_frames: Whether frames should be preserved.
    """
    if keep_frames:
        return
    for frame_path in frame_paths:
        if "gif_frames_lammps" in frame_path.parts and frame_path.is_file():
            frame_path.unlink()
    for frame_path in frame_paths:
        parent = frame_path.parent
        if parent.name == "gif_frames_lammps" and parent.is_dir():
            try:
                parent.rmdir()
            except OSError:
                pass
            break


def default_output_path(base_dir: Path, subdir_mode: bool) -> Path:
    """Return the default output GIF path.

    Args:
        base_dir: Processed directory.
        subdir_mode: Whether the GIF uses subdirectory frames.

    Returns:
        Default GIF path.
    """
    if subdir_mode:
        return base_dir / "analysis" / "current_lammps_snapshots_subdirs.gif"
    return base_dir / "analysis" / "current_lammps_snapshots_dump.gif"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LAMMPS snapshot GIF workflow.

    Args:
        argv: Optional argument vector. If None, argparse reads from sys.argv.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    validate_args(args)
    base_dir, explicit_subdir_mode = resolve_target(args.target_dir)
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Target directory not found: {base_dir}")

    subdirectories = get_immediate_subdirectories(base_dir)
    subdir_mode = explicit_subdir_mode or (
        not (base_dir / args.structure).is_file() and bool(subdirectories)
    )
    if subdir_mode:
        frame_paths = generate_subdirectory_frames(base_dir, args)
    else:
        frame_paths = generate_dump_frames(base_dir, args)

    output_path = args.output or default_output_path(base_dir, subdir_mode)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    write_gif(frame_paths, output_path, args.pause, args.width)
    cleanup_frames(frame_paths, args.keep_frames or subdir_mode)
    print(f"Wrote {output_path}")
    print(f"Frames: {len(frame_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
