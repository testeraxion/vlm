from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable


def parse_kitti_calib(path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        parsed: list[float] = []
        for item in parts:
            try:
                parsed.append(float(item))
            except ValueError:
                pass
        data[key.strip()] = parsed
    return data


def parse_timestamps(path: Path, max_frames: int | None = None) -> list[float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    def _parse_kitti_time(raw: str) -> datetime:
        # KITTI raw timestamps can include nanoseconds; datetime supports microseconds.
        normalized = raw.replace(" ", "T", 1)
        if "." in normalized:
            head, frac = normalized.split(".", 1)
            digits = "".join(ch for ch in frac if ch.isdigit())
            normalized = f"{head}.{digits[:6].ljust(6, '0')}"
        return datetime.fromisoformat(normalized)

    stamps: list[datetime] = [_parse_kitti_time(x) for x in lines]
    if max_frames is not None:
        stamps = stamps[:max_frames]
    if not stamps:
        return []
    t0 = stamps[0]
    return [(x - t0).total_seconds() for x in stamps]


def write_yaml_from_calib(calib_path: Path, yaml_path: Path, image_pair: tuple[str, str]) -> None:
    calib = parse_kitti_calib(calib_path)
    left_suffix = image_pair[0].split("_")[-1]
    right_suffix = image_pair[1].split("_")[-1]

    left_key = f"P_rect_{left_suffix}"
    right_key = f"P_rect_{right_suffix}"
    left_size_key = f"S_rect_{left_suffix}"

    p_left = calib[left_key]
    p_right = calib[right_key]
    size = calib[left_size_key]

    fx = p_left[0]
    fy = p_left[5]
    cx = p_left[2]
    cy = p_left[6]
    width = int(size[0])
    height = int(size[1])
    baseline = abs(p_right[3] / p_right[0] - p_left[3] / p_left[0])

    yaml_text = f'''%YAML:1.0

File.version: "1.0"
Camera.type: "Rectified"

Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}

Camera.width: {width}
Camera.height: {height}
Camera.fps: 10

Stereo.b: {baseline:.6f}
Camera.RGB: 0
Stereo.ThDepth: 35.0

ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.6
Viewer.KeyFrameLineWidth: 2.0
Viewer.GraphLineWidth: 1.0
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.7
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -100.0
Viewer.ViewpointZ: -0.1
Viewer.ViewpointF: 2000.0
'''
    yaml_path.write_text(yaml_text, encoding="utf-8")


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    import shutil
    shutil.copy2(src, dst)


def find_calib_for_drive(dataset_root: Path, sequence_id: str) -> Path | None:
    """Find calibration file for a given drive sequence.

    In KITTI Raw, calibration is per-recording setup, not per-drive.
    Drive 0011 uses calib_011, drive 0034 uses calib_034, etc.
    """
    # Extract the numeric part from the drive ID
    # e.g., 2011_09_26_drive_0011_sync -> 0011 -> 011
    parts = sequence_id.split("_drive_")
    if len(parts) != 2:
        return None
    date_part = parts[0]  # e.g., 2011_09_26
    drive_num = parts[1].split("_")[0]  # e.g., 0011

    # Try different calib number formats
    for num_variant in [drive_num, drive_num.lstrip("0") or "0"]:
        calib_id = f"{date_part}_calib_{num_variant}"
        for calib_candidate in dataset_root.rglob("calib_cam_to_cam.txt"):
            if calib_id in str(calib_candidate):
                return calib_candidate

    # Fallback: use any calibration from the same date
    for calib_candidate in dataset_root.rglob("calib_cam_to_cam.txt"):
        if date_part in str(calib_candidate):
            return calib_candidate

    return None


def prepare_sequence(repo_root: Path, sequence_id: str, max_frames: int | None, image_pair: tuple[str, str]) -> Path:
    # Discover the actual drive directory by searching for the sequence
    dataset_root = repo_root / "dataset"
    drive_dir = None
    # Try multiple nesting patterns
    for pattern in [f"*/*/{sequence_id}", f"*/{sequence_id}", sequence_id]:
        for candidate in dataset_root.glob(pattern):
            if candidate.is_dir() and (candidate / "oxts").exists():
                drive_dir = candidate
                break
        if drive_dir is not None:
            break
    if drive_dir is None or not drive_dir.exists():
        raise FileNotFoundError(f"Cannot find drive directory for {sequence_id}")

    # Discover calibration directory
    calib_path = find_calib_for_drive(dataset_root, sequence_id)
    if calib_path is None or not calib_path.exists():
        raise FileNotFoundError(f"Missing calibration file for {sequence_id}")

    left_src = drive_dir / image_pair[0] / "data"
    right_src = drive_dir / image_pair[1] / "data"
    timestamps_path = drive_dir / image_pair[0] / "timestamps.txt"

    if not left_src.exists() or not right_src.exists():
        raise FileNotFoundError(f"Missing stereo image folders for {sequence_id}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calibration file for {sequence_id}: {calib_path}")

    prepared_root = repo_root / "slam" / "prepared" / sequence_id
    image0_dir = prepared_root / "image_0"
    image1_dir = prepared_root / "image_1"
    image0_dir.mkdir(parents=True, exist_ok=True)
    image1_dir.mkdir(parents=True, exist_ok=True)

    left_files = sorted(left_src.glob("*.png"))
    right_files = sorted(right_src.glob("*.png"))
    n = min(len(left_files), len(right_files))
    if max_frames is not None:
        n = min(n, max_frames)

    rel_times = parse_timestamps(timestamps_path, max_frames=n)
    if len(rel_times) < n:
        n = len(rel_times)

    for idx in range(n):
        name = f"{idx:06d}.png"
        symlink_or_copy(left_files[idx], image0_dir / name)
        symlink_or_copy(right_files[idx], image1_dir / name)

    (prepared_root / "times.txt").write_text("\n".join(f"{t:.9f}" for t in rel_times[:n]) + "\n", encoding="utf-8")
    write_yaml_from_calib(calib_path, prepared_root / "settings.yaml", image_pair=image_pair)
    return prepared_root


def discover_sequence_ids(dataset_root: Path) -> Iterable[str]:
    for drive_dir in sorted(dataset_root.glob("*_drive_*_sync")):
        yield drive_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KITTI raw sequences for ORB-SLAM3 stereo_kitti")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sequence-id", type=str, default=None, help="Specific sequence id, e.g. 2011_09_26_drive_0009_sync")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-pair", nargs=2, default=("image_00", "image_01"), help="Stereo image folders to use")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    dataset_root = repo_root / "dataset"
    sequence_ids = [args.sequence_id] if args.sequence_id else list(discover_sequence_ids(dataset_root))

    for sequence_id in sequence_ids:
        prepared = prepare_sequence(repo_root, sequence_id, args.max_frames, tuple(args.image_pair))
        print(f"Prepared {sequence_id} at {prepared}")


if __name__ == "__main__":
    main()
