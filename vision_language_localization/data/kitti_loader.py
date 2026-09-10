from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class FrameRecord:
    timestamp_s: float
    lat: float
    lon: float
    alt: float
    roll: float
    pitch: float
    yaw: float
    speed_mps: float
    image_path: Path | None


@dataclass
class SequenceData:
    sequence_id: str
    frames: list[FrameRecord]
    gt_xyz: np.ndarray


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate local ENU coordinates from lat/lon with an equirectangular model."""
    lat0 = float(lat[0])
    lon0 = float(lon[0])
    earth_radius_m = 6378137.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    x = (lon_rad - lon0_rad) * np.cos(lat0_rad) * earth_radius_m
    y = (lat_rad - lat0_rad) * earth_radius_m
    return x, y


def _read_oxts_frame(file_path: Path, image_dir: Path | None = None) -> FrameRecord:
    values = file_path.read_text(encoding="utf-8").strip().split()
    if len(values) < 11:
        raise ValueError(f"Unexpected OXTS format in {file_path}")

    lat, lon, alt = map(float, values[0:3])
    roll, pitch, yaw = map(float, values[3:6])
    vn, ve = map(float, values[6:8])
    speed_mps = float(np.hypot(vn, ve))

    frame_idx = int(file_path.stem)
    timestamp_s = frame_idx * 0.1
    image_path: Path | None = None
    if image_dir is not None:
        candidate = image_dir / f"{file_path.stem}.png"
        if candidate.exists():
            image_path = candidate

    return FrameRecord(
        timestamp_s=timestamp_s,
        lat=lat,
        lon=lon,
        alt=alt,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        speed_mps=speed_mps,
        image_path=image_path,
    )


def discover_drive_roots(dataset_root: Path) -> list[Path]:
    """Discover KITTI drive directories with OXTS and image data.

    Supports two common KITTI Raw layouts:
    - dataset/2011_09_26_drive_0009_sync/2011_09_26/2011_09_26_drive_0009_sync/
    - dataset/2011_09_26/2011_09_26_drive_0009_sync/
    """
    drives: list[Path] = []
    seen: set[Path] = set()

    # Strategy 1: find all directories containing oxts/data
    for oxts_dir in dataset_root.rglob("oxts/data"):
        drive_root = oxts_dir.parent.parent
        if drive_root not in seen and drive_root.exists():
            if (drive_root / "image_02").exists() or (drive_root / "oxts").exists():
                drives.append(drive_root)
                seen.add(drive_root)

    return sorted(drives)


def load_kitti_sequence(drive_root: Path, max_frames: int | None = None) -> SequenceData:
    oxts_dir = drive_root / "oxts" / "data"
    image_dir = drive_root / "image_02" / "data"
    if not oxts_dir.exists():
        raise FileNotFoundError(f"Missing OXTS folder: {oxts_dir}")

    frame_files = sorted(oxts_dir.glob("*.txt"))
    if max_frames is not None:
        frame_files = frame_files[:max_frames]

    frames = [_read_oxts_frame(path, image_dir=image_dir if image_dir.exists() else None) for path in frame_files]
    if not frames:
        raise ValueError(f"No OXTS frames found in {oxts_dir}")

    lat = np.array([f.lat for f in frames], dtype=np.float64)
    lon = np.array([f.lon for f in frames], dtype=np.float64)
    alt = np.array([f.alt for f in frames], dtype=np.float64)

    x, y = latlon_to_local_xy(lat, lon)
    z = alt - alt[0]
    gt_xyz = np.column_stack([x, y, z])

    return SequenceData(sequence_id=drive_root.name, frames=frames, gt_xyz=gt_xyz)


def iter_sequences(dataset_root: Path, max_frames: int | None = None) -> Iterable[SequenceData]:
    for drive in discover_drive_roots(dataset_root):
        yield load_kitti_sequence(drive, max_frames=max_frames)
