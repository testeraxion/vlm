from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_language_localization.slam.mock_slam import SlamOutput


def _load_estimated_xyz(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[list[float]] = []
    for line in lines:
        parts = [float(x) for x in line.split()]
        rows.append(parts)

    arr = np.array(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Invalid trajectory matrix shape in {path}")

    if arr.shape[1] == 12:
        # KITTI pose row: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
        xyz = arr[:, [3, 7, 11]]
    elif arr.shape[1] >= 3:
        xyz = arr[:, :3]
    else:
        raise ValueError(f"Unsupported trajectory format with {arr.shape[1]} columns in {path}")

    return xyz


def load_external_slam_output(
    trajectory_file: Path,
    gt_xyz: np.ndarray,
) -> SlamOutput:
    est_xyz = _load_estimated_xyz(trajectory_file)

    n = min(len(est_xyz), len(gt_xyz))
    if n < 2:
        raise ValueError("Trajectory must contain at least 2 frames")

    est = est_xyz[:n]
    gt = gt_xyz[:n]

    # Compute tracking quality from local trajectory consistency (smoothness),
    # NOT from comparison with GPS ground truth, because the raw SLAM trajectory
    # is in an arbitrary coordinate frame and cannot be directly compared.
    frame_displacements = np.linalg.norm(np.diff(est, axis=0), axis=1)
    # Causal rolling median over the last W=50 frames (no future information)
    W = 50
    eps = 0.05  # prevent division by zero when vehicle is stationary
    n_disp = len(frame_displacements)
    tracking_quality = np.ones(n, dtype=np.float64)
    for i in range(n_disp):
        start = max(0, i - W + 1)
        window = frame_displacements[start:i + 1]
        median_disp = float(np.median(window))
        deviation = abs(frame_displacements[i] - median_disp) / (median_disp + eps)
        tracking_quality[i + 1] = float(np.clip(1.0 - deviation / 3.0, 0.0, 1.0))
    tracking_ok = tracking_quality > 0.25

    # Proxy metrics when external logs are not available.
    reprojection_error = 0.6 + 2.8 * (1.0 - tracking_quality)
    feature_count = (1400 * tracking_quality + 120).astype(np.int32)

    return SlamOutput(
        est_xyz=est,
        tracking_ok=tracking_ok,
        tracking_quality=tracking_quality,
        reprojection_error=reprojection_error,
        feature_count=feature_count,
    )
