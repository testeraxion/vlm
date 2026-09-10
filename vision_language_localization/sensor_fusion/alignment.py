from __future__ import annotations

import numpy as np

Transformed = tuple[float, np.ndarray, np.ndarray]


def similarity_transform_from_correspondences(
    src: np.ndarray,
    dst: np.ndarray,
) -> Transformed:
    """Umeyama similarity transform mapping ``src -> dst``.

    Solves ``min || dst - (s * R @ src + t) ||`` for a scale ``s``, rotation
    ``R`` and translation ``t`` from two matching point sets (``n x 3``).
    """
    if src.shape[0] != dst.shape[0] or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Correspondence sets must be matching n x 3 arrays")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    centered_src = src - mu_src
    centered_dst = dst - mu_dst

    covariance = centered_src.T @ centered_dst
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, d])
    rotation = vt.T @ diag @ u.T

    denom = float(np.sum(centered_src * centered_src))
    scale = float(np.sum(centered_dst * (centered_src @ rotation.T)) / denom) if denom > 0.0 else 1.0
    translation = mu_dst - scale * (rotation @ mu_src)

    return scale, rotation, translation


def apply_similarity(points: np.ndarray, transform: Transformed) -> np.ndarray:
    scale, rotation, translation = transform
    return scale * (points @ rotation.T) + translation


def estimate_registration(slam_xyz: np.ndarray, gps_xyz: np.ndarray, quality: np.ndarray, window: int, min_frames: int):
    """Estimate a SLAM->GPS transform from the first frames where tracking quality is adequate.

    Returns the ``(scale, rotation, translation)`` transform or ``None`` when
    too few reliable correspondences are available.
    """
    if len(slam_xyz) < 2:
        return None
    window = max(2, min(int(window), len(slam_xyz)))
    selected = np.arange(window)
    if quality is not None:
        selected = selected[quality[:window] >= 0.5]
    if len(selected) < min_frames:
        return None
    return similarity_transform_from_correspondences(slam_xyz[selected], gps_xyz[selected])
