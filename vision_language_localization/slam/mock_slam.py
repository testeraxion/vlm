from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_language_localization.data.kitti_loader import SequenceData


@dataclass
class SlamOutput:
    est_xyz: np.ndarray
    tracking_ok: np.ndarray
    tracking_quality: np.ndarray
    reprojection_error: np.ndarray
    feature_count: np.ndarray


class MockVisualSlam:
    """Deterministic baseline that simulates typical SLAM failure modes.

    This is a research scaffold. Replace this module with ORB-SLAM3/OpenVSLAM logs
    when moving from baseline experiments to full experiments.
    """

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)

    def run(self, seq: SequenceData) -> SlamOutput:
        n = len(seq.frames)
        if n < 2:
            raise ValueError("Need at least 2 frames for SLAM simulation")

        gt = seq.gt_xyz
        speed = np.array([f.speed_mps for f in seq.frames])
        yaw = np.array([f.yaw for f in seq.frames])

        yaw_rate = np.abs(np.gradient(yaw))
        stress = 0.4 * (speed / (speed.max() + 1e-6)) + 0.6 * (yaw_rate / (yaw_rate.max() + 1e-6))

        noise_scale = 0.15 + 0.9 * stress
        drift = np.cumsum(self.rng.normal(0.0, 0.02, size=(n, 3)), axis=0)
        white_noise = self.rng.normal(0.0, 1.0, size=(n, 3)) * noise_scale[:, None] * 0.15

        est = gt + drift + white_noise

        tracking_quality = np.clip(1.0 - stress, 0.0, 1.0)
        tracking_ok = tracking_quality > 0.25
        reprojection_error = 0.4 + 3.2 * (1.0 - tracking_quality)
        feature_count = (1400 * tracking_quality + 80).astype(np.int32)

        return SlamOutput(
            est_xyz=est,
            tracking_ok=tracking_ok,
            tracking_quality=tracking_quality,
            reprojection_error=reprojection_error,
            feature_count=feature_count,
        )
