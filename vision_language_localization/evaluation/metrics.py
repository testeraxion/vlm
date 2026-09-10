from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryMetrics:
    ate_rmse: float
    rpe_mean: float
    drift_final_m: float


def compute_trajectory_metrics(gt_xyz: np.ndarray, est_xyz: np.ndarray) -> TrajectoryMetrics:
    if gt_xyz.shape != est_xyz.shape:
        raise ValueError("gt and est must have the same shape")

    errors = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    ate_rmse = float(np.sqrt(np.mean(np.square(errors))))

    gt_delta = np.diff(gt_xyz, axis=0)
    est_delta = np.diff(est_xyz, axis=0)
    rpe = np.linalg.norm(est_delta - gt_delta, axis=1)
    rpe_mean = float(np.mean(rpe)) if len(rpe) else 0.0

    drift_final_m = float(np.linalg.norm((est_xyz - gt_xyz)[-1]))

    return TrajectoryMetrics(ate_rmse=ate_rmse, rpe_mean=rpe_mean, drift_final_m=drift_final_m)
