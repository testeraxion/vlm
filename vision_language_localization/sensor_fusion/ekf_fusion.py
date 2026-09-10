from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_language_localization.data.kitti_loader import SequenceData, latlon_to_local_xy
from vision_language_localization.sensor_fusion.alignment import apply_similarity, estimate_registration
from vision_language_localization.slam.mock_slam import SlamOutput


@dataclass
class FusionOutput:
    fused_xyz: np.ndarray


@dataclass
class EkfFusionConfig:
    """Tunable noise parameters for the Extended Kalman Filter fusion."""

    gps_noise_m: float = 1.5
    slam_noise_m: float = 1.0
    dead_reckoning_noise_mps: float = 3.0
    process_noise_accel_mps2: float = 2.0
    initial_position_noise_m: float = 10.0
    initial_velocity_noise_mps: float = 5.0
    lost_tracking_noise_scale: float = 20.0
    gate_chi2: float = 0.0
    align_slam_to_gps: bool = True
    registration_window: int = 50
    min_registration_frames: int = 10


class LightweightFusion:
    """Legacy weighted-blend baseline retained for comparison (``fusion_method=blend``)."""

    def __init__(self, alpha_visual: float = 0.65) -> None:
        if not 0.0 <= alpha_visual <= 1.0:
            raise ValueError("alpha_visual must be in [0, 1]")
        self.alpha_visual = alpha_visual

    def run(self, seq: SequenceData, slam: SlamOutput) -> FusionOutput:
        gt = seq.gt_xyz
        n = min(len(gt), len(slam.est_xyz), len(seq.frames))
        vel_integrated = np.zeros_like(gt[:n])

        for i in range(1, n):
            dt = max(seq.frames[i].timestamp_s - seq.frames[i - 1].timestamp_s, 1e-3)
            heading = seq.frames[i].yaw
            speed = seq.frames[i].speed_mps
            dx = speed * np.cos(heading) * dt
            dy = speed * np.sin(heading) * dt
            vel_integrated[i] = vel_integrated[i - 1] + np.array([dx, dy, 0.0])

        fused = self.alpha_visual * slam.est_xyz[:n] + (1.0 - self.alpha_visual) * vel_integrated
        return FusionOutput(fused_xyz=fused)


def compute_aligned_slam(seq: SequenceData, slam: SlamOutput, config: EkfFusionConfig | None = None) -> FusionOutput:
    """Apply Umeyama registration to SLAM trajectory without EKF fusion."""
    cfg = config or EkfFusionConfig()
    n = min(len(seq.frames), len(slam.est_xyz))
    if n < 2:
        raise ValueError("Need at least 2 frames")

    frames = seq.frames[:n]
    lat = np.array([f.lat for f in frames], dtype=float)
    lon = np.array([f.lon for f in frames], dtype=float)
    alt = np.array([f.alt for f in frames], dtype=float)
    gps_x, gps_y = latlon_to_local_xy(lat, lon)
    gps_xyz = np.column_stack([gps_x, gps_y, alt - alt[0]])

    est_xyz = slam.est_xyz[:n].copy()
    tracking_quality = slam.tracking_quality[:n]

    registration = estimate_registration(
        est_xyz, gps_xyz, tracking_quality,
        window=cfg.registration_window,
        min_frames=cfg.min_registration_frames,
    )
    if registration is not None:
        est_xyz = apply_similarity(est_xyz, registration)

    return FusionOutput(fused_xyz=est_xyz)


def compute_gps_only(seq: SequenceData) -> FusionOutput:
    """Return GPS trajectory as the fused estimate."""
    n = len(seq.frames)
    frames = seq.frames[:n]
    lat = np.array([f.lat for f in frames], dtype=float)
    lon = np.array([f.lon for f in frames], dtype=float)
    alt = np.array([f.alt for f in frames], dtype=float)
    gps_x, gps_y = latlon_to_local_xy(lat, lon)
    gps_xyz = np.column_stack([gps_x, gps_y, alt - alt[0]])
    return FusionOutput(fused_xyz=gps_xyz)


class EkfFusion:
    """Extended Kalman Filter fusing Visual SLAM, GPS, and IMU-derived dead-reckoning.

    State vector: ``[px, py, pz, vx, vy, vz]`` with a constant-velocity motion model.

    Measurements (all applied as sequential updates each frame):

    - **GPS position** from OXTS lat/lon/alt, in the same local ENU frame as the
      ground-truth trajectory.
    - **SLAM position** with *adaptive* measurement noise: the filter trusts the
      visual estimate less as tracking quality degrades (and very little when
      tracking is lost).
    - **Dead-reckoning velocity** from OXTS speed + yaw, which couples the IMU
      information into the state.

    Before filtering, the SLAM trajectory is **registered to the GPS frame** with
    a similarity (Umeyama) transform estimated from the first
    ``registration_window`` frames where tracking quality is adequate
    (``align_slam_to_gps``). External SLAM exports are frequently written in an
    arbitrary visual frame; registration makes the visual and global frames
    consistent so the filter can genuinely fuse both sources.

    An optional Mahalanobis innovation gate (``gate_chi2 > 0``) rejects outlier
    measurements, at the cost of a lockout risk on short, low-noise windows.
    """

    def __init__(self, config: EkfFusionConfig | None = None) -> None:
        self.config = config if config is not None else EkfFusionConfig()

    @staticmethod
    def _motion_matrix(dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _process_covariance(self, dt: float) -> np.ndarray:
        q = self.config.process_noise_accel_mps2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        block = np.array(
            [
                [dt4 / 4.0, dt3 / 2.0],
                [dt3 / 2.0, dt2],
            ]
        )
        Q = np.kron(np.eye(3, dtype=float), block) * q
        return Q

    @staticmethod
    def _update(
        state: np.ndarray,
        cov: np.ndarray,
        z: np.ndarray,
        r: np.ndarray,
        h: np.ndarray,
        gate_chi2: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sequential EKF update with optional Mahalanobis innovation gating.

        When ``gate_chi2 > 0``, a measurement whose squared Mahalanobis
        distance exceeds the threshold is rejected. This keeps the filter robust
        to outlier measurements, e.g. a SLAM trajectory that is not registered
        to the GPS frame.
        """
        innovation = z - h @ state
        s = h @ cov @ h.T + r
        if gate_chi2 > 0.0:
            mahalanobis_sq = float(innovation @ np.linalg.solve(s, innovation))
            if mahalanobis_sq > gate_chi2:
                return state, cov
        gain = np.linalg.solve(s, h @ cov.T).T
        state = state + gain @ innovation
        cov = (np.eye(state.shape[0]) - gain @ h) @ cov
        cov = (cov + cov.T) / 2.0
        return state, cov

    def run(self, seq: SequenceData, slam: SlamOutput) -> FusionOutput:
        n = min(len(seq.frames), len(slam.est_xyz))
        if n < 2:
            raise ValueError("Need at least 2 frames for EKF fusion")

        frames = seq.frames[:n]
        times = np.array([f.timestamp_s for f in frames], dtype=float)
        lat = np.array([f.lat for f in frames], dtype=float)
        lon = np.array([f.lon for f in frames], dtype=float)
        alt = np.array([f.alt for f in frames], dtype=float)
        speed = np.array([f.speed_mps for f in frames], dtype=float)
        yaw = np.array([f.yaw for f in frames], dtype=float)

        gps_x, gps_y = latlon_to_local_xy(lat, lon)
        gps_xyz = np.column_stack([gps_x, gps_y, alt - alt[0]])

        est_xyz = slam.est_xyz[:n].copy()
        tracking_quality = slam.tracking_quality[:n]

        if self.config.align_slam_to_gps:
            registration = estimate_registration(
                est_xyz,
                gps_xyz,
                tracking_quality,
                window=self.config.registration_window,
                min_frames=self.config.min_registration_frames,
            )
            if registration is not None:
                est_xyz = apply_similarity(est_xyz, registration)

        state = np.zeros(6, dtype=float)
        state[:3] = gps_xyz[0]
        cov = np.diag(
            [self.config.initial_position_noise_m] * 3
            + [self.config.initial_velocity_noise_mps] * 3
        )

        h_pos = np.hstack([np.eye(3), np.zeros((3, 3))])
        h_vel = np.hstack([np.zeros((3, 3)), np.eye(3)])

        fused = np.empty_like(gps_xyz)
        fused[0] = gps_xyz[0]

        for k in range(1, n):
            dt = max(float(times[k] - times[k - 1]), 1e-3)

            state = self._motion_matrix(dt) @ state
            cov = self._motion_matrix(dt) @ cov @ self._motion_matrix(dt).T + self._process_covariance(dt)

            r_gps = np.eye(3) * (self.config.gps_noise_m**2)
            state, cov = self._update(
                state, cov, gps_xyz[k], r_gps, h_pos, gate_chi2=self.config.gate_chi2
            )

            quality = tracking_quality[k]
            if quality < 0.25:
                scale = self.config.lost_tracking_noise_scale
            elif quality < 1.0:
                scale = 1.0 / max(quality, 0.05)
            else:
                scale = 1.0
            r_slam = np.eye(3) * (self.config.slam_noise_m * scale) ** 2
            state, cov = self._update(
                state, cov, est_xyz[k], r_slam, h_pos, gate_chi2=self.config.gate_chi2
            )

            r_dr = np.eye(3) * (self.config.dead_reckoning_noise_mps**2)
            velocity = np.array([speed[k] * np.cos(yaw[k]), speed[k] * np.sin(yaw[k]), 0.0])
            state, cov = self._update(state, cov, velocity, r_dr, h_vel)

            fused[k] = state[:3]

        return FusionOutput(fused_xyz=fused)
