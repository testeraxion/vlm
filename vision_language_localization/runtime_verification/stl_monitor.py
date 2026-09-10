from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RuntimeVerificationResult:
    stl_robustness: float
    violation_count: int
    total_violation_duration_s: float
    max_consecutive_lost_s: float
    property_summary: dict[str, bool]


def evaluate_runtime_properties(
    position_error_m: np.ndarray,
    tracking_ok: np.ndarray,
    dt_s: float,
    max_position_error_m: float,
    max_lost_duration_s: float,
    recovery_window_s: float,
) -> RuntimeVerificationResult:
    if len(position_error_m) != len(tracking_ok):
        raise ValueError("Error and tracking arrays must have the same length")

    error_robustness = max_position_error_m - position_error_m
    always_pos_ok = bool(np.all(error_robustness > 0.0))

    lost = ~tracking_ok
    violation_count = int(np.sum(lost))
    total_violation_duration_s = float(violation_count * dt_s)

    max_streak = 0
    current = 0
    for flag in lost:
        if flag:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    max_consecutive_lost_s = float(max_streak * dt_s)
    never_lost_too_long = max_consecutive_lost_s < max_lost_duration_s

    recovered_flags = []
    recovery_window_frames = max(1, int(round(recovery_window_s / dt_s)))
    for i in range(len(tracking_ok)):
        if not tracking_ok[i]:
            end = min(len(tracking_ok), i + recovery_window_frames)
            recovered_flags.append(bool(np.any(tracking_ok[i:end])))
    eventually_recovered = bool(all(recovered_flags)) if recovered_flags else True

    stl_robustness = float(
        min(
            float(np.min(error_robustness)),
            max_lost_duration_s - max_consecutive_lost_s,
            1.0 if eventually_recovered else -1.0,
        )
    )

    return RuntimeVerificationResult(
        stl_robustness=stl_robustness,
        violation_count=violation_count,
        total_violation_duration_s=total_violation_duration_s,
        max_consecutive_lost_s=max_consecutive_lost_s,
        property_summary={
            "always_position_error_lt_threshold": always_pos_ok,
            "never_tracking_lost_too_long": never_lost_too_long,
            "eventually_tracking_recovers": eventually_recovered,
        },
    )
