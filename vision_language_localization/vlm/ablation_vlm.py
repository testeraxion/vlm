"""VLM ablation backends that test sensitivity to input modality.

Ablation conditions:
- image_only: VLM receives ONLY the scene image (no numeric metrics)
- metrics_only: VLM receives ONLY numeric metrics (no image)
- combined: VLM receives both image and metrics (baseline)
- contradictory: VLM receives image but metrics are intentionally perturbed
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class VLMBackend(Protocol):
    def explain(
        self,
        ate_rmse: float,
        drift_final_m: float,
        mean_reprojection_error: float,
        mean_feature_count: float,
        violation_count: int,
        scene_image_path: Path | None = None,
    ): ...


@dataclass
class VLMExplanationResult:
    explanation: str
    hazards: list[str]
    hallucination_risk: float
    consistency_score: float
    scene_caption: str | None = None


class ImageOnlyVLM:
    """VLM receives ONLY the scene image. Metrics are suppressed to test
    whether the VLM can reason from visual evidence alone."""

    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend

    def explain(
        self,
        ate_rmse: float = 0.0,
        drift_final_m: float = 0.0,
        mean_reprojection_error: float = 0.0,
        mean_feature_count: float = 0.0,
        violation_count: int = 0,
        scene_image_path: Path | None = None,
    ) -> VLMExplanationResult:
        return self.backend.explain(
            ate_rmse=0.0,
            drift_final_m=0.0,
            mean_reprojection_error=0.0,
            mean_feature_count=0.0,
            violation_count=0,
            scene_image_path=scene_image_path,
        )


class MetricsOnlyVLM:
    """VLM receives ONLY numeric metrics. Image is suppressed to test
    whether the VLM can reason from metrics alone without visual grounding."""

    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend

    def explain(
        self,
        ate_rmse: float = 0.0,
        drift_final_m: float = 0.0,
        mean_reprojection_error: float = 0.0,
        mean_feature_count: float = 0.0,
        violation_count: int = 0,
        scene_image_path: Path | None = None,
    ) -> VLMExplanationResult:
        return self.backend.explain(
            ate_rmse=ate_rmse,
            drift_final_m=drift_final_m,
            mean_reprojection_error=mean_reprojection_error,
            mean_feature_count=mean_feature_count,
            violation_count=violation_count,
            scene_image_path=None,
        )


class CombinedVLM:
    """VLM receives both image and metrics (baseline)."""

    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend

    def explain(
        self,
        ate_rmse: float = 0.0,
        drift_final_m: float = 0.0,
        mean_reprojection_error: float = 0.0,
        mean_feature_count: float = 0.0,
        violation_count: int = 0,
        scene_image_path: Path | None = None,
    ) -> VLMExplanationResult:
        return self.backend.explain(
            ate_rmse=ate_rmse,
            drift_final_m=drift_final_m,
            mean_reprojection_error=mean_reprojection_error,
            mean_feature_count=mean_feature_count,
            violation_count=violation_count,
            scene_image_path=scene_image_path,
        )


class ContradictoryVLM:
    """VLM receives the scene image but metrics are intentionally perturbed
    to test whether the VLM trusts visual evidence over contradictory numbers.

    The perturbation inflates ATE by 10x and sets violation_count to 0
    regardless of actual violations."""

    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend

    def explain(
        self,
        ate_rmse: float = 0.0,
        drift_final_m: float = 0.0,
        mean_reprojection_error: float = 0.0,
        mean_feature_count: float = 0.0,
        violation_count: int = 0,
        scene_image_path: Path | None = None,
    ) -> VLMExplanationResult:
        # Perturb metrics: inflate ATE, hide violations
        fake_ate = ate_rmse * 10.0
        fake_drift = drift_final_m * 5.0
        fake_violations = 0

        result = self.backend.explain(
            ate_rmse=fake_ate,
            drift_final_m=fake_drift,
            mean_reprojection_error=mean_reprojection_error,
            mean_feature_count=mean_feature_count,
            violation_count=fake_violations,
            scene_image_path=scene_image_path,
        )
        return result
