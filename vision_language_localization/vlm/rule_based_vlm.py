from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VLMExplanationResult:
    explanation: str
    hazards: list[str]
    hallucination_risk: float
    consistency_score: float
    scene_caption: str | None = None


class RuleBasedVLM:
    """Baseline semantic explainer.

    This placeholder provides deterministic explanations so that evaluation,
    reliability studies, and dashboards can be built before integrating a real VLM.
    """

    def __init__(self, consistency_runs: int = 5) -> None:
        self.consistency_runs = max(1, consistency_runs)

    def explain(
        self,
        ate_rmse: float,
        drift_final_m: float,
        mean_reprojection_error: float,
        mean_feature_count: float,
        violation_count: int,
        scene_image_path: Path | None = None,
    ) -> VLMExplanationResult:
        reasons: list[str] = []
        hazards: list[str] = []

        if mean_feature_count < 500:
            reasons.append("low stable feature density likely reduced visual matching quality")
        if mean_reprojection_error > 1.8:
            reasons.append("elevated reprojection residuals suggest geometric inconsistency")
        if drift_final_m > 2.0:
            reasons.append("accumulated trajectory drift indicates long-horizon localization instability")
        if violation_count > 0:
            reasons.append("runtime safety constraints were violated during parts of the sequence")

        if not reasons:
            reasons.append("scene and odometry signals appear consistent with stable localization")

        if ate_rmse > 1.5:
            hazards.append("localization degradation risk")
        if violation_count > 0:
            hazards.append("safety constraint violation")
        if mean_feature_count < 350:
            hazards.append("weak visual observability")

        if not hazards:
            hazards.append("no immediate hazard detected")

        explanation = (
            "Localization analysis: "
            + "; ".join(reasons)
            + ". Suggested operator action: monitor confidence and switch to conservative control when risk rises."
        )

        hallucination_risk = float(np.clip(0.15 + 0.2 * (mean_reprojection_error > 2.2), 0.0, 1.0))
        consistency_score = 1.0 - hallucination_risk * 0.4

        return VLMExplanationResult(
            explanation=explanation,
            hazards=hazards,
            hallucination_risk=hallucination_risk,
            consistency_score=consistency_score,
            scene_caption=None,
        )
