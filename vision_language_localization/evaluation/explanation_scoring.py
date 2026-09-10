from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvidenceAlignmentResult:
    supported_claim_count: int
    unsupported_claim_count: int
    evidence_alignment_score: float
    flagged_claims: list[str]


def score_explanation_against_evidence(
    explanation: str,
    mean_reprojection_error: float,
    mean_feature_count: float,
    drift_final_m: float,
    violation_count: int,
) -> EvidenceAlignmentResult:
    text = explanation.lower()

    checks: list[tuple[str, bool, bool]] = [
        (
            "low features / weak visual landmarks",
            ("feature" in text) or ("landmark" in text),
            mean_feature_count < 500,
        ),
        (
            "high reprojection error / geometric inconsistency",
            ("reprojection" in text) or ("geometric" in text),
            mean_reprojection_error > 1.8,
        ),
        (
            "trajectory drift",
            "drift" in text,
            drift_final_m > 2.0,
        ),
        (
            "runtime safety violations",
            ("violation" in text) or ("unsafe" in text),
            violation_count > 0,
        ),
    ]

    supported = 0
    unsupported = 0
    flagged: list[str] = []

    for label, mentioned, evidence_true in checks:
        if not mentioned:
            continue
        if evidence_true:
            supported += 1
        else:
            unsupported += 1
            flagged.append(label)

    total = supported + unsupported
    score = 1.0 if total == 0 else supported / total

    return EvidenceAlignmentResult(
        supported_claim_count=supported,
        unsupported_claim_count=unsupported,
        evidence_alignment_score=float(score),
        flagged_claims=flagged,
    )
