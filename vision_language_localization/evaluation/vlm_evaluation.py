from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

CLAIM_TAXONOMY: dict[str, list[str]] = {
    "low_feature_support": ["feature", "landmark", "texture", "repetitive", "sparse"],
    "geometric_inconsistency": ["reprojection", "geometric", "parallax", "inconsistent", "calibration"],
    "trajectory_drift": ["drift", "accumulat", "long-horizon", "position error"],
    "safety_violation": ["violat", "unsafe", "constraint", "boundary", "threshold"],
    "environmental_obstruction": ["occlus", "tree", "shadow", "fog", "rain", "obstruct", "sunlight", "glare", "weather"],
    "motion_blur": ["blur", "motion", "shake", "vibration"],
    "gps_degradation": ["gps", "multipath", "signal loss", "satellite", "positioning"],
    "illumination_change": ["lighting", "illumination", "exposure", "bright", "dark", "night", "tunnel"],
    "road_geometry": ["curve", "intersection", "turn", "lane", "road layout", "junction"],
    "dynamic_objects": ["pedestrian", "vehicle", "cyclist", "traffic", "crowd"],
}

HAZARD_TAXONOMY: dict[str, list[str]] = {
    "localization_degradation": ["localization degradation", "degradation", "position error"],
    "safety_violation": ["violation", "unsafe", "constraint"],
    "visual_ambiguity": ["ambigu", "weak visual", "occlusion", "observability", "obscur", "feature"],
    "dynamic_obstacle": ["car", "vehicle", "pedestrian", "truck", "traffic", "cyclist"],
    "model_reported_risk": ["risk", "hazard"],
    "gps_issue": ["gps", "multipath", "signal", "positioning"],
    "environmental_hazard": ["weather", "fog", "rain", "glare", "lighting", "illumination"],
    "none": ["no immediate hazard"],
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "with", "by", "as", "it", "its", "this",
    "that", "these", "those", "from", "due", "may", "can", "could", "might", "likely",
    "not", "no", "also", "there", "which", "indicate", "indicates", "indicated", "1",
    "2", "3", "4", "scene", "shows", "explain", "causes", "include", "safety", "hazards",
    "uncertainty", "localization", "presence", "possible", "result",
}


def _normalize_item(text: str, taxonomy: dict[str, list[str]]) -> str | None:
    """Map a free-text item to the most specific taxonomy label (longest keyword match)."""
    best_label: str | None = None
    best_len = -1
    for label, keywords in taxonomy.items():
        for keyword in keywords:
            if keyword in text and len(keyword) > best_len:
                best_len = len(keyword)
                best_label = label
    return best_label


def detect_claims(explanation: str) -> set[str]:
    """Detect which claim categories a VLM explanation mentions."""
    text = explanation.lower()
    return {label for label, keywords in CLAIM_TAXONOMY.items() if any(kw in text for kw in keywords)}


def normalize_hazards(hazards: list[str]) -> set[str]:
    """Map raw VLM hazard strings to canonical taxonomy labels."""
    normalized: set[str] = set()
    for hazard in hazards:
        label = _normalize_item(hazard.lower(), HAZARD_TAXONOMY)
        if label is not None:
            normalized.add(label)
    return normalized


@dataclass
class ExplanationAnnotation:
    sequence_id: str
    expected_claims: list[str]
    expected_hazards: list[str]


def load_annotations(path: Path) -> list[ExplanationAnnotation]:
    """Load expert annotations from a JSON file.

    Expected schema::

        {
          "sequences": [
            {
              "sequence_id": "2011_09_26_drive_0009_sync",
              "expected_claims": ["trajectory_drift", "safety_violation"],
              "expected_hazards": ["localization_degradation", "safety_violation"]
            }
          ]
        }

    Labels are drawn from :data:`CLAIM_TAXONOMY` / :data:`HAZARD_TAXONOMY`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sequences = data.get("sequences", data if isinstance(data, list) else [])
    return [
        ExplanationAnnotation(
            sequence_id=str(item["sequence_id"]),
            expected_claims=list(item.get("expected_claims", [])),
            expected_hazards=list(item.get("expected_hazards", [])),
        )
        for item in sequences
    ]


@dataclass
class ExplanationMetrics:
    sequence_id: str
    claim_precision: float
    claim_recall: float
    claim_f1: float
    hazard_precision: float
    hazard_recall: float
    hazard_f1: float
    hallucination_rate: float
    model_claims: list[str]
    expected_claims: list[str]
    correct_claims: list[str]
    model_hazards: list[str]
    expected_hazards: list[str]
    hallucinated_hazards: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_explanation_metrics(
    explanation: str,
    hazards: list[str],
    annotation: ExplanationAnnotation | None,
) -> ExplanationMetrics | None:
    """Compute claim accuracy, hazard precision/recall/F1, and hallucination rate.

    Returns ``None`` when no expert annotation is available for the sequence.
    """
    if annotation is None:
        return None

    expected_claims = set(annotation.expected_claims)
    model_claims = detect_claims(explanation)
    correct_claims = model_claims & expected_claims

    claim_precision = len(correct_claims) / len(model_claims) if model_claims else 0.0
    claim_recall = len(correct_claims) / len(expected_claims) if expected_claims else 0.0

    expected_hazards = set(annotation.expected_hazards)
    model_hazards = normalize_hazards(hazards)
    correct_hazards = model_hazards & expected_hazards

    hazard_precision = len(correct_hazards) / len(model_hazards) if model_hazards else 0.0
    hazard_recall = len(correct_hazards) / len(expected_hazards) if expected_hazards else 0.0

    hallucinated = model_hazards - expected_hazards
    hallucination_rate = len(hallucinated) / len(model_hazards) if model_hazards else 0.0

    return ExplanationMetrics(
        sequence_id=annotation.sequence_id,
        claim_precision=round(claim_precision, 4),
        claim_recall=round(claim_recall, 4),
        claim_f1=round(_f1(claim_precision, claim_recall), 4),
        hazard_precision=round(hazard_precision, 4),
        hazard_recall=round(hazard_recall, 4),
        hazard_f1=round(_f1(hazard_precision, hazard_recall), 4),
        hallucination_rate=round(hallucination_rate, 4),
        model_claims=sorted(model_claims),
        expected_claims=sorted(expected_claims),
        correct_claims=sorted(correct_claims),
        model_hazards=sorted(model_hazards),
        expected_hazards=sorted(expected_hazards),
        hallucinated_hazards=sorted(hallucinated),
    )


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def compute_consistency(explanations: list[str]) -> dict:
    """Consistency across repeated runs of the same scene/prompt.

    Reports the mean pairwise Jaccard similarity of the explanation vocabulary
    and of the detected claim sets, plus the sample standard deviation of the
    text-similarity distribution.
    """
    runs = len(explanations)
    if runs < 2:
        return {
            "runs": runs,
            "text_similarity": 1.0,
            "claim_agreement": 1.0,
            "text_std": 0.0,
            "note": "consistency requires multiple runs (vlm_consistency_runs > 1)",
        }

    token_sets = [_tokens(text) for text in explanations]
    claim_sets = [detect_claims(text) for text in explanations]

    pairwise_text: list[float] = []
    pairwise_claims: list[float] = []
    for i in range(runs):
        for j in range(i + 1, runs):
            pairwise_text.append(jaccard(token_sets[i], token_sets[j]))
            pairwise_claims.append(jaccard(claim_sets[i], claim_sets[j]))

    mean_text = float(sum(pairwise_text) / len(pairwise_text))
    mean_claims = float(sum(pairwise_claims) / len(pairwise_claims))
    std_text = float(np_std(pairwise_text))

    return {
        "runs": runs,
        "text_similarity": round(mean_text, 4),
        "claim_agreement": round(mean_claims, 4),
        "text_std": round(std_text, 4),
    }


def compute_latency_stats(latencies: list[float]) -> dict:
    if not latencies:
        return {"runs": 0, "mean_s": 0.0, "std_s": 0.0}
    mean = float(sum(latencies) / len(latencies))
    std = np_std(latencies)
    return {"runs": len(latencies), "mean_s": round(mean, 4), "std_s": round(std, 4)}


def np_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return float(var ** 0.5)


def aggregate_metrics(per_sequence: list[dict]) -> dict:
    """Aggregate mean/std over the evaluated sequences."""
    keys = [
        "claim_precision",
        "claim_recall",
        "claim_f1",
        "hazard_precision",
        "hazard_recall",
        "hazard_f1",
        "hallucination_rate",
    ]
    aggregates: dict = {}
    for key in keys:
        values = [float(row[key]) for row in per_sequence if row.get(key) is not None]
        if values:
            aggregates[key] = {
                "mean": round(sum(values) / len(values), 4),
                "std": round(np_std(values), 4),
                "n": len(values),
            }
    return aggregates
