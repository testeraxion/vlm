from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from vision_language_localization.evaluation.correlation import correlate_vlm_with_safety
from vision_language_localization.evaluation.vlm_evaluation import (
    aggregate_metrics,
    compute_explanation_metrics,
    load_annotations,
)


def _optional_float(value) -> float | None:
    return float(value) if value is not None else None


def evaluate_report(
    report_path: Path,
    annotations_path: Path | None = None,
    output_root: Path | None = None,
    n_perm: int = 2000,
    seed: int = 7,
) -> Path:
    """Evaluate a pipeline report against expert annotations and correlation analysis.

    Produces ``evaluation_report.json`` under ``output_root`` containing:

    - per-sequence explanation metrics (claim/hazard precision, recall, F1)
    - hallucination rate
    - consistency and latency when recorded by the pipeline
    - pairwise VLM-signal vs safety-metric correlations with p-values (RQ3)
    """
    report_path = Path(report_path)
    output_root = Path(output_root) if output_root is not None else report_path.parent
    records = json.loads(report_path.read_text(encoding="utf-8"))

    annotations: dict[str, object] = {}
    if annotations_path is not None:
        ann_path = Path(annotations_path)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {ann_path}")
        annotations = {a.sequence_id: a for a in load_annotations(ann_path)}

    per_sequence: list[dict] = []
    for record in records:
        sequence_id = record["sequence_id"]
        annotation = annotations.get(sequence_id)
        metrics = compute_explanation_metrics(
            explanation=record["vlm"].get("explanation", ""),
            hazards=record["vlm"].get("hazards", []),
            annotation=annotation,
        )

        vlm = record["vlm"]
        entry = {
            "sequence_id": sequence_id,
            "backends": record.get("backends", {}),
            "vlm_signals": {
                "hazard_count": float(len(vlm.get("hazards", []))),
                "hallucination_risk": _optional_float(vlm.get("hallucination_risk")),
                "consistency_score": _optional_float(vlm.get("consistency_score")),
                "evidence_alignment": _optional_float(
                    record.get("explanation_evidence", {}).get("evidence_alignment_score")
                ),
            },
            "safety_metrics": {
                "ate_rmse": _optional_float(record["slam_metrics"].get("ate_rmse")),
                "rpe_mean": _optional_float(record["slam_metrics"].get("rpe_mean")),
                "drift_final_m": _optional_float(record["slam_metrics"].get("drift_final_m")),
                "violation_count": _optional_float(record["runtime_verification"].get("violation_count")),
                "stl_robustness": _optional_float(record["runtime_verification"].get("stl_robustness")),
            },
            "explanation_metrics": metrics.as_dict() if metrics is not None else None,
            "consistency": vlm.get("consistency"),
            "latency": vlm.get("latency"),
            "annotated": annotation is not None,
        }
        per_sequence.append(entry)

    correlation = correlate_vlm_with_safety(records, n_perm=n_perm, seed=seed)

    explained_entries = [e for e in per_sequence if e["explanation_metrics"] is not None]
    aggregates = aggregate_metrics([e["explanation_metrics"] for e in explained_entries])

    consistency_values = [
        float(e["consistency"]["text_similarity"]) for e in per_sequence if e.get("consistency") is not None
    ]
    latency_values = [float(e["latency"]["mean_s"]) for e in per_sequence if e.get("latency") is not None]

    summary: dict = {
        "num_sequences": len(records),
        "num_annotated": len(explained_entries),
        "explanation_metrics": aggregates,
        "consistency": (
            {
                "mean_text_similarity": round(statistics.mean(consistency_values), 4) if consistency_values else None,
                "n": len(consistency_values),
            }
            if consistency_values
            else {"mean_text_similarity": None, "n": 0}
        ),
        "latency": (
            {
                "mean_s": round(statistics.mean(latency_values), 4) if latency_values else None,
                "n": len(latency_values),
            }
            if latency_values
            else {"mean_s": None, "n": 0}
        ),
    }

    evaluation = {
        "report_source": str(report_path),
        "annotations_source": str(annotations_path) if annotations_path is not None else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "correlation_rq3": correlation,
        "per_sequence": per_sequence,
    }

    out_file = output_root / "evaluation_report.json"
    output_root.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return out_file
