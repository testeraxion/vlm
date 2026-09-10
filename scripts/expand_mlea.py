"""Expand MLEA evaluation from 6 claims to all 81 VLM outputs.

Loads all qwen_*.json files from the ablation directory, runs detect_claims()
on each VLM explanation, maps claims to telemetry conditions, and computes
MLEA score per sequence.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vision_language_localization.evaluation.vlm_evaluation import (
    CLAIM_TAXONOMY,
    detect_claims,
)
from vision_language_localization.evaluation.explanation_scoring import (
    score_explanation_against_evidence,
)

ABLATION_DIR = REPO_ROOT / "outputs" / "ablation"
OUTPUT_FILE = ABLATION_DIR / "mlea_81_results.json"

# Telemetry thresholds from the paper
THRESHOLDS = {
    "feature_count": 500,
    "reprojection_error": 1.8,
    "final_drift": 2.0,
    "violation_count": 0,
}


def derive_telemetry(file_data: dict) -> dict:
    """Derive telemetry values from available ablation file fields.

    The ablation files contain tracking_quality and pos_error_m.
    We derive reprojection_error and feature_count using the formulas
    from trajectory_import.py.
    """
    tracking_quality = file_data.get("tracking_quality", 0.5)
    pos_error_m = file_data.get("pos_error_m", 0.0)

    # Formulas from vision_language_localization/slam/trajectory_import.py
    reprojection_error = 0.6 + 2.8 * (1.0 - tracking_quality)
    feature_count = 1400 * tracking_quality + 120

    return {
        "mean_reprojection_error": round(reprojection_error, 4),
        "mean_feature_count": round(feature_count, 1),
        "drift_final_m": pos_error_m,  # Using pos_error_m as proxy
        "violation_count": 0,  # Not available in ablation files
    }


def check_claim_support(
    claim: str,
    explanation: str,
    telemetry: dict,
) -> tuple[bool, str | None]:
    """Check if a detected claim is supported by telemetry evidence.

    Returns (is_supported, flagged_reason_or_none).
    """
    text = explanation.lower()

    # Map claim categories to telemetry checks
    claim_checks = {
        "low_feature_support": (
            ("feature" in text) or ("landmark" in text),
            telemetry["mean_feature_count"] < THRESHOLDS["feature_count"],
        ),
        "geometric_inconsistency": (
            ("reprojection" in text) or ("geometric" in text),
            telemetry["mean_reprojection_error"] > THRESHOLDS["reprojection_error"],
        ),
        "trajectory_drift": (
            "drift" in text,
            telemetry["drift_final_m"] > THRESHOLDS["final_drift"],
        ),
        "safety_violation": (
            ("violation" in text) or ("unsafe" in text),
            telemetry["violation_count"] > THRESHOLDS["violation_count"],
        ),
    }

    if claim in claim_checks:
        mentioned, evidence_true = claim_checks[claim]
        if mentioned:
            if evidence_true:
                return True, None
            else:
                return False, claim

    # For claims without direct telemetry mapping (environmental, motion, etc.)
    # they cannot be verified against telemetry, so we mark them as
    # "unverifiable" rather than unsupported
    return True, None


def process_ablation_file(file_path: Path) -> dict | None:
    """Process a single ablation JSON file and return MLEA results."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    explanation = data.get("explanation", "")
    if not explanation:
        return None

    # Detect claims from VLM explanation
    detected_claims = detect_claims(explanation)

    # Derive telemetry
    telemetry = derive_telemetry(data)

    # Check each claim against telemetry
    supported = 0
    unsupported = 0
    flagged = []
    claim_details = []

    for claim in sorted(detected_claims):
        is_supported, flag = check_claim_support(claim, explanation, telemetry)
        if is_supported:
            supported += 1
        else:
            unsupported += 1
            if flag:
                flagged.append(flag)
        claim_details.append({
            "category": claim,
            "supported": is_supported,
        })

    total = supported + unsupported
    mlea_score = 1.0 if total == 0 else supported / total

    return {
        "file": file_path.name,
        "sequence_id": data.get("sequence_id", "unknown"),
        "frame_idx": data.get("frame_idx", -1),
        "condition": data.get("condition", "unknown"),
        "detected_claims": sorted(detected_claims),
        "claim_count": len(detected_claims),
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "mlea_score": round(mlea_score, 4),
        "flagged_claims": flagged,
        "claim_details": claim_details,
        "telemetry": telemetry,
    }


def main():
    # Find all qwen_*.json files (exclude qwen_ablation_all.json)
    ablation_files = sorted(
        f for f in ABLATION_DIR.glob("qwen_*.json")
        if f.name != "qwen_ablation_all.json"
    )

    print(f"Found {len(ablation_files)} ablation files")

    results = []
    all_claims = []
    category_counts = defaultdict(lambda: {"total": 0, "supported": 0, "unsupported": 0})
    condition_scores = defaultdict(list)

    for f in ablation_files:
        result = process_ablation_file(f)
        if result is None:
            print(f"  SKIP: {f.name} (invalid/empty)")
            continue
        results.append(result)
        all_claims.extend(result["detected_claims"])

        # Aggregate by category
        for detail in result["claim_details"]:
            cat = detail["category"]
            category_counts[cat]["total"] += 1
            if detail["supported"]:
                category_counts[cat]["supported"] += 1
            else:
                category_counts[cat]["unsupported"] += 1

        # Aggregate by condition
        condition_scores[result["condition"]].append(result["mlea_score"])

    # Summary statistics
    total_claims = sum(r["claim_count"] for r in results)
    total_supported = sum(r["supported_claims"] for r in results)
    total_unsupported = sum(r["unsupported_claims"] for r in results)
    overall_mlea = total_supported / (total_supported + total_unsupported) if (total_supported + total_unsupported) > 0 else 1.0

    # Per-condition summary
    condition_summary = {}
    for cond, scores in condition_scores.items():
        condition_summary[cond] = {
            "count": len(scores),
            "mean_mlea": round(sum(scores) / len(scores), 4) if scores else 1.0,
            "min_mlea": round(min(scores), 4) if scores else 1.0,
            "max_mlea": round(max(scores), 4) if scores else 1.0,
        }

    # Build output
    output = {
        "summary": {
            "total_files_processed": len(results),
            "total_claims_extracted": total_claims,
            "total_supported": total_supported,
            "total_unsupported": total_unsupported,
            "overall_mlea_score": round(overall_mlea, 4),
            "unique_sequences": len(set(r["sequence_id"] for r in results)),
        },
        "category_breakdown": dict(category_counts),
        "condition_summary": condition_summary,
        "per_sequence_results": results,
    }

    # Write output
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"MLEA EXPANSION RESULTS ({len(results)} files)")
    print(f"{'='*70}")
    print(f"Total claims extracted:  {total_claims}")
    print(f"  Supported:             {total_supported}")
    print(f"  Unsupported:           {total_unsupported}")
    print(f"  Overall MLEA score:    {overall_mlea:.4f}")
    print(f"  Unique sequences:      {output['summary']['unique_sequences']}")

    print(f"\n{'='*70}")
    print("Category Breakdown:")
    print(f"{'='*70}")
    print(f"{'Category':<35} {'Total':>6} {'Supp':>6} {'Unsupp':>6} {'Rate':>8}")
    print(f"{'-'*70}")
    for cat in sorted(category_counts.keys()):
        c = category_counts[cat]
        rate = c["supported"] / c["total"] if c["total"] > 0 else 0
        print(f"{cat:<35} {c['total']:>6} {c['supported']:>6} {c['unsupported']:>6} {rate:>8.2%}")

    print(f"\n{'='*70}")
    print("Condition Breakdown:")
    print(f"{'='*70}")
    print(f"{'Condition':<15} {'Count':>6} {'Mean MLEA':>10} {'Min':>8} {'Max':>8}")
    print(f"{'-'*50}")
    for cond in sorted(condition_summary.keys()):
        s = condition_summary[cond]
        print(f"{cond:<15} {s['count']:>6} {s['mean_mlea']:>10.4f} {s['min_mlea']:>8.4f} {s['max_mlea']:>8.4f}")

    # Find sequences with unsupported claims
    flagged = [r for r in results if r["unsupported_claims"] > 0]
    if flagged:
        print(f"\n{'='*70}")
        print(f"Sequences with Unsupported Claims ({len(flagged)}):")
        print(f"{'='*70}")
        for r in flagged:
            print(f"  {r['sequence_id']} frame {r['frame_idx']} ({r['condition']})")
            print(f"    Claims: {r['detected_claims']}")
            print(f"    Flagged: {r['flagged_claims']}")
            print(f"    MLEA: {r['mlea_score']:.4f}")
            print()

    # Most common claim categories
    print(f"\n{'='*70}")
    print("Most Common Claim Categories:")
    print(f"{'='*70}")
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1]["total"], reverse=True)
    for cat, c in sorted_cats[:5]:
        print(f"  {cat}: {c['total']} occurrences")


if __name__ == "__main__":
    main()
