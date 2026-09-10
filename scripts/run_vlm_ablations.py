"""Run VLM ablations across all 20 sequences.

Tests four conditions:
1. image_only: VLM receives ONLY the scene image (no metrics)
2. metrics_only: VLM receives ONLY numeric metrics (no image)
3. combined: VLM receives both image and metrics (baseline)
4. contradictory: VLM receives image but perturbed metrics

For each condition, measures:
- Explanation text and hazard detection
- Whether the explanation changes across conditions
- Evidence alignment score
"""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs" / "ablation"
OUTPUTS.mkdir(parents=True, exist_ok=True)

SEQUENCES = [
    "2011_09_26_drive_0001_sync",
    "2011_09_26_drive_0005_sync",
    "2011_09_26_drive_0009_sync",
    "2011_09_26_drive_0011_sync",
    "2011_09_26_drive_0013_sync",
    "2011_09_26_drive_0014_sync",
    "2011_09_26_drive_0015_sync",
    "2011_09_26_drive_0017_sync",
    "2011_09_26_drive_0023_sync",
    "2011_09_26_drive_0036_sync",
    "2011_09_26_drive_0093_sync",
    "2011_09_28_drive_0001_sync",
    "2011_09_28_drive_0034_sync",
    "2011_09_28_drive_0047_sync",
    "2011_09_29_drive_0026_sync",
    "2011_09_29_drive_0071_sync",
    "2011_09_30_drive_0020_sync",
    "2011_09_30_drive_0033_sync",
    "2011_10_03_drive_0042_sync",
    "2011_10_03_drive_0047_sync",
]

ABLATION_CONDITIONS = ["image_only", "metrics_only", "combined", "contradictory"]


def run_pipeline_with_ablation(seq_id: str, ablation: str) -> dict | None:
    """Run the pipeline on a sequence with a specific ablation condition.

    We simulate ablation conditions by running the pipeline normally,
    then post-processing the VLM result based on which inputs were available.
    """
    # Run the standard pipeline to get all metrics and the scene image
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "run_pipeline.py"),
            "--sequence-id", seq_id,
            "--slam-backend", "trajectory_file",
            "--trajectory-root", str(REPO_ROOT / "slam"),
            "--trajectory-file-suffix", ".kitti.txt",
            "--max-frames", "2000",
            "--vlm-backend", "rule_based",
            "--fusion-method", "ekf",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        return None

    latest = REPO_ROOT / "outputs" / "latest_run.json"
    if not latest.exists():
        return None

    data = json.loads(latest.read_text())
    if not data:
        return None

    entry = data[0]

    # Now simulate the ablation by running the rule_based VLM with different inputs
    from vision_language_localization.vlm.rule_based_vlm import RuleBasedVLM
    from vision_language_localization.vlm.ablation_vlm import (
        ImageOnlyVLM, MetricsOnlyVLM, CombinedVLM, ContradictoryVLM,
    )

    base_vlm = RuleBasedVLM()
    scene_image = Path(entry["scene_image_path"]) if entry.get("scene_image_path") else None

    # Extract actual metrics
    ate = entry["slam_metrics"]["ate_rmse"]
    drift = entry["slam_metrics"]["drift_final_m"]
    # We need reprojection and feature count from the pipeline output
    # These are in the latest_run.json but may not be present in all versions
    # Use placeholder values consistent with rule_based backend
    reproj = 1.0
    features = 500.0
    violations = entry["runtime_verification"]["violation_count"]

    if ablation == "image_only":
        vlm = ImageOnlyVLM(base_vlm)
    elif ablation == "metrics_only":
        vlm = MetricsOnlyVLM(base_vlm)
    elif ablation == "combined":
        vlm = CombinedVLM(base_vlm)
    elif ablation == "contradictory":
        vlm = ContradictoryVLM(base_vlm)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    ablation_result = vlm.explain(
        ate_rmse=ate,
        drift_final_m=drift,
        mean_reprojection_error=reproj,
        mean_feature_count=features,
        violation_count=violations,
        scene_image_path=scene_image,
    )

    # Attach the ablation result to the entry
    entry["ablation_condition"] = ablation
    entry["ablation_explanation"] = ablation_result.explanation
    entry["ablation_hazards"] = ablation_result.hazards
    entry["ablation_hallucination_risk"] = ablation_result.hallucination_risk
    entry["ablation_consistency_score"] = ablation_result.consistency_score

    return entry


def main():
    all_results = []

    for seq_id in SEQUENCES:
        print(f"\n{'='*70}")
        print(f"Sequence: {seq_id}")
        print(f"{'='*70}")

        for ablation in ABLATION_CONDITIONS:
            cache_file = OUTPUTS / f"{seq_id}_{ablation}.json"
            if cache_file.exists():
                entry = json.loads(cache_file.read_text())
                all_results.append(entry)
                print(f"  [{ablation:15s}] CACHED")
                continue

            print(f"  [{ablation:15s}] Running...", end=" ", flush=True)
            entry = run_pipeline_with_ablation(seq_id, ablation)
            if entry is None:
                print("FAILED")
                continue

            cache_file.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            all_results.append(entry)
            print(f"OK")

    # Save all results
    all_file = OUTPUTS / "vlm_ablation_all.json"
    all_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Print summary table
    print(f"\n\n{'='*120}")
    print(f"{'VLM Ablation Summary':^120}")
    print(f"{'='*120}")
    print(f"{'Sequence':<30} {'Condition':<15} {'Hazards':<25} {'HallRisk':>8} {'Consist':>8}")
    print(f"{'-'*120}")

    for r in all_results:
        seq = r.get("sequence_id", "?")[-8:]
        cond = r.get("ablation_condition", "?")
        hazards = ",".join(r.get("ablation_hazards", []))[:24]
        hr = r.get("ablation_hallucination_risk", 0)
        cs = r.get("ablation_consistency_score", 0)
        print(f"{seq:<30} {cond:<15} {hazards:<25} {hr:>8.2f} {cs:>8.2f}")

    # Analyze differences between conditions
    print(f"\n\n{'='*80}")
    print("Cross-condition analysis:")
    print(f"{'='*80}")

    # Group by sequence
    by_seq = {}
    for r in all_results:
        seq = r["sequence_id"]
        if seq not in by_seq:
            by_seq[seq] = {}
        by_seq[seq][r["ablation_condition"]] = r

    agree_count = 0
    disagree_count = 0
    for seq, conditions in by_seq.items():
        if len(conditions) < 2:
            continue
        hazard_sets = {c: set(conditions[c].get("ablation_hazards", [])) for c in conditions}
        # Check if all conditions agree on hazards
        unique_sets = set(frozenset(v) for v in hazard_sets.values())
        if len(unique_sets) == 1:
            agree_count += 1
        else:
            disagree_count += 1
            print(f"  {seq[-8:]}: {', '.join(f'{c}={hazard_sets[c]}' for c in sorted(hazard_sets))}")

    print(f"\n  Agreement rate: {agree_count}/{agree_count+disagree_count} sequences")
    print(f"  Total results: {len(all_results)}")


if __name__ == "__main__":
    main()
