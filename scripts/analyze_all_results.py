"""Analyze all pipeline results across 20 sequences."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"

# Collect all sequence results from the pipeline
all_results = []

# Check the latest_run.json first
latest = OUTPUTS / "ablation" / "latest_run.json"
if latest.exists():
    data = json.loads(latest.read_text())
    for entry in data:
        all_results.append(entry)

# Also check for per-sequence results in the pipeline output directory
for report in OUTPUTS.rglob("report_*.json"):
    if report.name == "report_latest.json":
        continue
    try:
        data = json.loads(report.read_text())
        if isinstance(data, dict) and "sequence_id" in data:
            seq_id = data["sequence_id"]
            if not any(r.get("sequence_id") == seq_id for r in all_results):
                all_results.append(data)
    except Exception:
        pass

print(f"Found results for {len(all_results)} sequences\n")

# Print summary table
print(f"{'Sequence':<45} {'Frames':>6} {'SLAM ATE':>10} {'EKF ATE':>10} {'AL ATE':>10} {'STL rob':>10}")
print("-" * 95)

for r in all_results:
    seq = r.get("sequence_id", "unknown")
    frames = r.get("num_frames", 0)
    slam_ate = r.get("slam_metrics", {}).get("ate_rmse", 0)
    ekf_ate = r.get("fusion_metrics", {}).get("ate_rmse", 0)
    al_ate = r.get("aligned_slam_metrics", {}).get("ate_rmse", 0)
    stl = r.get("runtime_verification", {}).get("stl_robustness", 0)
    print(f"{seq:<45} {frames:>6} {slam_ate:>10.2f} {ekf_ate:>10.2f} {al_ate:>10.2f} {stl:>10.2f}")

# Save aggregated results
aggregated_path = OUTPUTS / "ablation" / "all_sequences.json"
aggregated_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
print(f"\nSaved aggregated results to {aggregated_path}")
