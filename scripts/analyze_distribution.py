"""Analyze distribution of metrics across 20 sequences for failure event selection."""
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
all_results = json.loads((REPO_ROOT / "outputs" / "ablation" / "all_sequences.json").read_text())

print(f"=== Distribution Analysis: {len(all_results)} sequences ===\n")

# Extract metrics
slam_ates = [r["slam_metrics"]["ate_rmse"] for r in all_results]
ekf_ates = [r["fusion_metrics"]["ate_rmse"] for r in all_results]
al_ates = [r["aligned_slam_metrics"]["ate_rmse"] for r in all_results]
stl_robs = [r["runtime_verification"]["stl_robustness"] for r in all_results]
drifts = [r["slam_metrics"]["drift_final_m"] for r in all_results]
frames = [r["num_frames"] for r in all_results]

for name, vals in [("SLAM ATE", slam_ates), ("EKF ATE", ekf_ates), ("Aligned ATE", al_ates),
                    ("STL robustness", stl_robs), ("Drift (m)", drifts), ("Frames", frames)]:
    print(f"{name}: min={min(vals):.2f}, max={max(vals):.2f}, "
          f"median={statistics.median(vals):.2f}, mean={statistics.mean(vals):.2f}, "
          f"stdev={statistics.stdev(vals):.2f}")

# Improvement ratio
print("\n=== EKF Improvement ===")
improved = 0
degraded = 0
for r in all_results:
    slam = r["slam_metrics"]["ate_rmse"]
    ekf = r["fusion_metrics"]["ate_rmse"]
    if ekf < slam:
        improved += 1
    elif ekf > slam:
        degraded += 1
print(f"Improved: {improved}/{len(all_results)}")
print(f"Degraded: {degraded}/{len(all_results)}")

# Identify high-error sequences (good candidates for failure analysis)
print("\n=== High Error Sequences (EKF ATE > 10m) ===")
for r in sorted(all_results, key=lambda x: x["fusion_metrics"]["ate_rmse"], reverse=True):
    ekf = r["fusion_metrics"]["ate_rmse"]
    if ekf > 10:
        print(f"  {r['sequence_id']}: EKF ATE={ekf:.1f}m, SLAM ATE={r['slam_metrics']['ate_rmse']:.1f}m")

# Identify sequences with STL violations
print("\n=== STL Violation Sequences ===")
for r in sorted(all_results, key=lambda x: x["runtime_verification"]["stl_robustness"]):
    rob = r["runtime_verification"]["stl_robustness"]
    if rob < 0:
        print(f"  {r['sequence_id']}: STL robustness={rob:.1f}")

# Scene diversity
print("\n=== Frame Counts (diversity check) ===")
for r in all_results:
    print(f"  {r['sequence_id']}: {r['num_frames']} frames")
