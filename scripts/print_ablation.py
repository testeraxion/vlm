"""Print ablation results from pipeline output."""
import json

data = json.load(open("outputs/ablation/latest_run.json"))

print("=== Full Ablation Results ===")
print()

print("ATE (m) - lower is better")
header = f"{'Seq':<8} {'Raw SLAM':>10} {'Aligned':>10} {'GPS':>10} {'EKF':>10}"
print(header)
print("-" * 52)
for s in data:
    sid = s["sequence_id"].split("_drive_")[1].split("_sync")[0]
    raw = s["slam_metrics"]["ate_rmse"]
    aligned = s["aligned_slam_metrics"]["ate_rmse"]
    gps = s["gps_only_metrics"]["ate_rmse"]
    fused = s["fusion_metrics"]["ate_rmse"]
    print(f"{sid:<8} {raw:>10.2f} {aligned:>10.2f} {gps:>10.2f} {fused:>10.2f}")

print()
print("Drift (m) - lower is better")
print(header)
print("-" * 52)
for s in data:
    sid = s["sequence_id"].split("_drive_")[1].split("_sync")[0]
    raw = s["slam_metrics"]["drift_final_m"]
    aligned = s["aligned_slam_metrics"]["drift_final_m"]
    gps = s["gps_only_metrics"]["drift_final_m"]
    fused = s["fusion_metrics"]["drift_final_m"]
    print(f"{sid:<8} {raw:>10.2f} {aligned:>10.2f} {gps:>10.2f} {fused:>10.2f}")

print()
print("RPE (m) - lower is better")
print(header)
print("-" * 52)
for s in data:
    sid = s["sequence_id"].split("_drive_")[1].split("_sync")[0]
    raw = s["slam_metrics"]["rpe_mean"]
    aligned = s["aligned_slam_metrics"]["rpe_mean"]
    gps = s["gps_only_metrics"]["rpe_mean"]
    fused = s["fusion_metrics"]["rpe_mean"]
    print(f"{sid:<8} {raw:>10.4f} {aligned:>10.4f} {gps:>10.4f} {fused:>10.4f}")
