"""Extract aligned ATEs from all_sequences.json and compute post-calibration stats."""
import json
import numpy as np
from pathlib import Path

data = json.loads(Path("outputs/ablation/all_sequences.json").read_text())

print("All 20 sequences:")
for r in data:
    sid = r["sequence_id"]
    aligned = r["aligned_slam_metrics"]["ate_rmse"]
    ekf = r["fusion_metrics"]["ate_rmse"]
    raw = r["slam_metrics"]["ate_rmse"]
    n = r["num_frames"]
    print(f"  {sid}: raw={raw:.1f} aligned={aligned:.1f} ekf={ekf:.1f} frames={n}")

aligned_ates = [r["aligned_slam_metrics"]["ate_rmse"] for r in data]
ekf_ates = [r["fusion_metrics"]["ate_rmse"] for r in data]
raw_ates = [r["slam_metrics"]["ate_rmse"] for r in data]

print(f"\nN={len(data)}")
print(f"Raw mean={np.mean(raw_ates):.2f} median={np.median(raw_ates):.2f}")
print(f"Aligned mean={np.mean(aligned_ates):.2f} median={np.median(aligned_ates):.2f}")
print(f"EKF mean={np.mean(ekf_ates):.2f} median={np.median(ekf_ates):.2f}")
