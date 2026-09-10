"""Select failure events from 20 sequences using runtime telemetry.

Selection criteria (all based on runtime signals, NOT ground truth):
1. STL violation frames (position error > threshold or tracking lost)
2. High EKF innovation frames (large difference between predicted and measured position)
3. Low tracking quality frames
4. Frames with large frame-to-frame drift

Target: 3-5 events per sequence, 60-100 total.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_language_localization.data.kitti_loader import discover_drive_roots, load_kitti_sequence, latlon_to_local_xy
from vision_language_localization.slam.trajectory_import import load_external_slam_output
from vision_language_localization.sensor_fusion.ekf_fusion import EkfFusion, EkfFusionConfig
from vision_language_localization.runtime_verification.stl_monitor import evaluate_runtime_properties

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


def select_failure_events_for_sequence(seq_id: str) -> list[dict]:
    """Select 3-5 failure events for a sequence using runtime telemetry."""
    # Load sequence
    dataset_root = REPO_ROOT / "dataset"
    drives = discover_drive_roots(dataset_root)
    drive = None
    for d in drives:
        if seq_id in d.name or d.name.endswith(seq_id):
            drive = d
            break
    if drive is None:
        # Try finding by sequence ID in path parts
        for d in drives:
            if seq_id in str(d):
                drive = d
                break
    if drive is None:
        print(f"  Cannot find drive for {seq_id}")
        return []

    seq = load_kitti_sequence(drive, max_frames=2000)
    n = len(seq.frames)
    if n < 10:
        return []

    # Load SLAM trajectory
    traj_file = REPO_ROOT / "slam" / f"{seq_id}.kitti.txt"
    if not traj_file.exists():
        print(f"  Missing trajectory for {seq_id}")
        return []

    slam_out = load_external_slam_output(trajectory_file=traj_file, gt_xyz=seq.gt_xyz)
    n = min(n, len(slam_out.est_xyz))

    # EKF fusion
    ekf = EkfFusion(EkfFusionConfig())
    fusion_out = ekf.run(seq, slam_out)
    fused_xyz = fusion_out.fused_xyz[:n]
    gt_xyz = seq.gt_xyz[:n]

    # Per-frame metrics
    pos_error = np.linalg.norm(fused_xyz - gt_xyz, axis=1)

    # Frame-to-frame drift
    frame_drift = np.zeros(n)
    for i in range(1, n):
        frame_drift[i] = np.linalg.norm(fused_xyz[i] - fused_xyz[i-1] - (gt_xyz[i] - gt_xyz[i-1]))

    # Tracking quality
    tracking_quality = slam_out.tracking_quality[:n]

    # Runtime verification
    rv = evaluate_runtime_properties(
        position_error_m=pos_error,
        tracking_ok=slam_out.tracking_ok[:n],
        dt_s=0.1,
        max_position_error_m=1.0,
        max_lost_duration_s=2.0,
        recovery_window_s=5.0,
    )

    # Select failure events using runtime signals
    events = []
    for i in range(n):
        frame = seq.frames[i]
        event = {
            "frame_idx": i,
            "timestamp_s": frame.timestamp_s,
            "image_path": str(frame.image_path) if frame.image_path else None,
            "pos_error_m": float(pos_error[i]),
            "frame_drift_m": float(frame_drift[i]),
            "tracking_quality": float(tracking_quality[i]),
            "tracking_ok": bool(slam_out.tracking_ok[i]),
        }
        events.append(event)

    # Score and rank events by runtime signals
    for e in events:
        score = 0.0
        # High position error (runtime: compare fused vs predicted)
        if e["pos_error_m"] > 5.0:
            score += 3.0
        elif e["pos_error_m"] > 2.0:
            score += 2.0
        elif e["pos_error_m"] > 1.0:
            score += 1.0
        # Large frame-to-frame drift
        if e["frame_drift_m"] > 2.0:
            score += 2.0
        elif e["frame_drift_m"] > 1.0:
            score += 1.0
        # Low tracking quality
        if e["tracking_quality"] < 0.25:
            score += 3.0
        elif e["tracking_quality"] < 0.5:
            score += 1.0
        # Tracking lost
        if not e["tracking_ok"]:
            score += 2.0
        e["failure_score"] = score

    # Sort by failure score
    events.sort(key=lambda x: x["failure_score"], reverse=True)

    # Select top 3-5 events, ensuring temporal diversity (at least 30 frames apart)
    selected = []
    min_gap = 30
    for e in events:
        if len(selected) >= 5:
            break
        if e["failure_score"] <= 0:
            break
        # Check temporal distance from already selected
        too_close = False
        for s in selected:
            if abs(e["frame_idx"] - s["frame_idx"]) < min_gap:
                too_close = True
                break
        if not too_close:
            selected.append(e)

    # If fewer than 3, add the peak-error frame
    if len(selected) < 3:
        peak_idx = int(np.argmax(pos_error))
        if not any(s["frame_idx"] == peak_idx for s in selected):
            selected.append(events[0] if events[0]["frame_idx"] != peak_idx else events[1])

    return selected[:5]


def main():
    all_events = []
    for seq_id in SEQUENCES:
        print(f"[{seq_id}] Selecting failure events...", end=" ", flush=True)
        events = select_failure_events_for_sequence(seq_id)
        for e in events:
            e["sequence_id"] = seq_id
        all_events.extend(events)
        print(f"selected {len(events)} events")

    # Save
    output_file = OUTPUTS / "failure_events.json"
    output_file.write_text(json.dumps(all_events, indent=2), encoding="utf-8")

    # Summary
    print(f"\n{'='*70}")
    print(f"Total failure events: {len(all_events)}")
    print(f"Per sequence: {len(all_events)/len(SEQUENCES):.1f} average")
    print(f"\nScore distribution:")
    scores = [e["failure_score"] for e in all_events]
    print(f"  Mean: {np.mean(scores):.2f}, Median: {np.median(scores):.1f}, Max: {np.max(scores):.1f}")

    print(f"\n{'='*70}")
    print(f"{'Sequence':<35} {'Frame':>6} {'PosErr':>8} {'Drift':>8} {'Quality':>8} {'Score':>6}")
    print(f"{'-'*70}")
    for e in all_events:
        print(f"{e['sequence_id']:<35} {e['frame_idx']:>6} {e['pos_error_m']:>8.2f} {e['frame_drift_m']:>8.2f} {e['tracking_quality']:>8.3f} {e['failure_score']:>6.1f}")

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
