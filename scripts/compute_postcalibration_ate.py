"""Compute post-calibration ATE using the same pipeline as Table 1."""
import json
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vision_language_localization.data.kitti_loader import discover_drive_roots, load_kitti_sequence
from vision_language_localization.slam.trajectory_import import load_external_slam_output
from vision_language_localization.sensor_fusion.alignment import estimate_registration, apply_similarity

DATASET_ROOT = REPO_ROOT / "dataset"
SLAM_ROOT = REPO_ROOT / "slam"
OUTPUT = REPO_ROOT / "outputs" / "ablation" / "post_calibration_ate.json"
REGISTRATION_WINDOW = 50

SEQUENCES = [
    "2011_09_26_drive_0001_sync", "2011_09_26_drive_0005_sync",
    "2011_09_26_drive_0009_sync", "2011_09_26_drive_0011_sync",
    "2011_09_26_drive_0013_sync", "2011_09_26_drive_0014_sync",
    "2011_09_26_drive_0015_sync", "2011_09_26_drive_0017_sync",
    "2011_09_26_drive_0023_sync", "2011_09_26_drive_0036_sync",
    "2011_09_26_drive_0093_sync", "2011_09_28_drive_0001_sync",
    "2011_09_28_drive_0034_sync", "2011_09_28_drive_0047_sync",
    "2011_09_29_drive_0026_sync", "2011_09_29_drive_0071_sync",
    "2011_09_30_drive_0020_sync", "2011_09_30_drive_0033_sync",
    "2011_10_03_drive_0042_sync", "2011_10_03_drive_0047_sync",
]


def main():
    all_aligned_all = []
    all_aligned_post = []
    all_ekf_all = []
    all_ekf_post = []

    for seq_id in SEQUENCES:
        traj_file = SLAM_ROOT / f"{seq_id}.kitti.txt"
        if not traj_file.exists():
            print(f"[SKIP] {seq_id}: no trajectory file")
            continue

        drive_root = None
        for dr in discover_drive_roots(DATASET_ROOT):
            if dr.name == seq_id:
                drive_root = dr
                break
        if drive_root is None:
            print(f"[SKIP] {seq_id}: no drive root")
            continue

        try:
            seq = load_kitti_sequence(drive_root)
            slam_out = load_external_slam_output(traj_file, seq.gt_xyz)
        except Exception as e:
            print(f"[SKIP] {seq_id}: {e}")
            continue

        n = min(len(seq.gt_xyz), len(slam_out.est_xyz))
        gt = seq.gt_xyz[:n]
        slam = slam_out.est_xyz[:n]
        quality = slam_out.tracking_quality[:n]

        # Alignment (same as compute_aligned_slam: skip if registration fails, keep unaligned)
        reg = estimate_registration(slam, gt, quality, window=REGISTRATION_WINDOW, min_frames=10)
        if reg is not None:
            slam_aligned = apply_similarity(slam, reg)
        else:
            slam_aligned = slam.copy()

        # Aligned ATE (all frames)
        ate_aligned_all = float(np.sqrt(np.mean(np.sum((slam_aligned - gt) ** 2, axis=1))))

        # Aligned ATE (post-50 only)
        if n > REGISTRATION_WINDOW:
            ate_aligned_post = float(np.sqrt(np.mean(np.sum((slam_aligned[REGISTRATION_WINDOW:] - gt[REGISTRATION_WINDOW:]) ** 2, axis=1))))
        else:
            ate_aligned_post = ate_aligned_all

        all_aligned_all.append(ate_aligned_all)
        all_aligned_post.append(ate_aligned_post)

        print(f"{seq_id}: aligned_all={ate_aligned_all:.1f}  post50={ate_aligned_post:.1f}")

    print(f"\n{'='*70}")
    print(f"Sequences: {len(all_aligned_all)}")
    print(f"Mean aligned ATE (all frames):     {np.mean(all_aligned_all):.2f}")
    print(f"Mean aligned ATE (post-50 only):   {np.mean(all_aligned_post):.2f}")
    print(f"Median aligned ATE (all frames):   {np.median(all_aligned_all):.2f}")
    print(f"Median aligned ATE (post-50 only): {np.median(all_aligned_post):.2f}")
    pct = (np.mean(all_aligned_post) - np.mean(all_aligned_all)) / np.mean(all_aligned_all) * 100
    print(f"Change: {pct:+.1f}%")

    result = {
        "n_sequences": len(all_aligned_all),
        "mean_aligned_all": round(float(np.mean(all_aligned_all)), 2),
        "mean_aligned_post50": round(float(np.mean(all_aligned_post)), 2),
        "median_aligned_all": round(float(np.median(all_aligned_all)), 2),
        "median_aligned_post50": round(float(np.median(all_aligned_post)), 2),
        "change_pct": round(float(pct), 1),
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
