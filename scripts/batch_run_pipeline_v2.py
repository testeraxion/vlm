"""Batch run pipeline on all sequences, saving per-sequence results."""
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

all_results = []

for seq_id in SEQUENCES:
    seq_file = OUTPUTS / f"{seq_id}.json"
    if seq_file.exists():
        data = json.loads(seq_file.read_text())
        all_results.append(data)
        print(f"[SKIP] {seq_id} (cached)")
        continue

    print(f"[RUN] {seq_id}...", flush=True)
    try:
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
        if result.returncode == 0:
            latest = REPO_ROOT / "outputs" / "latest_run.json"
            if latest.exists():
                data = json.loads(latest.read_text())
                if data:
                    entry = data[0]
                    seq_file.write_text(json.dumps(entry, indent=2), encoding="utf-8")
                    all_results.append(entry)
                    print(f"  OK (SLAM ATE={entry['slam_metrics']['ate_rmse']:.1f}, EKF ATE={entry['fusion_metrics']['ate_rmse']:.1f})")
                else:
                    print(f"  WARN: empty result")
            else:
                print(f"  WARN: no latest_run.json")
        else:
            print(f"  FAIL (rc={result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-3:]:
                    print(f"    {line}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
    except Exception as e:
        print(f"  ERROR: {e}")

# Save aggregated results
aggregated = OUTPUTS / "all_sequences.json"
aggregated.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

# Print summary
print(f"\n{'='*100}")
print(f"{'Sequence':<45} {'N':>5} {'SLAM ATE':>10} {'Align':>10} {'EKF ATE':>10} {'STL rob':>10}")
print(f"{'-'*100}")

for r in all_results:
    seq = r.get("sequence_id", "unknown")
    n = r.get("num_frames", 0)
    slam_ate = r.get("slam_metrics", {}).get("ate_rmse", 0)
    al_ate = r.get("aligned_slam_metrics", {}).get("ate_rmse", 0)
    ekf_ate = r.get("fusion_metrics", {}).get("ate_rmse", 0)
    stl = r.get("runtime_verification", {}).get("stl_robustness", 0)
    print(f"{seq:<45} {n:>5} {slam_ate:>10.1f} {al_ate:>10.1f} {ekf_ate:>10.1f} {stl:>10.1f}")

print(f"\nCompleted: {len(all_results)}/{len(SEQUENCES)} sequences")
