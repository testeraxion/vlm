"""Batch run pipeline on all sequences with rule-based VLM."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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

results = []
for seq_id in SEQUENCES:
    print(f"[RUN] {seq_id}...", flush=True)
    try:
        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "run_pipeline.py"),
                "--slam-backend", "trajectory_file",
                "--trajectory-root", str(REPO_ROOT / "slam"),
                "--trajectory-file-suffix", ".kitti.txt",
                "--max-frames", "2000",
                "--vlm-backend", "rule_based",
                "--fusion-method", "ekf",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=300,
            env={**dict(__import__("os").environ), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if result.returncode == 0:
            print(f"  OK")
            results.append(seq_id)
        else:
            print(f"  FAIL (rc={result.returncode})")
            # Print last few lines of stderr for debugging
            if result.stderr:
                for line in result.stderr.strip().splitlines()[-5:]:
                    print(f"    {line}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nCompleted: {len(results)}/{len(SEQUENCES)} sequences")
