"""Batch run ORB-SLAM3 stereo on all prepared sequences and convert to KITTI format."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORB_ROOT = REPO_ROOT / "slam" / "ORB_SLAM3"
VOCAB = ORB_ROOT / "Vocabulary" / "ORBvoc.txt"
STAMPS_ROOT = REPO_ROOT / "slam" / "prepared"
OUTPUT_DIR = REPO_ROOT / "slam"
PANGOLIN_LIB = "/tmp/pangolin_install/lib"

# Collect all prepared sequences
sequences = sorted([d.name for d in STAMPS_ROOT.iterdir() if d.is_dir()])
print(f"Found {len(sequences)} prepared sequences")

# Check vocabulary exists
if not VOCAB.exists():
    tarball = ORB_ROOT / "Vocabulary" / "ORBvoc.txt.tar.gz"
    if tarball.exists():
        print("Extracting ORB vocabulary...")
        subprocess.run(["tar", "-xzf", str(tarball), "-C", str(ORB_ROOT / "Vocabulary")], check=True)
    else:
        print(f"Missing vocabulary at {VOCAB}")
        sys.exit(1)

binary = ORB_ROOT / "Examples" / "Stereo" / "stereo_kitti"
if not binary.exists():
    print(f"Missing binary at {binary}. Build ORB-SLAM3 first.")
    sys.exit(1)

for seq_id in sequences:
    prepared = STAMPS_ROOT / seq_id
    output_traj = OUTPUT_DIR / f"{seq_id}.kitti.txt"

    if output_traj.exists():
        n_lines = len(output_traj.read_text().strip().splitlines())
        if n_lines > 10:
            print(f"[SKIP] {seq_id} already has trajectory ({n_lines} lines)")
            continue

    settings = prepared / "settings.yaml"
    if not settings.exists():
        print(f"[SKIP] {seq_id}: no settings.yaml")
        continue

    print(f"[RUN] {seq_id}...", flush=True)
    env = {"LD_LIBRARY_PATH": f"{PANGOLIN_LIB}:{ORB_ROOT}/lib", "DISPLAY": ""}
    try:
        result = subprocess.run(
            [str(binary), str(VOCAB), str(settings), str(prepared)],
            cwd=str(ORB_ROOT),
            env={**dict(__import__("os").environ), **env},
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {seq_id}")
        continue

    cam_traj = ORB_ROOT / "CameraTrajectory.txt"
    if cam_traj.exists():
        # Convert to KITTI format (12 floats per line from 4x4 matrix)
        kitti_lines = []
        for line in cam_traj.read_text().strip().splitlines():
            vals = [float(v) for v in line.split()]
            if len(vals) == 12:
                kitti_lines.append(" ".join(f"{v:.9f}" for v in vals))
            elif len(vals) == 3:
                # Timestamp-only format from ORB-SLAM3
                pass
            elif len(vals) >= 7:
                # Quaternion format: tx ty tz qx qy qz qw
                tx, ty, tz = vals[0], vals[1], vals[2]
                qx, qy, qz, qw = vals[3], vals[4], vals[5], vals[6]
                # Convert to rotation matrix
                r00 = 1 - 2*(qy*qy + qz*qz)
                r01 = 2*(qx*qy - qz*qw)
                r02 = 2*(qx*qz + qy*qw)
                r10 = 2*(qx*qy + qz*qw)
                r11 = 1 - 2*(qx*qx + qz*qz)
                r12 = 2*(qy*qz - qx*qw)
                r20 = 2*(qx*qz - qy*qw)
                r21 = 2*(qy*qz + qx*qw)
                r22 = 1 - 2*(qx*qx + qy*qy)
                kitti_lines.append(f"{r00:.9f} {r01:.9f} {r02:.9f} {tx:.9f} {r10:.9f} {r11:.9f} {r12:.9f} {ty:.9f} {r20:.9f} {r21:.9f} {r22:.9f} {tz:.9f}")

        if kitti_lines:
            output_traj.write_text("\n".join(kitti_lines) + "\n")
            print(f"  Saved {len(kitti_lines)} poses to {output_traj.name}")
        else:
            print(f"  [WARN] CameraTrajectory.txt has no valid poses")
            # Check if it's the tum format
            lines = cam_traj.read_text().strip().splitlines()
            print(f"  Raw format: {lines[0][:80] if lines else 'empty'}")
    else:
        print(f"  [FAIL] No CameraTrajectory.txt produced")

print("\nDone. Checking results:")
for seq_id in sequences:
    traj = OUTPUT_DIR / f"{seq_id}.kitti.txt"
    if traj.exists():
        n = len(traj.read_text().strip().splitlines())
        print(f"  {seq_id}: {n} poses")
    else:
        print(f"  {seq_id}: NO TRAJECTORY")
