#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/c/Users/Qurban/Documents/GitHub/VLM"
ORB_ROOT="$REPO_ROOT/slam/ORB_SLAM3"
VOCAB="$ORB_ROOT/Vocabulary/ORBvoc.txt"
OUTPUT_DIR="$REPO_ROOT/slam"
export LD_LIBRARY_PATH="/tmp/pangolin_install/lib:$ORB_ROOT/lib"

SEQUENCES=(
    "2011_10_03_drive_0042_sync"
    "2011_10_03_drive_0047_sync"
)

for seq_id in "${SEQUENCES[@]}"; do
    PREPARED="$REPO_ROOT/slam/prepared/$seq_id"
    OUTPUT="$OUTPUT_DIR/${seq_id}.kitti.txt"
    
    if [ -f "$OUTPUT" ]; then
        echo "[SKIP] $seq_id"
        continue
    fi
    
    echo -n "[RUN] $seq_id... "
    rm -f "$ORB_ROOT/CameraTrajectory.txt"
    cd "$ORB_ROOT"
    timeout 600 ./Examples/Stereo/stereo_kitti "$VOCAB" "$PREPARED/settings.yaml" "$PREPARED" 2>/dev/null
    
    if [ -f "$ORB_ROOT/CameraTrajectory.txt" ]; then
        n=$(wc -l < "$ORB_ROOT/CameraTrajectory.txt")
        cp "$ORB_ROOT/CameraTrajectory.txt" "$OUTPUT"
        echo "OK ($n poses)"
    else
        echo "FAIL"
    fi
done
