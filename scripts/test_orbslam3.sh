#!/usr/bin/env bash
# Test run on a single small sequence to debug
REPO_ROOT="/mnt/c/Users/Qurban/Documents/GitHub/VLM"
ORB_ROOT="$REPO_ROOT/slam/ORB_SLAM3"
VOCAB="$ORB_ROOT/Vocabulary/ORBvoc.txt"
SEQ="2011_09_28_drive_0047_sync"
PREPARED="$REPO_ROOT/slam/prepared/$SEQ"

export LD_LIBRARY_PATH="/tmp/pangolin_install/lib:$ORB_ROOT/lib"

cd "$ORB_ROOT"
echo "Running ORB-SLAM3 on $SEQ (31 frames)..."
timeout 300 ./Examples/Stereo/stereo_kitti "$VOCAB" "$PREPARED/settings.yaml" "$PREPARED" 2>&1 | tail -30

echo ""
echo "CameraTrajectory.txt:"
if [ -f CameraTrajectory.txt ]; then
    echo "Lines: $(wc -l < CameraTrajectory.txt)"
    head -2 CameraTrajectory.txt
else
    echo "NOT FOUND"
fi
