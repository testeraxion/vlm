#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/c/Users/Qurban/Documents/GitHub/VLM"
ORB_ROOT="$REPO_ROOT/slam/ORB_SLAM3"
VOCAB="$ORB_ROOT/Vocabulary/ORBvoc.txt"
STAMPS_ROOT="$REPO_ROOT/slam/prepared"
OUTPUT_DIR="$REPO_ROOT/slam"
PANGOLIN_LIB="/tmp/pangolin_install/lib"

export LD_LIBRARY_PATH="$PANGOLIN_LIB:$ORB_ROOT/lib:${LD_LIBRARY_PATH:-}"

for prepared_dir in "$STAMPS_ROOT"/*/; do
    seq_id=$(basename "$prepared_dir")
    output_traj="$OUTPUT_DIR/${seq_id}.kitti.txt"
    settings="$prepared_dir/settings.yaml"

    if [ -f "$output_traj" ]; then
        n_lines=$(wc -l < "$output_traj")
        if [ "$n_lines" -gt 10 ]; then
            echo "[SKIP] $seq_id ($n_lines lines)"
            continue
        fi
    fi

    if [ ! -f "$settings" ]; then
        echo "[SKIP] $seq_id: no settings.yaml"
        continue
    fi

    echo -n "[RUN] $seq_id... "
    rm -f "$ORB_ROOT/CameraTrajectory.txt"

    cd "$ORB_ROOT"
    timeout 600 ./Examples/Stereo/stereo_kitti "$VOCAB" "$settings" "$prepared_dir" 2>/dev/null

    cam_traj="$ORB_ROOT/CameraTrajectory.txt"
    if [ ! -f "$cam_traj" ]; then
        echo "FAIL (no output)"
        continue
    fi

    n=$(wc -l < "$cam_traj")
    cp "$cam_traj" "$output_traj"
    echo "OK ($n poses)"
done

echo ""
echo "=== Summary ==="
for prepared_dir in "$STAMPS_ROOT"/*/; do
    seq_id=$(basename "$prepared_dir")
    traj="$OUTPUT_DIR/${seq_id}.kitti.txt"
    if [ -f "$traj" ]; then
        n=$(wc -l < "$traj")
        printf "  %-45s %4d poses\n" "$seq_id" "$n"
    else
        printf "  %-45s MISSING\n" "$seq_id"
    fi
done
