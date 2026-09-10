#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORB_ROOT="$REPO_ROOT/slam/ORB_SLAM3"
VOCAB="$ORB_ROOT/Vocabulary/ORBvoc.txt"
SEQUENCE_ID="${1:-2011_09_26_drive_0009_sync}"
PREPARED_ROOT="$REPO_ROOT/slam/prepared/$SEQUENCE_ID"
OUTPUT_TRAJ="$REPO_ROOT/slam/${SEQUENCE_ID}.kitti.txt"

if [ ! -x "$ORB_ROOT/Examples/Stereo/stereo_kitti" ]; then
  echo "Missing ORB_SLAM3 stereo_kitti binary. Build ORB_SLAM3 first."
  exit 1
fi

if [ ! -f "$VOCAB" ]; then
  if [ -f "$ORB_ROOT/Vocabulary/ORBvoc.txt.tar.gz" ]; then
    echo "Extracting ORB vocabulary..."
    tar -xzf "$ORB_ROOT/Vocabulary/ORBvoc.txt.tar.gz" -C "$ORB_ROOT/Vocabulary"
  else
    echo "Missing ORB vocabulary file at $VOCAB"
    exit 1
  fi
fi

if [ ! -f "$PREPARED_ROOT/settings.yaml" ]; then
  echo "Missing prepared sequence at $PREPARED_ROOT"
  exit 1
fi

pushd "$ORB_ROOT" >/dev/null
export LD_LIBRARY_PATH="$REPO_ROOT/slam/local/lib:$ORB_ROOT/lib:$REPO_ROOT/slam/Pangolin/build:${LD_LIBRARY_PATH:-}"
if command -v xvfb-run >/dev/null 2>&1; then
  xvfb-run -a ./Examples/Stereo/stereo_kitti "$VOCAB" "$PREPARED_ROOT/settings.yaml" "$PREPARED_ROOT"
else
  ./Examples/Stereo/stereo_kitti "$VOCAB" "$PREPARED_ROOT/settings.yaml" "$PREPARED_ROOT"
fi
popd >/dev/null

if [ ! -f "$ORB_ROOT/CameraTrajectory.txt" ]; then
  echo "ORB_SLAM3 did not produce CameraTrajectory.txt"
  exit 1
fi

cp "$ORB_ROOT/CameraTrajectory.txt" "$OUTPUT_TRAJ"
echo "Saved trajectory to $OUTPUT_TRAJ"
