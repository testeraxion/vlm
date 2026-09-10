#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLAM_ROOT="$REPO_ROOT/slam"
PANGOLIN_ROOT="$SLAM_ROOT/Pangolin"
ORB_ROOT="$SLAM_ROOT/ORB_SLAM3"
LOCAL_PREFIX="$SLAM_ROOT/local"

require_pkg() {
  local pkg="$1"
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "Missing required WSL package: $pkg"
    echo "Install it with: sudo apt-get update && sudo apt-get install -y $pkg"
    exit 1
  fi
}

require_pkg build-essential
require_pkg cmake
require_pkg git
require_pkg libopencv-dev
require_pkg libeigen3-dev
require_pkg libepoxy-dev

mkdir -p "$LOCAL_PREFIX"

if [ ! -d "$PANGOLIN_ROOT" ]; then
  echo "Missing Pangolin source at $PANGOLIN_ROOT"
  exit 1
fi

if [ ! -d "$ORB_ROOT" ]; then
  echo "Missing ORB_SLAM3 source at $ORB_ROOT"
  exit 1
fi

echo "Building Pangolin into $LOCAL_PREFIX"
cmake -S "$PANGOLIN_ROOT" -B "$PANGOLIN_ROOT/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LOCAL_PREFIX" \
  -DBUILD_PANGOLIN_PYTHON=OFF
cmake --build "$PANGOLIN_ROOT/build" -j"$(nproc)"
cmake --install "$PANGOLIN_ROOT/build"

echo "Building ORB_SLAM3 with local Pangolin"
cmake -S "$ORB_ROOT" -B "$ORB_ROOT/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LOCAL_PREFIX"
cmake --build "$ORB_ROOT/build" -j"$(nproc)"

echo "ORB_SLAM3 build complete"
