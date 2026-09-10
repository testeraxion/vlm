#!/usr/bin/env bash
for f in /mnt/c/Users/Qurban/Documents/GitHub/VLM/slam/*.kitti.txt; do
  n=$(wc -l < "$f")
  name=$(basename "$f" .kitti.txt)
  echo "$name: $n"
done
