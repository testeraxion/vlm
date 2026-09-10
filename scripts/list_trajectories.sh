#!/usr/bin/env bash
for f in /mnt/c/Users/Qurban/Documents/GitHub/VLM/slam/*.kitti.txt; do
  n=$(wc -l < "$f")
  name=$(basename "$f" .kitti.txt)
  printf "%-45s %4d\n" "$name" "$n"
done
