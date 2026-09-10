#!/usr/bin/env bash
# Delete only the suspicious new trajectories (433 lines from batch run)
cd /mnt/c/Users/Qurban/Documents/GitHub/VLM/slam
for f in *.kitti.txt; do
  seq=$(basename "$f" .kitti.txt)
  n=$(wc -l < "$f")
  # Skip the original 5 sequences and example
  case "$seq" in
    2011_09_26_drive_0009|2011_09_26_drive_0015|2011_09_26_drive_0023|2011_09_26_drive_0036|2011_09_26_drive_0093|example*)
      echo "KEEP: $seq ($n lines)"
      ;;
    *)
      echo "DELETE: $seq ($n lines)"
      rm "$f"
      ;;
  esac
done
echo ""
echo "Remaining:"
ls *.kitti.txt 2>/dev/null | wc -l
