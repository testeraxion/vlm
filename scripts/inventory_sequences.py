"""Inventory available KITTI sequences."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_language_localization.data.kitti_loader import discover_drive_roots

dataset = Path("dataset")
drives = discover_drive_roots(dataset)

print(f"Total drive roots found: {len(drives)}")
print()

for d in drives:
    # Extract drive ID from path
    parts = d.parts
    drive_name = parts[-1] if parts[-1].endswith("_sync") else parts[-2]
    oxts = d / "oxts" / "data"
    img = d / "image_02" / "data"
    n_oxts = len(list(oxts.glob("*.txt"))) if oxts.exists() else 0
    n_img = len(list(img.glob("*.png"))) if img.exists() else 0
    usable = n_oxts > 0 and n_img > 0
    marker = " *" if usable else ""
    print(f"  {drive_name}: {n_oxts} frames, {n_img} images{marker}")

print()
usable_count = sum(1 for d in drives if (d / "oxts" / "data").exists() and len(list((d / "oxts" / "data").glob("*.txt"))) > 0)
print(f"Total usable sequences: {usable_count}")
