"""Repair YOLO ID card labels by clipping boxes to image bounds.

Creates a backup of `ml/datasets/id_card/labels/` at
`ml/datasets/id_card/labels_backup_before_repair/` before modifying files.

For each label line `class x y w h` (YOLO normalized):
- converts to pixel box, clips to image bounds,
- removes boxes that become too small,
- converts back to normalized format and writes label files.

Produces `ml/datasets/id_card/label_repair_report.txt` and runs
`ml/training/verify_yolo_idcard_dataset.py` at the end.

Usage:
    python ml/training/repair_yolo_idcard_labels.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import cv2
    import numpy as np
except Exception:
    print("This script requires OpenCV and numpy. Install with: pip install opencv-python numpy")
    raise


ROOT = Path("ml/datasets/id_card")
BACKUP = ROOT / "labels_backup_before_repair"
REPORT = ROOT / "label_repair_report.txt"
IMAGE_SUBSETS = ("train", "val", "test")
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_matching_image(img_dir: Path, stem: str) -> Path | None:
    for ext in SUPPORTED_EXT:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    # case-insensitive search fallback
    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        if p.stem.lower() == stem.lower() and p.suffix.lower() in SUPPORTED_EXT:
            return p
    return None


def clip_and_normalize(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    # convert normalized to pixel
    x_c = cx * img_w
    y_c = cy * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = x_c - bw / 2.0
    y1 = y_c - bh / 2.0
    x2 = x_c + bw / 2.0
    y2 = y_c + bh / 2.0

    # clip
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))

    # ensure x2>x1 and y2>y1
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0

    bw2 = x2 - x1
    bh2 = y2 - y1

    # minimal box size (pixels)
    if bw2 < 2 or bh2 < 2:
        return 0.0, 0.0, 0.0, 0.0

    # convert back to normalized cx,cy,w,h
    ncx = (x1 + x2) / 2.0 / img_w
    ncy = (y1 + y2) / 2.0 / img_h
    nw = bw2 / img_w
    nh = bh2 / img_h

    # clamp to [0,1]
    ncx = max(0.0, min(ncx, 1.0))
    ncy = max(0.0, min(ncy, 1.0))
    nw = max(0.0, min(nw, 1.0))
    nh = max(0.0, min(nh, 1.0))

    return ncx, ncy, nw, nh


def backup_labels(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"Backup directory already exists: {dst}. Writing into it anyway.")
    else:
        dst.mkdir(parents=True, exist_ok=True)
    # copy entire labels tree
    for subset in IMAGE_SUBSETS:
        src_dir = src / "labels" / subset
        if not src_dir.exists():
            continue
        dst_dir = dst / subset
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)


def repair_labels(root: Path) -> Tuple[int, int, int, int]:
    total_files = 0
    total_repaired_lines = 0
    total_removed_lines = 0
    missing_pairs = 0

    for subset in IMAGE_SUBSETS:
        label_dir = root / "labels" / subset
        img_dir = root / "images" / subset
        if not label_dir.exists():
            continue
        for label_file in label_dir.iterdir():
            if not label_file.is_file() or label_file.suffix.lower() != ".txt":
                continue
            total_files += 1
            stem = label_file.stem
            img_path = find_matching_image(img_dir, stem)
            if img_path is None:
                missing_pairs += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                missing_pairs += 1
                continue
            img_h, img_w = img.shape[:2]

            lines = label_file.read_text(encoding="utf-8").strip().splitlines()
            new_lines: List[str] = []
            repaired_here = 0
            removed_here = 0
            for ln in lines:
                if not ln.strip():
                    continue
                parts = ln.strip().split()
                if len(parts) != 5:
                    removed_here += 1
                    continue
                try:
                    cls = int(parts[0])
                    vals = [float(x) for x in parts[1:]]
                except Exception:
                    removed_here += 1
                    continue
                # enforce class id 0
                if cls != 0:
                    # keep but set to 0
                    cls = 0

                ncx, ncy, nw, nh = clip_and_normalize(vals[0], vals[1], vals[2], vals[3], img_w, img_h)
                if nw == 0.0 or nh == 0.0:
                    removed_here += 1
                    continue

                new_lines.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                # count as repaired if values changed significantly
                if (abs(ncx - vals[0]) > 1e-6) or (abs(ncy - vals[1]) > 1e-6) or (abs(nw - vals[2]) > 1e-6) or (abs(nh - vals[3]) > 1e-6):
                    repaired_here += 1

            # write back
            label_file.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
            total_repaired_lines += repaired_here
            total_removed_lines += removed_here

    return total_files, total_repaired_lines, total_removed_lines, missing_pairs


def main(argv=None):
    root = ROOT
    if not root.exists():
        print(f"Dataset root not found: {root}")
        return 1

    # create backup
    backup_labels(root, BACKUP)

    # repair
    total_files, repaired, removed, missing_pairs = repair_labels(root)

    # write report
    lines = [
        f"total_label_files_checked: {total_files}",
        f"labels_repaired_lines: {repaired}",
        f"labels_removed_lines: {removed}",
        f"missing_image_label_pairs: {missing_pairs}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Repair complete.")
    print(REPORT)

    # run verifier automatically
    verifier = Path("ml/training/verify_yolo_idcard_dataset.py")
    if verifier.exists():
        print("Running verifier after repair...")
        proc = subprocess.run([sys.executable, str(verifier)], capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            print("Verifier exited with errors. See output above.")
    else:
        print("Verifier script not found. Please run: python ml/training/verify_yolo_idcard_dataset.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
