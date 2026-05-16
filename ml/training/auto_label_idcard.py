"""Auto-label ID card images with simple OpenCV-based document detection.

Creates YOLO-format label files (class 0) and preview images with boxes.
Does NOT create random boxes; creates empty .txt when no reliable rectangle found.

Usage:
    python ml/training/auto_label_idcard.py [--dst ROOT] [--preview PREVIEW_DIR] [--overwrite]

Example:
    python ml/training/auto_label_idcard.py --overwrite
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
except Exception as e:
    print("This script requires OpenCV (cv2) and numpy. Install with: pip install opencv-python numpy")
    raise


IMAGE_SUBSETS = ("train", "val", "test")


def find_image_files(root: Path) -> List[Tuple[str, Path]]:
    out = []
    for subset in IMAGE_SUBSETS:
        p = root / "images" / subset
        if not p.exists():
            continue
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                out.append((subset, f))
    return out


def detect_document_rectangle(img: np.ndarray, min_area_frac: float = 0.02) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
    """Return bounding box (x,y,w,h) and approx polygon for best document-like rectangle, or None."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    img_area = w * h
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            bx, by, bw, bh = cv2.boundingRect(approx)
            area = bw * bh
            if area < min_area_frac * img_area:
                continue
            ar = bw / float(bh) if bh > 0 else 0
            if ar <= 0.2 or ar >= 5.0:
                continue
            candidates.append((area, (bx, by, bw, bh), approx))

    if candidates:
        # pick largest area
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1], candidates[0][2]

    # fallback: try minAreaRect on largest contour
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        xs = box[:, 0]
        ys = box[:, 1]
        bx, by = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - xs.min()), int(ys.max() - ys.min())
        area = bw * bh
        if bw <= 0 or bh <= 0:
            continue
        if area >= min_area_frac * img_area:
            return (bx, by, bw, bh), box

    return None


def write_yolo_label(path: Path, class_id: int, bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int]):
    w = img_shape[1]
    h = img_shape[0]
    x, y, bw, bh = bbox
    cx = (x + bw / 2.0) / float(w)
    cy = (y + bh / 2.0) / float(h)
    nw = bw / float(w)
    nh = bh / float(h)
    line = f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
    path.write_text(line, encoding="utf-8")


def save_preview(preview_dir: Path, img_path: Path, bbox: Optional[Tuple[int, int, int, int]], poly: Optional[np.ndarray]):
    preview_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(img_path))
    if img is None:
        return
    if bbox and poly is not None:
        # draw polygon and bbox
        cv2.drawContours(img, [poly], -1, (0, 255, 0), 3)
        x, y, bw, bh = bbox
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    outp = preview_dir / img_path.name
    cv2.imwrite(str(outp), img)


def process_all(root: Path, preview_dir: Path, overwrite: bool = False) -> None:
    images = find_image_files(root)
    total = len(images)
    created = 0
    skipped = 0
    empty_created = 0
    need_manual: List[str] = []

    for subset, img_path in images:
        label_dir = root / "labels" / subset
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / (img_path.stem + ".txt")

        if label_path.exists() and label_path.stat().st_size > 0 and not overwrite:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            # create empty label and mark for review
            label_path.write_text("", encoding="utf-8")
            empty_created += 1
            need_manual.append(str(img_path.name))
            save_preview(preview_dir, img_path, None, None)
            continue

        res = detect_document_rectangle(img)
        if res is None:
            # no reliable rectangle
            label_path.write_text("", encoding="utf-8")
            empty_created += 1
            need_manual.append(str(img_path.name))
            save_preview(preview_dir, img_path, None, None)
        else:
            (bx, by, bw, bh), poly = res
            write_yolo_label(label_path, 0, (bx, by, bw, bh), img.shape)
            created += 1
            save_preview(preview_dir, img_path, (bx, by, bw, bh), poly)

    # write report
    report_path = root / "auto_label_report.txt"
    lines = [
        f"total_images_processed: {total}",
        f"labels_created: {created}",
        f"skipped_existing_labels: {skipped}",
        f"empty_labels_created: {empty_created}",
        f"images_needing_manual_review: {len(need_manual)}",
    ]
    if need_manual:
        lines.append("\n# images needing manual review:")
        for n in need_manual:
            lines.append(n)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Auto-labeling complete.")
    print(report_path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dst", default="ml/datasets/id_card", help="ID card dataset root")
    p.add_argument("--preview", default="ml/datasets/id_card/preview_labels", help="Preview output folder")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty labels")
    args = p.parse_args(argv)

    root = Path(args.dst)
    preview = Path(args.preview)
    if not root.exists():
        print(f"Dataset root not found: {root}")
        return 1

    process_all(root, preview, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
