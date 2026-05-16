"""Verify YOLO-format labels for the ID card dataset.

Checks pairing, format, ranges, and class ids.
Writes report to ml/datasets/id_card/yolo_label_report.txt

Usage:
    python ml/training/verify_yolo_idcard_dataset.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np
except Exception:
    print("This script requires numpy. Install with: pip install numpy")
    raise


IMAGE_SUBSETS = ("train", "val", "test")


def find_images(root: Path):
    out = []
    for subset in IMAGE_SUBSETS:
        p = root / "images" / subset
        if not p.exists():
            continue
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                out.append((subset, f))
    return out


def verify_label_file(label_path: Path) -> Tuple[bool, List[str]]:
    """Return (is_valid, list_of_errors). An empty file is considered valid but flagged earlier as missing/empty."""
    errors: List[str] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if text == "":
        # empty files are allowed (but should be reported by the verifier)
        return True, ["EMPTY"]

    lines = text.splitlines()
    for i, ln in enumerate(lines, start=1):
        parts = ln.strip().split()
        if len(parts) != 5:
            errors.append(f"line {i}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            errors.append(f"line {i}: class id not integer: {parts[0]}")
            continue
        if cls != 0:
            errors.append(f"line {i}: class id is {cls}, expected 0")
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"line {i}: x/y/w/h contain non-float values")
            continue
        if not all(0.0 <= v <= 1.0 for v in vals):
            errors.append(f"line {i}: x/y/w/h values must be between 0 and 1: {vals}")
        if vals[2] <= 0 or vals[3] <= 0:
            errors.append(f"line {i}: width/height must be > 0: {vals[2:]}")

    return (len(errors) == 0), errors


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dst", default="ml/datasets/id_card", help="ID card dataset root")
    args = p.parse_args(argv)

    root = Path(args.dst)
    images = find_images(root)

    total = len(images)
    missing_labels: List[str] = []
    invalid_labels: List[Tuple[str, List[str]]] = []
    empty_labels: List[str] = []

    for subset, img_path in images:
        label_path = root / "labels" / subset / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_labels.append(str(img_path.relative_to(root)))
            continue
        valid, errors = verify_label_file(label_path)
        if not valid:
            invalid_labels.append((str(label_path.relative_to(root)), errors))
        else:
            if errors and errors[0] == "EMPTY":
                empty_labels.append(str(label_path.relative_to(root)))

    rpt = []
    rpt.append(f"total_images: {total}")
    rpt.append(f"missing_label_files: {len(missing_labels)}")
    rpt.append(f"empty_label_files: {len(empty_labels)}")
    rpt.append(f"invalid_label_files: {len(invalid_labels)}")
    rpt.append("")
    if missing_labels:
        rpt.append("# missing label files (image relative to dataset root)")
        rpt.extend(missing_labels)
        rpt.append("")
    if empty_labels:
        rpt.append("# empty label files")
        rpt.extend(empty_labels)
        rpt.append("")
    if invalid_labels:
        rpt.append("# invalid label files and errors")
        for path, errs in invalid_labels:
            rpt.append(path)
            for e in errs:
                rpt.append(f"  - {e}")
            rpt.append("")

    outp = root / "yolo_label_report.txt"
    outp.write_text("\n".join(rpt) + "\n", encoding="utf-8")
    print("Verification complete.")
    print(outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
