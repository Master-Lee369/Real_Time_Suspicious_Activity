"""Prepare ID card dataset folder structure.

Usage: run from repository root:
    python ml/training/setup_idcard_structure.py

This script will:
- Collect images from `ml/datasets/id_card/raw/` (recursive)
- Ensure unique filenames by renaming duplicates safely
- Split into train/val/test (70/20/10)
- Copy images into `ml/datasets/id_card/images/{train,val,test}`
- Create empty label folders `ml/datasets/id_card/labels/{train,val,test}`
- Write `ml/datasets/id_card/data.yaml` (nc:1, names:['id_card'])
- Create report `ml/datasets/id_card/idcard_dataset_report.txt`

It will NOT create any label files or modify Django code.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(src: Path) -> List[Tuple[Path, str, bool]]:
    """Recursively collect image files from src.

    Returns list of tuples (path, unique_name, has_label)
    where unique_name is a filename (with ext) made unique among collected files.
    has_label is True if a .txt with same stem exists alongside the image.
    """
    files: List[Path] = []
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)

    used = {}
    out = []
    for p in files:
        name = p.name
        stem = p.stem
        ext = p.suffix
        key = name.lower()
        if key in used:
            used[key] += 1
            # create new name like stem_2.ext
            new_name = f"{stem}_{used[key]}{ext}"
            # ensure this new name isn't already used (rare)
            while new_name.lower() in used:
                used[key] += 1
                new_name = f"{stem}_{used[key]}{ext}"
            used[new_name.lower()] = 1
            unique_name = new_name
        else:
            used[key] = 1
            unique_name = name

        # check for label file alongside the image
        label_path = p.with_suffix(".txt")
        has_label = label_path.exists()

        out.append((p, unique_name, has_label))

    return out


def split_list(items: List, ratios=(0.7, 0.2, 0.1), seed: int | None = 42):
    if seed is not None:
        random.seed(seed)
    items = items.copy()
    random.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    return (
        items[:n_train],
        items[n_train : n_train + n_val],
        items[n_train + n_val :],
    )


def ensure_dirs(root: Path):
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "test").mkdir(parents=True, exist_ok=True)


def write_data_yaml(root: Path):
    data = """train: images/train
val: images/val
test: images/test
nc: 1
names: ['id_card']
"""
    (root / "data.yaml").write_text(data, encoding="utf-8")


def write_report(root: Path, total: int, train: int, val: int, test: int, missing_labels: int):
    rpt = []
    rpt.append(f"Total images: {total}")
    rpt.append(f"Train: {train}")
    rpt.append(f"Val: {val}")
    rpt.append(f"Test: {test}")
    rpt.append(f"Images with missing labels: {missing_labels}")
    text = "\n".join(rpt) + "\n"
    (root / "idcard_dataset_report.txt").write_text(text, encoding="utf-8")


def copy_items(items: List[Tuple[Path, str, bool]], dst_dir: Path) -> int:
    """Copy list of tuples (srcpath, unique_name, has_label) into dst_dir images folder.

    Returns number of items copied.
    """
    count = 0
    for src, uniq_name, _has_label in items:
        dst = dst_dir / uniq_name
        # copy preserving metadata
        shutil.copy2(src, dst)
        count += 1
    return count


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="ml/datasets/id_card/raw", help="Source raw folder")
    p.add_argument("--dst", default="ml/datasets/id_card", help="Target id_card folder")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = p.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"Source folder does not exist: {src}")
        return 1

    ensure_dirs(dst)

    collected = collect_images(src)
    total = len(collected)
    if total == 0:
        print("No supported image files found. Exiting.")
        return 0

    # count missing labels
    missing_labels = sum(1 for _p, _n, has_label in collected if not has_label)

    train_items, val_items, test_items = split_list(collected, seed=args.seed)

    n_train = copy_items(train_items, dst / "images" / "train")
    n_val = copy_items(val_items, dst / "images" / "val")
    n_test = copy_items(test_items, dst / "images" / "test")

    write_data_yaml(dst)
    write_report(dst, total, n_train, n_val, n_test, missing_labels)

    # Print required summary
    print(f"Total images found: {total}")
    print(f"Train images copied: {n_train}")
    print(f"Val images copied: {n_val}")
    print(f"Test images copied: {n_test}")
    print(f"Report written to: {dst / 'idcard_dataset_report.txt'}")
    print(f"YAML written to: {dst / 'data.yaml'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
Prepare ID card dataset folders for YOLO training.
This script will:
 - create `ml/datasets/id_card/images/{train,val,test}`
 - create `ml/datasets/id_card/labels/{train,val,test}` (empty)
 - copy images from `ml/datasets/id_card/raw/files` into images with a 80/10/10 split
 - create `ml/datasets/id_card/data.yaml` describing the dataset

It does NOT create annotation labels. Use LabelImg or Roboflow to annotate images and place `.txt` label files into the labels/... folders (one .txt per image, YOLO format).
"""
import os
import shutil
from pathlib import Path
import random


def ensure_dirs(base):
    for sub in ('images','labels'):
        for split in ('train','val','test'):
            d = base / sub / split
            d.mkdir(parents=True, exist_ok=True)


def split_and_copy(src_files, dst_base, ratios=(0.8,0.1,0.1)):
    random.seed(42)
    files = list(src_files)
    random.shuffle(files)
    n = len(files)
    n1 = int(ratios[0]*n)
    n2 = int((ratios[0]+ratios[1])*n)
    splits = {
        'train': files[:n1],
        'val': files[n1:n2],
        'test': files[n2:]
    }
    for split, items in splits.items():
        for f in items:
            shutil.copy2(f, dst_base / 'images' / split / Path(f).name)


def write_data_yaml(base, names=['id_card']):
    trainp = (base / 'images' / 'train').as_posix()
    valp = (base / 'images' / 'val').as_posix()
    testp = (base / 'images' / 'test').as_posix()
    nc = len(names)
    with open(base / 'data.yaml','w') as fh:
        fh.write(f"train: {trainp}\n")
        fh.write(f"val: {valp}\n")
        fh.write(f"test: {testp}\n")
        fh.write(f"nc: {nc}\n")
        fh.write("names:\n")
        for i,n in enumerate(names):
            fh.write(f"  {i}: '{n}'\n")


if __name__ == '__main__':
    SRC = Path('ml/datasets/id_card/raw/files')
    DST = Path('ml/datasets/id_card')
    ensure_dirs(DST)
    if not SRC.exists():
        print('Source folder not found:', SRC)
        raise SystemExit(1)
    imgs = [p for p in SRC.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.png')]
    if not imgs:
        print('No images found in', SRC)
        raise SystemExit(1)
    split_and_copy(imgs, DST)
    write_data_yaml(DST)
    print('ID card structure prepared under', DST)
    print('Images counts:')
    for split in ('train','val','test'):
        cnt = len(list((DST / 'images' / split).iterdir()))
        print(f'  {split}: {cnt}')
