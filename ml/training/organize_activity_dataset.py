"""
Organize activity dataset from `ml/datasets/activity/raw` into the standard structure:

ml/datasets/activity/
  train/normal
  train/suspicious
  val/normal
  val/suspicious
  test/normal
  test/suspicious

Rules:
- If `SCVD/SCVD_converted` exists with Train/Test folders, copy Train->train and Test->test.
- If `SCVD_converted_sec_split` exists, prefer it.
- If `Files/NonViolence` and `Files/Violence` exist, include them into train/normal and train/suspicious.
- Create a small validation split (10%) taken from train copies.

This script only copies files; it does not remove originals.
"""
import os
import shutil
from pathlib import Path
import random


def ensure_dirs(base):
    for split in ('train','val','test'):
        for cls in ('normal','suspicious'):
            d = base / split / cls
            d.mkdir(parents=True, exist_ok=True)


def copy_files(file_list, dest_dir):
    for f in file_list:
        dest = dest_dir / Path(f).name
        if not dest.exists():
            shutil.copy2(f, dest)


def gather_and_copy(src_base, dest_base):
    # prefer sec_split if available
    sec = src_base / 'SCVD' / 'SCVD_converted_sec_split'
    conv = src_base / 'SCVD' / 'SCVD_converted'
    files_dir = src_base / 'Files'
    # collect lists
    train_normal = []
    train_susp = []
    test_normal = []
    test_susp = []

    if sec.exists():
        tn = sec / 'Train' / 'Normal'
        ts = sec / 'Train' / 'Violence'
        ttn = sec / 'Test' / 'Normal'
        tts = sec / 'Test' / 'Violence'
        if tn.exists():
            train_normal += [str(p) for p in tn.iterdir() if p.is_file()]
        if ts.exists():
            train_susp += [str(p) for p in ts.iterdir() if p.is_file()]
        if ttn.exists():
            test_normal += [str(p) for p in ttn.iterdir() if p.is_file()]
        if tts.exists():
            test_susp += [str(p) for p in tts.iterdir() if p.is_file()]
    elif conv.exists():
        tn = conv / 'Train' / 'Normal'
        ts = conv / 'Train' / 'Violence'
        ttn = conv / 'Test' / 'Normal'
        tts = conv / 'Test' / 'Violence'
        if tn.exists():
            train_normal += [str(p) for p in tn.iterdir() if p.is_file()]
        if ts.exists():
            train_susp += [str(p) for p in ts.iterdir() if p.is_file()]
        if ttn.exists():
            test_normal += [str(p) for p in ttn.iterdir() if p.is_file()]
        if tts.exists():
            test_susp += [str(p) for p in tts.iterdir() if p.is_file()]

    if files_dir.exists():
        nv = files_dir / 'NonViolence'
        v = files_dir / 'Violence'
        if nv.exists():
            train_normal += [str(p) for p in nv.iterdir() if p.is_file()]
        if v.exists():
            train_susp += [str(p) for p in v.iterdir() if p.is_file()]

    # copy to train/test
    copy_files(train_normal, dest_base / 'train' / 'normal')
    copy_files(train_susp, dest_base / 'train' / 'suspicious')
    copy_files(test_normal, dest_base / 'test' / 'normal')
    copy_files(test_susp, dest_base / 'test' / 'suspicious')

    # create val by sampling 10% from train if val empty
    for cls in ('normal','suspicious'):
        val_dir = dest_base / 'val' / cls
        if any(val_dir.iterdir()):
            continue
        train_dir = dest_base / 'train' / cls
        items = [p for p in train_dir.iterdir() if p.is_file()]
        k = max(1, int(0.1 * len(items)))
        if k==0:
            continue
        sample = random.sample(items, k)
        for s in sample:
            dest = val_dir / s.name
            if not dest.exists():
                shutil.copy2(s, dest)


def summarize_structure(base):
    print('Organized dataset structure under:', base)
    for split in ('train','val','test'):
        for cls in ('normal','suspicious'):
            d = base / split / cls
            cnt = sum(1 for _ in d.iterdir() if _.is_file()) if d.exists() else 0
            print(f'  {split}/{cls}: {cnt} files')


if __name__ == '__main__':
    random.seed(42)
    SRC = Path('ml/datasets/activity/raw')
    DST = Path('ml/datasets/activity')
    ensure_dirs(DST)
    gather_and_copy(SRC, DST)
    summarize_structure(DST)
