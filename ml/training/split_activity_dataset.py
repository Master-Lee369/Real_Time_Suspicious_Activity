"""
Split merged activity dataset into train/val/test with given ratios while preserving class balance as much as possible.

Output folders (overwrite existing files if present):
 ml/datasets/activity/train/normal
 ml/datasets/activity/train/suspicious
 ml/datasets/activity/val/...
 ml/datasets/activity/test/...

Run: python ml/training/split_activity_dataset.py
"""
import os
import shutil
from pathlib import Path
import random

MERGED = Path('ml/datasets/activity/merged')
DEST = Path('ml/datasets/activity')

RATIOS = (0.7, 0.15, 0.15)  # train, val, test
SEED = 42


def clear_and_make(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def split_class(src_dir: Path, out_base: Path, ratios=RATIOS):
    files = [p for p in src_dir.iterdir() if p.is_file()]
    random.Random(SEED).shuffle(files)
    n = len(files)
    n1 = int(ratios[0]*n)
    n2 = n1 + int(ratios[1]*n)
    splits = {
        'train': files[:n1],
        'val': files[n1:n2],
        'test': files[n2:]
    }
    for split, items in splits.items():
        dest_dir = out_base / split / src_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in items:
            dest = dest_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
    return {k: len(v) for k,v in splits.items()}


def main():
    if not MERGED.exists():
        print('Merged folder not found:', MERGED)
        return
    totals = {}
    for cls in ('normal','suspicious'):
        src = MERGED / cls
        if not src.exists():
            totals[cls] = {'train':0,'val':0,'test':0}
            continue
        res = split_class(src, DEST)
        totals[cls] = res

    # print summary
    print('Split summary:')
    for cls, vals in totals.items():
        print(f'  {cls}: train={vals["train"]}, val={vals["val"]}, test={vals["test"]}')


if __name__ == '__main__':
    main()
