"""
Verify activity dataset counts and report imbalance warnings.
Prints totals and detailed counts per split.

Run: python ml/training/verify_activity_dataset.py
"""
from pathlib import Path
import math

BASE = Path('ml/datasets/activity')


def count_files(p: Path):
    if not p.exists():
        return 0
    return sum(1 for _ in p.rglob('*') if _.is_file())


def main():
    merged_normal = Path('ml/datasets/activity/merged/normal')
    merged_susp = Path('ml/datasets/activity/merged/suspicious')
    tn = count_files(merged_normal)
    ts = count_files(merged_susp)
    print('Merged totals:')
    print('  normal:', tn)
    print('  suspicious:', ts)

    splits = ['train','val','test']
    classes = ['normal','suspicious']
    totals = {}
    for s in splits:
        for c in classes:
            p = BASE / s / c
            cnt = count_files(p)
            totals[(s,c)] = cnt

    print('\nSplit counts:')
    for s in splits:
        print(f' {s}: normal={totals[(s,"normal")]}, suspicious={totals[(s,"suspicious")]}' )

    # check imbalance
    if tn == 0 or ts == 0:
        print('\nWARNING: one class has zero samples.')
        return
    ratio = max(tn,ts)/min(tn,ts)
    if ratio > 2.0:
        print(f'WARNING: class imbalance is high (ratio {ratio:.2f}:1)')
    else:
        print(f'Class balance OK (ratio {ratio:.2f}:1)')


if __name__ == '__main__':
    main()
