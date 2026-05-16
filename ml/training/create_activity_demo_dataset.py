#!/usr/bin/env python3
"""
Create a small balanced demo dataset by sampling from ml/datasets/activity.
Writes selected videos to ml/datasets/activity_demo/<split>/<class>/ and
creates a demo_dataset_report.txt summarizing selections and skipped files.

Usage:
    python ml/training/create_activity_demo_dataset.py

Do not modify original files; script copies videos only.
"""
import argparse
import random
import shutil
import time
from pathlib import Path
import json
import cv2
import os

VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mpg', '.mpeg', '.mkv'}

DEFAULT_LIMITS = {
    'train': {'normal': 300, 'suspicious': 300},
    'val': {'normal': 60, 'suspicious': 60},
    'test': {'normal': 60, 'suspicious': 60},
}


def is_video_readable(path: Path) -> bool:
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return bool(ret)
    except Exception:
        return False


def gather_candidates(split_dir: Path, cls: str):
    folder = split_dir / cls
    if not folder.exists() or not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT and p.is_file()]
    return files


def safe_copy(src: Path, dst_dir: Path, prefix: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = src.name
    candidate = f"{prefix}_{base}"
    dst = dst_dir / candidate
    i = 1
    # ensure no overwrite
    while dst.exists():
        candidate = f"{prefix}_{i}_{base}"
        dst = dst_dir / candidate
        i += 1
    shutil.copy2(str(src), str(dst))
    return dst


def create_demo_dataset(src_root: Path, out_root: Path, limits: dict):
    report = {
        'timestamp': time.asctime(),
        'requested_limits': limits,
        'selected': {},
        'skipped': [],
        'notes': [],
    }

    for split in ('train', 'val', 'test'):
        report['selected'][split] = {}
        split_src = src_root / split
        for cls in ('normal', 'suspicious'):
            want = limits.get(split, {}).get(cls, 0)
            report['selected'][split][cls] = {'requested': want, 'selected': 0, 'available': 0}

            candidates = gather_candidates(split_src, cls)
            report['selected'][split][cls]['available'] = len(candidates)
            if len(candidates) == 0:
                report['notes'].append(f'No candidate files found for {split}/{cls} in {split_src}')
                continue

            random.shuffle(candidates)
            selected_count = 0
            dst_dir = out_root / split / cls
            prefix_base = f"{split}_{cls}"

            for src_path in candidates:
                if selected_count >= want:
                    break
                # verify readability
                if not is_video_readable(src_path):
                    report['skipped'].append({'path': str(src_path), 'reason': 'unreadable'})
                    continue
                # copy safely
                try:
                    dest = safe_copy(src_path, dst_dir, prefix_base)
                except Exception as e:
                    report['skipped'].append({'path': str(src_path), 'reason': f'copy_failed: {e}'})
                    continue
                selected_count += 1

            report['selected'][split][cls]['selected'] = selected_count

            if selected_count < want:
                report['notes'].append(f'Could only select {selected_count}/{want} for {split}/{cls} (available {len(candidates)})')

    return report


def write_report(report: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"Demo dataset report - {report.get('timestamp')}\n")
    lines.append('Requested limits:')
    lines.append(json.dumps(report.get('requested_limits', {}), indent=2))
    lines.append('\nSelected counts:')
    lines.append(json.dumps(report.get('selected', {}), indent=2))
    lines.append('\nSkipped files:')
    if report.get('skipped'):
        for s in report['skipped']:
            lines.append(f"- {s.get('path')}  ({s.get('reason')})")
    else:
        lines.append('None')
    if report.get('notes'):
        lines.append('\nNotes:')
        for n in report['notes']:
            lines.append(f"- {n}")

    out_path.write_text('\n'.join(lines), encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='ml/datasets/activity', help='source activity dataset root')
    parser.add_argument('--out', type=str, default='ml/datasets/activity_demo', help='output demo dataset root')
    parser.add_argument('--train', type=int, default=DEFAULT_LIMITS['train']['normal'], help='per-class train limit')
    parser.add_argument('--val', type=int, default=DEFAULT_LIMITS['val']['normal'], help='per-class val limit')
    parser.add_argument('--test', type=int, default=DEFAULT_LIMITS['test']['normal'], help='per-class test limit')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    src_root = Path(args.src)
    out_root = Path(args.out)

    limits = {
        'train': {'normal': args.train, 'suspicious': args.train},
        'val': {'normal': args.val, 'suspicious': args.val},
        'test': {'normal': args.test, 'suspicious': args.test},
    }

    report = create_demo_dataset(src_root, out_root, limits)

    report_path = out_root / 'demo_dataset_report.txt'
    write_report(report, report_path)

    print('Demo dataset creation finished. Report saved to', report_path)


if __name__ == '__main__':
    main()
