"""
Merge multiple raw activity datasets into a clean merged dataset with two classes:
  ml/datasets/activity/merged/normal/
  ml/datasets/activity/merged/suspicious/

Rules:
- Classify by matching keywords in file path/name against mapping lists.
- Skip unreadable/corrupted videos and files without label keywords (do not fake labels).
- Rename copied files to avoid duplicates: use incremental counters per class.
- Produce report at ml/datasets/activity/activity_dataset_report.txt

Run: python ml/training/merge_activity_datasets.py
"""
import os
from pathlib import Path
import shutil
import re
import cv2
import time

RAW_ROOT = Path('ml/datasets/activity/raw')
MERGED_ROOT = Path('ml/datasets/activity/merged')
REPORT_PATH = Path('ml/datasets/activity/activity_dataset_report.txt')

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

SUSPICIOUS_KEYWORDS = {"fight","fighting","violence","abnormal","anomaly","assault","abuse","arrest","arson","burglary","explosion","robbery","shooting","stealing","theft","vandalism","shoplifting"}
NORMAL_KEYWORDS = {"normal","nonviolence","non-violence","nofight","no_fight","neutral","regular","nonviolence"}


def is_video(path: Path):
    return path.suffix.lower() in VIDEO_EXTS


def probe_video(path: Path):
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return bool(ret)
    except Exception:
        return False


def classify_by_keywords(path_str: str):
    s = path_str.lower()
    # check suspicious first
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in s:
            return 'suspicious'
    for kw in NORMAL_KEYWORDS:
        if kw in s:
            return 'normal'
    return None


def safe_copy(src: Path, dest_dir: Path, counter: dict, src_root: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = src.stem
    ext = src.suffix
    # create a safe name including a short source folder name and counter
    src_rel = src.relative_to(src_root)
    # compress path to single token
    token = '_'.join(src_rel.parts[:-1]) if len(src_rel.parts) > 1 else src_rel.parts[0]
    token = re.sub(r'[^0-9a-zA-Z]+', '_', token)[:64]
    key = dest_dir.name
    idx = counter.get(key, 0) + 1
    counter[key] = idx
    new_name = f"{token}_{idx}{ext}"
    dest = dest_dir / new_name
    shutil.copy2(src, dest)
    return dest


def main():
    report = []
    merged_counts = {'normal':0, 'suspicious':0}
    skipped = []
    skipped_unlabeled = []
    per_source_counts = {}
    counter = {}

    if not RAW_ROOT.exists():
        print('Raw activity folder not found:', RAW_ROOT)
        return

    for root, dirs, files in os.walk(RAW_ROOT):
        root_p = Path(root)
        rel_root = root_p.relative_to(RAW_ROOT)
        src_label = str(rel_root)
        src_total = 0
        src_normal = 0
        src_susp = 0
        for f in files:
            src_total += 1
            p = root_p / f
            if not is_video(p):
                continue
            # probe readability
            ok = probe_video(p)
            if not ok:
                skipped.append(str(p))
                continue
            cls = classify_by_keywords(str(p) + ' ' + str(root_p))
            if cls is None:
                skipped_unlabeled.append(str(p))
                continue
            dest_dir = MERGED_ROOT / cls
            dest = safe_copy(p, dest_dir, counter, RAW_ROOT)
            merged_counts[cls] += 1
            if cls=='normal':
                src_normal += 1
            else:
                src_susp += 1

        per_source_counts[src_label] = {'total_files': src_total, 'normal': src_normal, 'suspicious': src_susp}

    # write report
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as fh:
        fh.write(f'Report generated: {time.ctime()}\n')
        fh.write(f'Raw root: {RAW_ROOT}\n\n')
        fh.write('Per-source summary:\n')
        for src, vals in per_source_counts.items():
            fh.write(f'  Source: {src} - total: {vals["total_files"]}, normal: {vals["normal"]}, suspicious: {vals["suspicious"]}\n')
        fh.write('\nMerged counts:\n')
        fh.write(f"  normal: {merged_counts['normal']}\n")
        fh.write(f"  suspicious: {merged_counts['suspicious']}\n")
        fh.write('\nSkipped (unreadable):\n')
        for s in skipped:
            fh.write('  '+s+'\n')
        fh.write('\nSkipped (unlabeled - no keyword match):\n')
        for s in skipped_unlabeled:
            fh.write('  '+s+'\n')

    # print quick summary
    print('Merged complete.')
    print('Normal:', merged_counts['normal'])
    print('Suspicious:', merged_counts['suspicious'])
    if skipped:
        print('Skipped unreadable videos:', len(skipped))
    if skipped_unlabeled:
        print('Skipped unlabeled videos (no keyword match):', len(skipped_unlabeled))


if __name__ == '__main__':
    main()
