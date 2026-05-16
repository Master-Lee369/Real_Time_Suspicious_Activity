"""
Copy the most recently downloaded ID card images into the activity raw folder.

By default this script copies everything from:
  ml/datasets/id_card/raw/files
to:
  ml/datasets/activity/raw/ID_CARDS_last_download

It creates the destination if missing and writes a small index file with counts.
"""
import shutil
from pathlib import Path
import time


SRC = Path('ml/datasets/id_card/raw/files')
DST_BASE = Path('ml/datasets/activity/raw')
DST = DST_BASE / f'ID_CARDS_last_download'


def copy_all():
    if not SRC.exists():
        print('Source folder not found:', SRC)
        return
    DST.mkdir(parents=True, exist_ok=True)
    files = [p for p in SRC.iterdir() if p.is_file()]
    copied = 0
    for f in files:
        dest = DST / f.name
        shutil.copy2(f, dest)
        copied += 1
    idx = DST / 'INDEX.txt'
    with open(idx, 'w') as fh:
        fh.write(f'copied_at: {time.ctime()}\n')
        fh.write(f'count: {copied}\n')
        fh.write('files:\n')
        for f in files:
            fh.write(f' - {f.name}\n')
    print(f'Copied {copied} files to {DST}')


if __name__ == '__main__':
    copy_all()
