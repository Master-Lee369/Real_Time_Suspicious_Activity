import os
from pathlib import Path
import collections


def summarize(path):
    p = Path(path)
    if not p.exists():
        print('Missing path:', path)
        return
    print('Path:', path)
    for root, dirs, files in os.walk(path):
        level = Path(root).relative_to(path).parts
        indent = '  ' * (len(level))
        print(f"{indent}{Path(root).name}/ ({len(files)} files)")
        # show file type counts at top-level subfolders
        if files:
            exts = collections.Counter([os.path.splitext(f)[1].lower() for f in files])
            print(f"{indent}  types: {dict(exts)}")
        # limit recursion print depth
        # continue walking


if __name__ == '__main__':
    print('--- Activity dataset summary ---')
    summarize('ml/datasets/activity')
    print('\n--- ID card dataset summary ---')
    summarize('ml/datasets/id_card')
