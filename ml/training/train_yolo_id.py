"""Train YOLOv5 on the ID card dataset.

This script validates the dataset (runs the verifier), then attempts to run
YOLOv5 training using a local `yolov5` repository in the project root.

If the `yolov5` repo is not present this script will print clear instructions
including the exact training command and exit.

It expects dataset at `ml/datasets/id_card/` with `data.yaml` present.
The best model will be saved (copied) to `ml/models/id_card_yolov5/best.pt`.

Important: this script does not modify Django or other project code.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run_verify(root: Path) -> Path:
    verifier = Path("ml/training/verify_yolo_idcard_dataset.py")
    if not verifier.exists():
        raise SystemExit("Verifier script not found: ml/training/verify_yolo_idcard_dataset.py")
    print("Running verifier...")
    res = subprocess.run([sys.executable, str(verifier)], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise SystemExit("Verifier failed")
    report = root / "yolo_label_report.txt"
    if not report.exists():
        raise SystemExit(f"Verifier did not produce report: {report}")
    return report


def parse_verifier_report(report_path: Path):
    txt = report_path.read_text(encoding="utf-8")
    missing = 0
    invalid = 0
    for line in txt.splitlines():
        if line.startswith("missing_label_files:"):
            missing = int(line.split(":", 1)[1].strip())
        if line.startswith("invalid_label_files:"):
            invalid = int(line.split(":", 1)[1].strip())
    return missing, invalid, txt


def find_yolov5_repo(root: Path) -> Path | None:
    yolov5 = root / "yolov5"
    if yolov5.exists() and yolov5.is_dir():
        return yolov5
    return None


def run_yolov5_train(root: Path, epochs: int = 50, img_size: int = 640, batch: int = 16):
    # project/name configuration
    project = Path("ml/models/id_card_yolov5")
    name = "idcard"
    project.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "yolov5/train.py",
        "--img",
        str(img_size),
        "--batch",
        str(batch),
        "--epochs",
        str(epochs),
        "--data",
        "ml/datasets/id_card/data.yaml",
        "--weights",
        "yolov5s.pt",
        "--project",
        str(project),
        "--name",
        name,
        "--exist-ok",
    ]

    print("Running YOLOv5 train command:")
    print(" ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("YOLOv5 training failed. See output above.")

    # attempt to extract metrics from stdout
    out = proc.stdout + "\n" + proc.stderr
    precision = recall = map50 = map5095 = None
    m = re.search(r"Precision[:\s]+([0-9.]+)", out)
    if m:
        precision = float(m.group(1))
    m = re.search(r"Recall[:\s]+([0-9.]+)", out)
    if m:
        recall = float(m.group(1))
    m = re.search(r"mAP@0.5[:\s]+([0-9.]+)", out)
    if m:
        map50 = float(m.group(1))
    m = re.search(r"mAP@0.5:0.95[:\s]+([0-9.]+)", out) or re.search(r"mAP@0.5-0.95[:\s]+([0-9.]+)", out)
    if m:
        map5095 = float(m.group(1))

    # copy best.pt to desired location
    best_src = project / name / "weights" / "best.pt"
    target = project / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, target)
        print(f"Best model copied to: {target}")
    else:
        print(f"Warning: best.pt not found at expected location: {best_src}")

    print("Training metrics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"mAP50: {map50}")
    print(f"mAP50-95: {map5095}")


def main(argv=None):
    root = Path("ml/datasets/id_card")
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    report = run_verify(root)
    missing, invalid, full = parse_verifier_report(report)
    if missing > 0 or invalid > 0:
        print("Dataset verification failed. See verifier report below:\n")
        print(full)
        raise SystemExit("Aborting training due to missing or invalid labels.")

    project_root = Path.cwd()
    yolov5 = find_yolov5_repo(project_root)
    if yolov5 is None:
        print("YOLOv5 repository not found in project root.")
        print("To train, clone the YOLOv5 repo and install requirements, then run the command below:")
        print("")
        print("git clone https://github.com/ultralytics/yolov5.git")
        print("pip install -r yolov5/requirements.txt")
        print("")
        print("Exact training command:")
        print("python yolov5/train.py --img 640 --batch 16 --epochs 50 --data ml/datasets/id_card/data.yaml --weights yolov5s.pt --project ml/models/id_card_yolov5 --name idcard --exist-ok")
        raise SystemExit(2)

    # run training
    run_yolov5_train(project_root)


if __name__ == "__main__":
    raise SystemExit(main())
"""
Train YOLOv5/Ultralytics YOLO for ID card detection using the `ultralytics` package.
Requires `pip install ultralytics` or `pip install -r requirements.txt`.

Example:
python train_yolo_id.py --data ml/datasets/id_card/data.yaml --epochs 50 --batch 8
"""
import argparse
import os


def train(data_yaml, epochs=50, batch=8, imgsz=640, project='ml/models/id_card_yolov5', name='yolo_id'):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('ultralytics package not installed. Run: pip install ultralytics')
        raise

    os.makedirs(project, exist_ok=True)
    # Use pretrained small model as a starting point
    model = YOLO('yolov5s.pt')  # will download if not available
    print(f'Starting training: data={data_yaml} epochs={epochs}')
    model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=imgsz, project=project, name=name)
    print('Training finished. Best weights are saved under the project folder.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='ml/datasets/id_card/data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    train(args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
