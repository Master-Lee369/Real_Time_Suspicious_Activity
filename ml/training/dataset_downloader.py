"""
Dataset downloader helpers.
Supports Kaggle datasets (requires kaggle.json) and Roboflow projects.
Usage examples:
- python dataset_downloader.py --kaggle username/dataset-name --out ml/datasets/id_card
- python dataset_downloader.py --roboflow "workspace/project" --api-key YOUR_KEY --out ml/datasets/id_card

Note: you must configure Kaggle credentials at ~/.kaggle/kaggle.json
Roboflow usage requires an API key; see https://roboflow.com for instructions.
"""
import argparse
import os
import zipfile
import shutil


def download_kaggle(dataset, out_dir):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading Kaggle dataset {dataset} to {out_dir}...")
    api.dataset_download_files(dataset, path=out_dir, unzip=True)
    print("Done.")


def download_roboflow(project_path, api_key, out_dir, version=None):
    # This is an example; Roboflow recommends using their Python package or REST API
    import requests
    os.makedirs(out_dir, exist_ok=True)
    workspace, project = project_path.split('/')
    version_part = f"/versions/{version}" if version else ""
    url = f"https://api.roboflow.com/dataset/{workspace}/{project}{version_part}/download"
    params = {'api_key': api_key}
    print(f"Roboflow downloading from {url} to {out_dir}... (may require manual steps)")
    # For many projects, Roboflow returns a redirect or link; follow their instructions.
    print("Please follow Roboflow instructions and export to YOLOv5 format into the target folder.")


def ensure_yolo_structure(base_dir):
    """Create id_card dataset structure expected by training scripts."""
    os.makedirs(os.path.join(base_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'test'), exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', help='Kaggle dataset identifier (owner/dataset)')
    parser.add_argument('--roboflow', help='Roboflow workspace/project')
    parser.add_argument('--api-key', help='Roboflow API key')
    parser.add_argument('--out', default='ml/datasets/id_card', help='Output dataset folder')
    args = parser.parse_args()

    if args.kaggle:
        download_kaggle(args.kaggle, args.out)
        ensure_yolo_structure(args.out)
    elif args.roboflow:
        download_roboflow(args.roboflow, args.api_key, args.out)
        ensure_yolo_structure(args.out)
    else:
        print('Provide --kaggle or --roboflow to download a dataset.')
