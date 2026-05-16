# RealTimeSecurity

Local development notes

Prerequisites:
- Python 3.11+
- pip and virtualenv

Quick start (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # if you have one
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Notes:
- The project uses SQLite for development (`db.sqlite3`). For production, use PostgreSQL and update `RealTimeSecurity/settings.py`.
- Static files are in `static/`. Media uploads use `media/`.
- Basic Bootstrap-based frontend is in `templates/`.
Next recommended steps:
- Add tests for critical behavior.
- Replace SECRET_KEY in production and set `DEBUG = False`.
- Configure a production DB (Postgres) and environment-based settings.

ML pipeline (new files under `ml/`)
 - `ml/training/dataset_downloader.py` — helper to download datasets from Kaggle or Roboflow.
 - `ml/training/train_yolo_id.py` — trains YOLO (Ultralytics) for ID card detection.
 - `ml/training/preprocess_activity_videos.py` — extracts frames and prepares sequences for activity training.
 - `ml/training/train_activity_lstm.py` — trains MobileNetV2+LSTM for activity detection, saves model and `class_names.json`.
 - `ml/inference/` — detector wrappers and `realtime_pipeline.py` to test locally.
 - `ml/utils/email_alert.py` and `ml/utils/video_utils.py` — helpers for alerts and video processing.

Quick ML setup
1. Install ML requirements:

```bash
pip install -r requirements.txt
```

2. Prepare datasets:
 - Use `dataset_downloader.py` to fetch datasets or export from Roboflow in YOLO format.
 - Ensure `ml/datasets/id_card/` has the YOLO structure (images/labels split into train/val/test) and `data.yaml` describing classes.
 - For activity, place videos under `ml/datasets/activity/train|val|test/{normal,suspicious}`.

Kaggle dataset download
-----------------------
If a dataset is hosted on Kaggle, obtain your `kaggle.json` API token (from your Kaggle account) and either place it at `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Linux/macOS), or set the `KAGGLE_CONFIG_DIR` environment variable to its folder.

Example (Windows PowerShell):

```powershell
mkdir $env:USERPROFILE\.kaggle
copy .\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
# set file permission as needed
# then download (replace <owner/dataset> with the Kaggle dataset id):
kaggle datasets download -d <owner/dataset> -p ml/datasets/id_card --unzip
```

Example (Linux/macOS):

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d <owner/dataset> -p ml/datasets/id_card --unzip
```

3. Preprocess activity videos:

```bash
python ml/training/preprocess_activity_videos.py --src ml/datasets/activity --out ml/datasets/activity_processed --seq-len 30
```

4. Train ID detector (YOLO):

Option A — run the helper script which validates labels and invokes YOLOv5 if available:

```bash
python ml/training/train_yolo_id.py
```

Option B — exact YOLOv5 command (requires cloning `https://github.com/ultralytics/yolov5` and installing its requirements):

```bash
python yolov5/train.py --img 640 --batch 16 --epochs 50 --data ml/datasets/id_card/data.yaml --weights yolov5s.pt --project ml/models/id_card_yolov5 --name idcard --exist-ok
```

5. Train activity detector:

```bash
python ml/training/train_activity_lstm.py
```

Integration notes
- Do NOT commit `ml/models/` or `ml/datasets/` (they are added to `.gitignore`).
- Django views will check for model files and display "model not trained yet" if missing.
- To enable email alerts, set environment variables `ALERT_SMTP_HOST`, `ALERT_SMTP_PORT`, `ALERT_SMTP_USER`, `ALERT_SMTP_PASS`.

If you want, I can now:
- Implement the Django-side integration (upload endpoint, live stream, model-loading safety), or
- Continue by polishing the ML training (augmentation, evaluation, graphs).
