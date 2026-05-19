"""ID detector wrapper (Ultralytics / YOLO).

Provides a safe API that checks for model files and lazily loads the model only when needed.
If model missing, methods return a dict: {"status":"model_missing","message":"Model not trained yet"}

This module avoids importing heavy dependencies at import time.
"""
import os
import numpy as np


class IDDetector:
    def __init__(self, weights_path='ml/models/id_card_yolov5/best.pt'):
        self.weights_path = weights_path
        self._model = None

    def is_model_available(self):
        return os.path.exists(self.weights_path)

    def _ensure_model_loaded(self):
        if self._model is not None:
            return True
        if not os.path.exists(self.weights_path):
            return False
        try:
            # Lazy import
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
            return True
        except Exception:
            self._model = None
            return False

    def detect_image(self, img):
        """Detect objects in a single image (numpy BGR or RGB).

        Returns:
          - if model missing: {"status":"model_missing","message":"Model not trained yet"}
          - if ok: {"status":"ok","detections":[{bbox:[x1,y1,x2,y2],"conf":f,"class":int}...]}
          - if error: {"status":"error","message":...}
        """
        if not self._ensure_model_loaded():
            return {"status": "model_missing", "message": "Model not trained yet"}

        try:
            # ultralytics models expect either path or numpy array (RGB)
            res = self._model(img)
            dets = []
            for r in res:
                # r.boxes.data may be a tensor-like; convert safely
                try:
                    data = r.boxes.data.tolist()
                except Exception:
                    data = []
                for d in data:
                    x1, y1, x2, y2, conf, cls = d
                    dets.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "conf": float(conf), "class": int(cls)})
            return {"status": "ok", "detections": dets}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def detect_from_video_path(self, video_path, max_frames=30, resize=None):
        """Simple OpenCV reader that samples up to `max_frames` frames and runs detection on them.

        Returns a dict as in `detect_image`, plus frame index information.
        """
        if not os.path.exists(video_path):
            return {"status": "error", "message": "Video file not found"}

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            detections = []
            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if resize:
                    frame = cv2.resize(frame, resize)
                out = self.detect_image(frame)
                detections.append({"frame": frame_idx, "result": out})
                frame_idx += 1
            cap.release()
            return {"status": "ok", "frames_processed": frame_idx, "results": detections}
        except Exception as e:
            return {"status": "error", "message": str(e)}
