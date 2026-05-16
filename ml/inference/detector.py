"""
ID card detector wrapper. Uses ultralytics YOLO (v8/yolov5). If model not present, sets `model_missing` flag.
Provides `detect_image(image)` returning detections list.
"""
import os


class IDDetector:
    def __init__(self, weights_path='ml/models/id_card_yolov5/best.pt'):
        self.weights_path = weights_path
        self.model = None
        self.model_missing = not os.path.exists(weights_path)
        if not self.model_missing:
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights_path)
            except Exception as e:
                print('Failed to load ultralytics YOLO model:', e)
                self.model_missing = True

    def detect_image(self, img):
        """Return list of detections with (x1,y1,x2,y2,conf,class_name) or empty list."""
        if self.model_missing or self.model is None:
            return []
        res = self.model(img)
        dets = []
        for r in res:
            for d in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = d
                dets.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'class': int(cls)})
        return dets
