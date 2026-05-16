"""
Activity predictor wrapper. Loads Keras model if available and predicts on sequences of frames.
If model missing, `model_missing=True` and `predict` returns None.
"""
import os
import numpy as np


class ActivityPredictor:
    def __init__(self, model_path='ml/models/activity_mobilenet_lstm/activity_model.keras', class_path='ml/models/activity_mobilenet_lstm/class_names.json'):
        self.model_path = model_path
        self.class_path = class_path
        self.model_missing = not os.path.exists(model_path)
        self.model = None
        self.class_names = []
        if not self.model_missing:
            try:
                from tensorflow.keras.models import load_model
                import json
                self.model = load_model(model_path)
                with open(class_path, 'r') as fh:
                    self.class_names = json.load(fh)
            except Exception as e:
                print('Failed to load activity model:', e)
                self.model_missing = True

    def predict_sequence(self, seq):
        """seq: numpy array shape (seq_len, H, W, C); returns (label, confidence)"""
        if self.model_missing or self.model is None:
            return None
        x = np.array(seq) / 255.0
        x = np.expand_dims(x, axis=0)
        probs = self.model.predict(x)
        idx = int(np.argmax(probs, axis=1)[0])
        return self.class_names[idx], float(np.max(probs))
