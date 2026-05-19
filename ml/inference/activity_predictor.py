"""
Activity predictor wrapper. Loads Keras model if available and predicts on sequences of frames.
If model missing, `model_missing=True` and `predict` returns None.
"""
import os
import numpy as np


class ActivityPredictor:
    """Safe activity predictor wrapper.

    Usage:
      ap = ActivityPredictor()
      if not ap.is_model_available():
          return {"status":"model_missing","message":"Model not trained yet"}
      out = ap.predict_sequence(seq)

    The heavy TF import and model loading are done lazily and only once.
    """

    def __init__(self, model_path='ml/models/activity_mobilenet_lstm/activity_model.keras', class_path='ml/models/activity_mobilenet_lstm/class_names.json'):
        self.model_path = model_path
        self.class_path = class_path
        self._model = None
        self._class_names = None

    def is_model_available(self):
        return os.path.exists(self.model_path) and os.path.exists(self.class_path)

    def _ensure_model_loaded(self):
        if self._model is not None and self._class_names is not None:
            return True
        if not os.path.exists(self.model_path) or not os.path.exists(self.class_path):
            return False
        try:
            # Lazy import heavy libs
            from tensorflow.keras.models import load_model
            import json
            self._model = load_model(self.model_path)
            with open(self.class_path, 'r') as fh:
                self._class_names = json.load(fh)
            return True
        except Exception:
            self._model = None
            self._class_names = None
            return False

    def predict_sequence(self, seq):
        """Predict on a sequence of frames.

        Returns:
          dict with keys:
            - status: 'ok' | 'model_missing' | 'error'
            - if ok: 'label' and 'confidence'
            - if error: 'message'
        """
        if not self._ensure_model_loaded():
            return {"status": "model_missing", "message": "Model not trained yet"}

        try:
            x = np.array(seq, dtype=np.float32) / 255.0
            x = np.expand_dims(x, axis=0)
            probs = self._model.predict(x)
            idx = int(np.argmax(probs, axis=1)[0])
            return {"status": "ok", "label": self._class_names[idx], "confidence": float(np.max(probs))}
        except Exception as e:
            return {"status": "error", "message": str(e)}
