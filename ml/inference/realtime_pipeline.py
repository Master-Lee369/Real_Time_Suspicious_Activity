"""Safe realtime pipeline helpers.

This module provides small helper functions that are safe to import at Django
startup (they avoid heavy ML imports) and can be called from views or CLI.

Functions:
 - read_video_frames(path, max_frames, resize) -> list of frames (BGR numpy arrays)
 - process_video_path(path) -> dict (uses IDDetector and ActivityPredictor wrappers)
 - run_camera(device, seq_len) -> interactive demo (safe checks)

Note: Actual inference is delegated to the wrapper classes which return structured
responses; this module intentionally leaves the heavy work to them.
"""
import os
from typing import List


def read_video_frames(path: str, max_frames: int = 300, resize=None) -> List:
    """Read up to `max_frames` frames from a video using OpenCV.

    Returns a list of BGR numpy arrays. Does not raise if OpenCV is unavailable; instead
    returns an empty list and an error message in the caller's context.
    """
    try:
        import cv2
    except Exception:
        return []

    if not os.path.exists(path):
        return []

    frames = []
    cap = cv2.VideoCapture(path)
    idx = 0
    while idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            try:
                frame = cv2.resize(frame, resize)
            except Exception:
                pass
        frames.append(frame)
        idx += 1
    cap.release()
    return frames


def process_video_path(path: str, sample_frames: int = 30) -> dict:
    """High-level processing of a video file.

    Returns structured dicts. If models are missing, returns:
      {"status":"model_missing","message":"Model not trained yet"}

    Otherwise returns summary information and per-frame/sequence results.
    """
    from ml.inference.id_detector import IDDetector
    from ml.inference.activity_predictor import ActivityPredictor

    id_det = IDDetector()
    act = ActivityPredictor()

    if not id_det.is_model_available() or not act.is_model_available():
        return {"status": "model_missing", "message": "Model not trained yet"}

    frames = read_video_frames(path, max_frames=sample_frames, resize=(640, 360))
    if not frames:
        return {"status": "error", "message": "No frames read from video"}

    results = []
    for i, f in enumerate(frames):
        # ID detection per frame
        id_out = id_det.detect_image(f)
        # Activity prediction requires sequences; here we provide placeholder per-frame
        results.append({"frame": i, "id": id_out})

    return {"status": "ok", "frames": len(frames), "results": results}


def run_camera(device=0, seq_len=30):
    """Interactive camera demo. This function will attempt to access the camera and
    run a simple display loop. It performs safe checks: if models are missing it will
    show a message on the window but will not crash.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        print("OpenCV is not available. Install opencv-python to run the demo.")
        return

    # Lazy import wrappers
    from ml.inference.id_detector import IDDetector
    from ml.inference.activity_predictor import ActivityPredictor
    try:
        from ml.inference.anti_spoof import frame_difference_score, repeated_frame_ratio
    except Exception:
        # Provide simple fallbacks
        def frame_difference_score(buf):
            return 10.0

        def repeated_frame_ratio(buf):
            return 0.0

    id_det = IDDetector()
    act = ActivityPredictor()

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print('Camera not available')
        return

    buffer_frames = []
    window = 'Realtime Pipeline'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        buffer_frames.append(gray)
        if len(buffer_frames) > seq_len:
            buffer_frames.pop(0)

        display = small.copy()

        if len(buffer_frames) == seq_len:
            diff_score = frame_difference_score(buffer_frames)
            repeat_ratio = repeated_frame_ratio(buffer_frames)
            spoof_flag = (diff_score < 2.0) or (repeat_ratio > 0.6)

            # Activity
            if act.is_model_available():
                seq_rgb = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), (224, 224)) for f in buffer_frames]
                activity_result = act.predict_sequence(np.array(seq_rgb))
                if isinstance(activity_result, dict) and activity_result.get('status') == 'ok':
                    lbl = activity_result.get('label')
                    conf = activity_result.get('confidence')
                    cv2.putText(display, f'{lbl} {conf:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                elif isinstance(activity_result, dict) and activity_result.get('status') == 'model_missing':
                    cv2.putText(display, 'Activity model missing', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display, 'Activity model missing', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ID detection on current frame
            if id_det.is_model_available():
                id_out = id_det.detect_image(small)
                if isinstance(id_out, dict) and id_out.get('status') == 'ok':
                    for d in id_out.get('detections', []):
                        x1, y1, x2, y2 = map(int, d['bbox'][:4])
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif isinstance(id_out, dict) and id_out.get('status') == 'model_missing':
                    cv2.putText(display, 'ID model missing', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display, 'ID model missing', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            status = 'OK'
            if spoof_flag:
                status = 'SPOOF'
            cv2.putText(display, f'Status: {status}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if status == 'OK' else (0, 0, 255), 2)

        cv2.imshow(window, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # simple CLI demo
    run_camera()
