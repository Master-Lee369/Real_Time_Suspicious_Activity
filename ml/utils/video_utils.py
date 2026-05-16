"""
Video utility helpers: extract frames, load sequence for prediction, save sequences.
"""
import cv2
import numpy as np


def extract_sequence_from_video(path, seq_len=30, fps=2, resize=(224,224)):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(video_fps / fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.resize(frame, resize)
            frames.append(frame)
            if len(frames) >= seq_len:
                break
        idx += 1
    cap.release()
    if len(frames) < seq_len:
        return None
    return np.array(frames[:seq_len])
