"""
Simple anti-spoofing heuristics:
- frame_difference_score: average pixel diff across consecutive frames
- repeated_frame_detector: counts how many frames are nearly identical
- low_motion_detector: detects videos with low motion overall

These are lightweight heuristics—replace or augment with ML-based methods for production.
"""
import cv2
import numpy as np


def frame_difference_score(frames):
    # frames: list/array of grayscale frames
    diffs = []
    for i in range(1, len(frames)):
        diffs.append(np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float))))
    if len(diffs)==0:
        return 0.0
    return float(np.mean(diffs))


def repeated_frame_ratio(frames, thresh=1.0):
    repeated = 0
    total = 0
    for i in range(1, len(frames)):
        diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
        if diff < thresh:
            repeated += 1
        total += 1
    return repeated / total if total>0 else 0.0


def low_motion(frames, motion_threshold=2.0):
    score = frame_difference_score(frames)
    return score < motion_threshold


if __name__ == '__main__':
    print('anti_spoof module: provide frames arrays to use functions')
