"""
Realtime pipeline example that ties together OpenCV capture, ID detection, activity prediction, and anti-spoof heuristics.
This is a local runner for testing and debugging; integration with Django will be done separately.
"""
import cv2
import time
import numpy as np
from ml.inference.detector import IDDetector
from ml.inference.activity_predictor import ActivityPredictor
from ml.inference.anti_spoof import frame_difference_score, repeated_frame_ratio


def run_camera(device=0, seq_len=30):
    id_det = IDDetector()
    act = ActivityPredictor()
    cap = cv2.VideoCapture(device)
    buffer_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (640,360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        buffer_frames.append(gray)
        if len(buffer_frames) > seq_len:
            buffer_frames.pop(0)
        # run inference when buffer full
        if len(buffer_frames) == seq_len:
            # anti-spoof heuristics
            diff_score = frame_difference_score(buffer_frames)
            repeat_ratio = repeated_frame_ratio(buffer_frames)
            spoof_flag = (diff_score < 2.0) or (repeat_ratio > 0.6)
            activity_result = None
            if not act.model_missing:
                # prepare RGB frames resized for model
                seq_rgb = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), (224,224)) for f in buffer_frames]
                activity_result = act.predict_sequence(np.array(seq_rgb))
            id_boxes = []
            if not id_det.model_missing:
                detections = id_det.detect_image(small)
                # map detections onto display
                id_boxes = detections
            # display overlay
            display = small.copy()
            if id_boxes:
                for d in id_boxes:
                    x1,y1,x2,y2 = map(int,d['bbox'][:4])
                    cv2.rectangle(display, (x1,y1),(x2,y2),(0,255,0),2)
            status = 'OK'
            if spoof_flag:
                status = 'SPOOF'
            if activity_result:
                label, conf = activity_result
                cv2.putText(display, f'{label} {conf:.2f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2)
            cv2.putText(display, f'Status: {status}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if spoof_flag else (0,255,0),2)
            cv2.imshow('Realtime Pipeline', display)
        else:
            cv2.imshow('Realtime Pipeline', small)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_camera()
