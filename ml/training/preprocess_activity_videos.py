"""
Preprocess activity videos into frame sequences for training MobileNetV2+LSTM.
- Extract frames at `frame_rate` fps
- Create sequences of length `seq_len`
- Save numpy arrays per split under ml/datasets/activity/<split>/npz

Usage:
python preprocess_activity_videos.py --src ml/datasets/activity --out ml/datasets/activity_processed --seq-len 30 --frame-rate 2
"""
import argparse
import os
import cv2
import numpy as np


def extract_frames(video_path, target_dir, fps=2, max_frames=None):
    os.makedirs(target_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(video_fps / fps)))
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            fname = os.path.join(target_dir, f'{saved:06d}.jpg')
            cv2.imwrite(fname, frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        count += 1
    cap.release()
    return saved


def video_dirs_to_sequences(root_dir, seq_len=30, out_dir=None):
    # root_dir: train/normal, train/suspicious, etc.
    if out_dir is None:
        out_dir = root_dir + '_processed'
    os.makedirs(out_dir, exist_ok=True)
    for cls in ['normal', 'suspicious']:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
            video_path = os.path.join(class_dir, fname)
            tmp_dir = os.path.join('/tmp', 'frames_'+os.path.splitext(fname)[0])
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            n = extract_frames(video_path, tmp_dir, fps=2)
            if n < seq_len:
                continue
            # build sequences
            imgs = sorted([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)])
            seqs = []
            for i in range(0, len(imgs)-seq_len+1, seq_len):
                seq = []
                for j in range(i, i+seq_len):
                    img = cv2.imread(imgs[j])
                    img = cv2.resize(img, (224,224))
                    seq.append(img)
                seqs.append(np.array(seq))
            if len(seqs)==0:
                continue
            out_npz = os.path.join(out_dir, f'{os.path.splitext(fname)[0]}.npz')
            np.savez_compressed(out_npz, sequences=np.array(seqs), label=cls)
    print('Finished processing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='ml/datasets/activity')
    parser.add_argument('--out', default='ml/datasets/activity_processed')
    parser.add_argument('--seq-len', type=int, default=30)
    args = parser.parse_args()
    video_dirs_to_sequences(args.src, seq_len=args.seq_len, out_dir=args.out)
