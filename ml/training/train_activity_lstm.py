import os
import json
import math
import argparse
import logging
import gc
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mpg", ".mpeg", ".mkv", ".webm")


def list_videos_and_labels(split_dir):
    split_dir = Path(split_dir)

    classes = []
    samples = []

    if not split_dir.exists():
        logger.warning("Split directory does not exist: %s", split_dir)
        return samples, classes

    for cls_path in sorted(split_dir.iterdir()):
        if not cls_path.is_dir():
            continue

        cls_name = cls_path.name
        classes.append(cls_name)

        for video_path in cls_path.rglob("*"):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append((str(video_path), cls_name))

    classes = sorted(list(set(classes)))
    return samples, classes


def read_video_frames(path, seq_len=16, target_size=(160, 160)):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        logger.warning("Unreadable video cannot open: %s", path)
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        logger.warning("Video has no readable frame count: %s", path)
        cap.release()
        return None

    indices = np.linspace(0, max(frame_count - 1, 0), num=seq_len)

    frames = []
    last_good = None

    for idx in indices:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(idx)))
            ret, frame = cap.read()
        except Exception as e:
            logger.warning("OpenCV read error in %s: %s", path, e)
            continue

        if not ret or frame is None:
            if last_good is None:
                logger.warning("Failed to read frame from %s at index %s", path, idx)
                continue
            frame = last_good.copy()
        else:
            last_good = frame.copy()

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        except Exception as e:
            logger.warning("Frame processing error in %s: %s", path, e)
            continue

    cap.release()

    if len(frames) == 0:
        logger.warning("No readable frames for video: %s", path)
        return None

    while len(frames) < seq_len:
        frames.append(frames[-1].copy())

    arr = np.stack(frames[:seq_len], axis=0)
    return arr


class VideoSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        samples,
        class2idx,
        batch_size=2,
        seq_len=16,
        target_size=(160, 160),
        shuffle=True,
        feature_extractor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.samples = samples
        self.class2idx = class2idx
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.target_size = target_size
        self.shuffle = shuffle
        self.feature_extractor = feature_extractor
        self.indexes = np.arange(len(self.samples))

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.samples) / float(self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indexes]

        feats_list = []
        labels = []

        for path, cls in batch_samples:
            frames = read_video_frames(
                path,
                seq_len=self.seq_len,
                target_size=self.target_size,
            )

            if frames is None:
                logger.warning("Skipping unreadable video: %s", path)
                continue

            try:
                frames = frames.astype("float32")
                frames_pre = preprocess_input(frames)

                if self.feature_extractor is not None:
                    feats = self.feature_extractor.predict(frames_pre, verbose=0)
                else:
                    feats = frames_pre

                feats_list.append(feats)
                labels.append(self.class2idx[cls])

            except Exception as e:
                logger.warning("Feature extraction failed for %s: %s", path, e)

            finally:
                try:
                    del frames
                    del frames_pre
                except Exception:
                    pass
                gc.collect()

        if len(feats_list) == 0:
            logger.warning("Batch %s has no readable videos. Returning zero dummy batch.", idx)

            if self.feature_extractor is not None:
                feat_dim = self.feature_extractor.output_shape[-1]
                dummy_x = np.zeros((1, self.seq_len, feat_dim), dtype="float32")
            else:
                dummy_x = np.zeros(
                    (1, self.seq_len, self.target_size[0], self.target_size[1], 3),
                    dtype="float32",
                )

            dummy_y = np.zeros((1,), dtype="int32")

            return dummy_x, tf.keras.utils.to_categorical(
                dummy_y,
                num_classes=len(self.class2idx),
            )

        X = np.stack(feats_list, axis=0).astype("float32")
        y = np.array(labels, dtype="int32")

        del feats_list
        del labels
        gc.collect()

        return X, tf.keras.utils.to_categorical(
            y,
            num_classes=len(self.class2idx),
        )


def build_feature_extractor(input_shape=(160, 160, 3)):
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    base.trainable = False
    return base


def build_model(seq_len=16, feat_dim=1280, num_classes=2):
    inp = layers.Input(shape=(seq_len, feat_dim), name="features")

    x = layers.Masking()(inp)
    x = layers.LSTM(256, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    return model


def plot_history(history, out_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix_image(cm, classes, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(args):
    base_dir = Path(args.data_dir)
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples, classes_train = list_videos_and_labels(train_dir)
    val_samples, classes_val = list_videos_and_labels(val_dir)
    test_samples, classes_test = list_videos_and_labels(test_dir)

    classes = sorted(list(set(classes_train + classes_val + classes_test)))

    if not classes:
        raise RuntimeError(
            f"No class folders found. Expected train/val/test folders inside {base_dir}"
        )

    if len(train_samples) == 0:
        raise RuntimeError(f"No training videos found inside {train_dir}")

    if len(val_samples) == 0:
        raise RuntimeError(f"No validation videos found inside {val_dir}")

    if len(test_samples) == 0:
        raise RuntimeError(f"No test videos found inside {test_dir}")

    class2idx = {c: i for i, c in enumerate(classes)}

    logger.info("Classes: %s", classes)
    logger.info("Train samples: %s", len(train_samples))
    logger.info("Validation samples: %s", len(val_samples))
    logger.info("Test samples: %s", len(test_samples))

    y_train = [class2idx[cls] for _, cls in train_samples]

    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )

    class_weights = {int(i): float(w) for i, w in zip(np.unique(y_train), cw)}
    logger.info("Class weights: %s", class_weights)

    feature_extractor = build_feature_extractor(
        input_shape=(args.img_size, args.img_size, 3)
    )

    feat_dim = feature_extractor.output_shape[-1]
    logger.info("Feature dim: %s", feat_dim)

    train_seq = VideoSequence(
        train_samples,
        class2idx,
        batch_size=args.batch_size,
        seq_len=args.frames,
        target_size=(args.img_size, args.img_size),
        shuffle=True,
        feature_extractor=feature_extractor,
    )

    val_seq = VideoSequence(
        val_samples,
        class2idx,
        batch_size=args.batch_size,
        seq_len=args.frames,
        target_size=(args.img_size, args.img_size),
        shuffle=False,
        feature_extractor=feature_extractor,
    )

    model = build_model(
        seq_len=args.frames,
        feat_dim=feat_dim,
        num_classes=len(classes),
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)

    ckpt_path = output_dir / "activity_model.keras"

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
    ]

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=cb,
    )

    plot_path = output_dir / "training_curves.png"
    plot_history(history, str(plot_path))

    model.save(str(ckpt_path))

    class_json = output_dir / "class_names.json"
    with open(class_json, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    test_seq = VideoSequence(
        test_samples,
        class2idx,
        batch_size=args.batch_size,
        seq_len=args.frames,
        target_size=(args.img_size, args.img_size),
        shuffle=False,
        feature_extractor=feature_extractor,
    )

    y_true = []
    y_pred = []

    for i in range(len(test_seq)):
        Xb, yb = test_seq[i]
        probs = model.predict(Xb, verbose=0)

        preds = probs.argmax(axis=1)

        y_true.extend(yb.argmax(axis=1).tolist())
        y_pred.extend(preds.tolist())

        del Xb
        del yb
        del probs
        gc.collect()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("Test accuracy:", acc)
    print("Test precision:", prec)
    print("Test recall:", rec)
    print("Test F1:", f1)
    print("Confusion matrix:\n", cm)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frames": args.frames,
        "img_size": args.img_size,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix_image(cm, classes, str(cm_path))

    print("\nTraining completed successfully.")
    print(f"Model saved to: {ckpt_path}")
    print(f"Class names saved to: {class_json}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Training curves saved to: {plot_path}")
    print(f"Confusion matrix image saved to: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="ml/datasets/activity",
        help="Base activity dataset directory containing train/val/test.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml/models/activity_mobilenet_lstm",
        help="Directory where model and artifacts will be saved.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    main(args)