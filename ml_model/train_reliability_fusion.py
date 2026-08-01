"""
train_reliability_fusion.py
AgroFedVision — Reliability-Aware Fusion Training
Compare result against baseline: 74.51%
"""

import os, sys, json, joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.stats import zscore

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reliability_fusion_model import build_reliability_fusion_model

tf.random.set_seed(42)
np.random.seed(42)

# ── Config ─────────────────────────────────────────────────
DATASET       = "data/multimodal_dataset.csv"
IMG_QUALITY   = "results/image_quality.csv"
RESULTS_DIR   = "results/reliability_fusion"
UAV_COLS      = ["NDVI_Mean", "NDVI_Std", "NDRE_Mean", "NDRE_Std"]
IMG_SIZE      = 224
BATCH_SIZE    = 16
EPOCHS        = 25
LR            = 3e-4
SEED          = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_sensor_reliability(df, sensor_cols):
    """
    Per-sample sensor reliability: 1 - mean(|z-score|) / 3, clipped to [0,1].
    Rows close to the dataset mean get high reliability (near 1.0).
    Outlier rows get lower reliability.
    """
    z = np.abs(zscore(df[sensor_cols].values, axis=0))
    mean_z = z.mean(axis=1)                    # (N,)
    reliability = np.clip(1.0 - mean_z / 3.0, 0.1, 1.0)
    return reliability.astype(np.float32)


def load_data():
    print("\nLoading dataset...")
    df  = pd.read_csv(DATASET)
    iq  = pd.read_csv(IMG_QUALITY)

    # ── Labels ─────────────────────────────────────────────
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["crop_health"])

    # ── Sensor features ────────────────────────────────────
    sensor_cols = [c for c in df.columns
                   if c not in ["image_path","crop_health","leaf_label"] + UAV_COLS]
    sensor_scaler = MinMaxScaler()
    X_sensor = sensor_scaler.fit_transform(df[sensor_cols]).astype(np.float32)

    # ── Per-sample sensor reliability ──────────────────────
    sensor_reliability = compute_sensor_reliability(df, sensor_cols)  # (N,)
    print(f"Sensor reliability — min:{sensor_reliability.min():.3f}  "
          f"mean:{sensor_reliability.mean():.3f}  max:{sensor_reliability.max():.3f}")

    # ── UAV features ───────────────────────────────────────
    uav_scaler = MinMaxScaler()
    X_uav = uav_scaler.fit_transform(df[UAV_COLS]).astype(np.float32)

    # ── Image quality scores ────────────────────────────────
    # Join on filename
    df["basename"] = df["image_path"].apply(os.path.basename)
    iq_map = dict(zip(iq["Image"], iq["Quality"]))
    img_quality_scores = df["basename"].map(iq_map).fillna(0.5).values.astype(np.float32)

    # ── Images ─────────────────────────────────────────────
    images, valid_idx = [], []
    for i, path in enumerate(df["image_path"]):
        try:
            img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            images.append(img_to_array(img))
            valid_idx.append(i)
        except:
            pass

    X_img    = np.array(images, dtype=np.float32)
    X_sensor = X_sensor[valid_idx]
    X_uav    = X_uav[valid_idx]
    y        = y[valid_idx]
    img_q    = img_quality_scores[valid_idx].reshape(-1, 1)
    sen_q    = sensor_reliability[valid_idx].reshape(-1, 1)

    print(f"Images loaded: {len(X_img)}")
    print(f"Sensor features: {X_sensor.shape[1]}")
    return (X_sensor, X_uav, X_img, img_q, sen_q,
            y, len(encoder.classes_), encoder,
            sensor_scaler, uav_scaler, sensor_cols)


def build_dataset(X_sensor, X_uav, X_img, img_q, sen_q, y,
                  batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((
        {"sensor_input":        X_sensor,
         "uav_input":           X_uav,
         "image_input":         X_img,
         "image_quality_input": img_q,
         "sensor_quality_input":sen_q},
        y
    ))
    if shuffle:
        ds = ds.shuffle(len(y), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train():
    (X_sensor, X_uav, X_img, img_q, sen_q,
     y, num_classes, encoder,
     sensor_scaler, uav_scaler, sensor_cols) = load_data()

    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.20,
                               random_state=SEED, stratify=y)

    train_ds = build_dataset(X_sensor[tr], X_uav[tr], X_img[tr],
                              img_q[tr], sen_q[tr], y[tr], BATCH_SIZE)
    val_ds   = build_dataset(X_sensor[va], X_uav[va], X_img[va],
                              img_q[va], sen_q[va], y[va], BATCH_SIZE,
                              shuffle=False)

    model = build_reliability_fusion_model(
        num_sensor_features=X_sensor.shape[1],
        num_classes=num_classes
    )
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True)
    ]

    # Stage 1 — frozen backbone
    print("\n=== Stage 1: Frozen EfficientNet ===")
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS, callbacks=callbacks, verbose=1)
    stage1_best = max(history.history["val_accuracy"])
    print(f"\nStage 1 best: {stage1_best:.4f}")

    # Stage 2 — fine-tune
    history2 = None
    if stage1_best >= 0.35:
        print("\n=== Stage 2: Fine-tuning EfficientNet ===")
        for layer in model.layers:
            if "efficientnetb0" in layer.name.lower():
                layer.trainable = True
                for sub in layer.layers[:-20]:
                    sub.trainable = False
                break
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        history2 = model.fit(train_ds, validation_data=val_ds,
                             epochs=10, callbacks=callbacks, verbose=1)

    # Evaluate
    loss, acc = model.evaluate(val_ds, verbose=1)
    print(f"\n{'='*40}")
    print(f"RELIABILITY-AWARE FUSION RESULT")
    print(f"{'='*40}")
    print(f"Validation Accuracy : {acc:.4f}")
    print(f"Validation Loss     : {loss:.4f}")
    print(f"Baseline            : 0.7451")
    print(f"Change              : {acc - 0.7451:+.4f}")

    # Save
    model.save(os.path.join(RESULTS_DIR, "reliability_fusion_model.keras"))
    joblib.dump(encoder,       os.path.join(RESULTS_DIR, "label_encoder.pkl"))
    joblib.dump(sensor_scaler, os.path.join(RESULTS_DIR, "sensor_scaler.pkl"))
    joblib.dump(uav_scaler,    os.path.join(RESULTS_DIR, "uav_scaler.pkl"))
    joblib.dump(sensor_cols,   os.path.join(RESULTS_DIR, "sensor_cols.pkl"))

    history_df = pd.DataFrame(history.history)
    if history2:
        history_df = pd.concat([history_df, pd.DataFrame(history2.history)],
                                ignore_index=True)
    history_df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"),
                      index=False)

    json.dump({"accuracy": float(acc), "loss": float(loss),
               "baseline": 0.7451, "delta": float(acc - 0.7451)},
              open(os.path.join(RESULTS_DIR, "metrics.json"), "w"), indent=4)

    print("\nSaved to results/reliability_fusion/")


if __name__ == "__main__":
    train()