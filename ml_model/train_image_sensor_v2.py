"""
train_image_sensor_v2.py

AgroFedVision -- Enhanced Image+Sensor Model
Adds:
  1. Image data augmentation (flip, rotation, zoom, contrast)
  2. Stratified 5-fold cross-validation (reports mean +/- std accuracy)
  3. Same two-stage training strategy as the original baseline

This does NOT touch fusion_model.py, train_multimodal.py, or the frozen
baseline in experiments/baseline_v1_confirmed/ -- it is a standalone
experiment script for the Image+Sensor configuration only.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sensor_transformer import build_sensor_transformer
from image_encoder import build_image_encoder

# ==========================================================
# CONFIG
# ==========================================================

DATASET = "data/multimodal_dataset.csv"
RESULTS_DIR = "results/image_sensor_v2_cv"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 10
LR_STAGE1 = 3e-4
LR_STAGE2 = 1e-5
N_FOLDS = 5
SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

UAV_COLS = ["NDVI_Mean", "NDVI_Std", "NDRE_Mean", "NDRE_Std"]

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ==========================================================
# AUGMENTATION LAYER
# ==========================================================
# Applied only during training (Keras automatically disables
# these at inference / model.evaluate time when trainable=False
# is not the issue -- these layers are no-ops outside training mode).

def build_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augmentation")


# ==========================================================
# LOAD DATA (once -- splitting happens per-fold)
# ==========================================================

def load_data():
    print("\nLoading dataset...")
    df = pd.read_csv(DATASET)
    print(df.shape)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["crop_health"])
    num_classes = len(encoder.classes_)
    print("Classes:", list(encoder.classes_))

    sensor_cols = [c for c in df.columns
                   if c not in ["image_path", "crop_health", "leaf_label"] + UAV_COLS]

    images, valid_idx = [], []
    for i, path in enumerate(df["image_path"]):
        try:
            img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            images.append(img_to_array(img))
            valid_idx.append(i)
        except Exception:
            pass

    X_img_raw = np.array(images, dtype=np.float32)
    df = df.iloc[valid_idx].reset_index(drop=True)
    y = y[valid_idx]

    print(f"Images loaded: {len(X_img_raw)}")
    print(f"Sensor features: {len(sensor_cols)}")

    return df, X_img_raw, y, num_classes, encoder, sensor_cols


# ==========================================================
# BUILD MODEL (fresh instance per fold)
# ==========================================================

def build_model(num_sensor_features, num_classes, dropout=0.30):
    augmentation = build_augmentation()

    sensor_encoder = build_sensor_transformer(num_features=num_sensor_features, d_model=64)
    sensor_input = sensor_encoder.input
    sensor_embedding = sensor_encoder.output

    image_encoder = build_image_encoder(
        embed_dim=256, dropout=dropout, trainable_backbone=False
    )
    raw_image_input = image_encoder.input
    # NOTE: image_encoder's internal preprocessing (EfficientNetPreprocess)
    # expects raw [0,255] pixel values -- augmentation must run BEFORE
    # that preprocessing layer, so we insert it at the very front here.
    augmented = augmentation(raw_image_input)
    image_embedding = image_encoder(augmented)

    fusion = tf.keras.layers.Concatenate(name="feature_fusion")(
        [sensor_embedding, image_embedding]
    )

    x = tf.keras.layers.Dense(256, activation="relu", name="head_dense1")(fusion)
    x = tf.keras.layers.BatchNormalization(name="head_bn1")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_drop1")(x)

    residual = x
    x = tf.keras.layers.Dense(256, activation="relu", name="head_dense2")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn2")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_drop2")(x)
    x = tf.keras.layers.Add(name="head_residual")([x, residual])

    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense3")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn3")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_drop3")(x)

    x = tf.keras.layers.Dense(64, activation="relu", name="head_dense4")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn4")(x)
    x = tf.keras.layers.Dropout(0.20, name="head_drop4")(x)

    output = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    name="crop_health_output")(x)

    model = tf.keras.Model(
        inputs=[sensor_input, raw_image_input],
        outputs=output,
        name="AgroFedVision_ImageSensor_V2"
    )
    return model, image_encoder


# ==========================================================
# TRAIN ONE FOLD
# ==========================================================

def train_one_fold(fold_idx, train_idx, val_idx, df, X_img_raw, y,
                    num_classes, sensor_cols):

    print(f"\n{'='*50}")
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'='*50}")

    # ---- Sensor scaling: fit ONLY on this fold's training rows ----
    # (fitting on the full dataset before splitting would leak
    #  validation-fold statistics into training -- avoided here)
    sensor_scaler = MinMaxScaler()
    X_sensor_train = sensor_scaler.fit_transform(df.loc[train_idx, sensor_cols]).astype(np.float32)
    X_sensor_val = sensor_scaler.transform(df.loc[val_idx, sensor_cols]).astype(np.float32)

    X_img_train = X_img_raw[train_idx]
    X_img_val = X_img_raw[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_ds = tf.data.Dataset.from_tensor_slices(
        ({"sensor_input": X_sensor_train, "image_input": X_img_train}, y_train)
    ).shuffle(len(y_train), seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        ({"sensor_input": X_sensor_val, "image_input": X_img_val}, y_val)
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model, image_encoder = build_model(len(sensor_cols), num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STAGE1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0
        ),
    ]

    print("\n-- Stage 1: frozen backbone --")
    history1 = model.fit(train_ds, validation_data=val_ds,
                          epochs=EPOCHS_STAGE1, callbacks=callbacks, verbose=1)
    stage1_best = max(history1.history["val_accuracy"])
    print(f"Stage 1 best val_accuracy: {stage1_best:.4f}")

    history2 = None
    if stage1_best >= 0.35:
        print("\n-- Stage 2: fine-tuning --")
        for layer in model.layers:
            if "efficientnetb0" in layer.name.lower():
                layer.trainable = True
                for sub in layer.layers[:-20]:
                    sub.trainable = False
                break
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STAGE2),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history2 = model.fit(train_ds, validation_data=val_ds,
                              epochs=EPOCHS_STAGE2, callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFold {fold_idx + 1} final validation accuracy: {acc:.4f}")

    # Save per-fold model (optional, comment out if disk space is tight)
    model.save(os.path.join(RESULTS_DIR, f"fold_{fold_idx + 1}_model.keras"))

    return acc, loss


# ==========================================================
# MAIN: 5-FOLD CROSS-VALIDATION
# ==========================================================

def main():
    df, X_img_raw, y, num_classes, encoder, sensor_cols = load_data()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_accuracies = []
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        acc, loss = train_one_fold(
            fold_idx, train_idx, val_idx, df, X_img_raw, y, num_classes, sensor_cols
        )
        fold_accuracies.append(acc)
        fold_losses.append(loss)

    fold_accuracies = np.array(fold_accuracies)
    fold_losses = np.array(fold_losses)

    print(f"\n{'='*50}")
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    for i, a in enumerate(fold_accuracies):
        print(f"Fold {i + 1}: {a:.4f}")
    print(f"\nMean Accuracy : {fold_accuracies.mean():.4f}")
    print(f"Std Accuracy  : {fold_accuracies.std():.4f}")
    print(f"Min / Max     : {fold_accuracies.min():.4f} / {fold_accuracies.max():.4f}")
    print(f"\nPrevious single-split baseline (no augmentation): 0.7778")

    # ---- Save summary ----
    summary = {
        "fold_accuracies": fold_accuracies.tolist(),
        "fold_losses": fold_losses.tolist(),
        "mean_accuracy": float(fold_accuracies.mean()),
        "std_accuracy": float(fold_accuracies.std()),
        "min_accuracy": float(fold_accuracies.min()),
        "max_accuracy": float(fold_accuracies.max()),
        "previous_baseline_single_split": 0.7778,
        "n_folds": N_FOLDS,
        "augmentation": ["RandomFlip", "RandomRotation(0.08)", "RandomZoom(0.10)", "RandomContrast(0.10)"],
    }
    with open(os.path.join(RESULTS_DIR, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    joblib.dump(encoder, os.path.join(RESULTS_DIR, "label_encoder.pkl"))
    joblib.dump(sensor_cols, os.path.join(RESULTS_DIR, "sensor_cols.pkl"))

    # ---- Plot fold accuracies ----
    plt.figure(figsize=(7, 4))
    plt.bar(range(1, N_FOLDS + 1), fold_accuracies, color="#2C5F2D")
    plt.axhline(fold_accuracies.mean(), color="#C98A2E", linestyle="--",
                label=f"Mean = {fold_accuracies.mean():.4f}")
    plt.axhline(0.7778, color="gray", linestyle=":",
                label="Previous single-split baseline = 0.7778")
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy")
    plt.title("5-Fold Cross-Validation: Image+Sensor (with augmentation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cv_fold_accuracies.png"), dpi=200)
    plt.close()

    print(f"\nSaved results to {RESULTS_DIR}/")
    print("  cv_summary.json")
    print("  cv_fold_accuracies.png")
    print("  fold_1_model.keras ... fold_5_model.keras")
    print("\nDone.")


if __name__ == "__main__":
    main()