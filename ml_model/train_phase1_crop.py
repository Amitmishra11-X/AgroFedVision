"""
train_phase1_crop.py  (FIXED -- lazy image loading)

AgroFedVision Phase 1 -- Crop-Specific Image Classifier

FIX vs previous version: the original script loaded every image into
RAM as a decoded float32 array before training (fine at 761 images,
~1.8GB -- but at 11,250 guava images that becomes ~27GB and crashes
the allocator). This version builds a lazy tf.data pipeline that
reads and decodes images from disk in small batches during training,
so memory use stays bounded regardless of dataset size.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from image_encoder import build_image_encoder

# ==========================================================
# CONFIG -- change this one line to switch crops
# ==========================================================

    # "guava" or "maize"

DATASET_MAP = {
    "guava": "data/guava_dataset_dedup.csv",   # was guava_dataset.csv
    "maize": "data/maize_dataset.csv",
    "paddy": "data/paddy_dataset.csv",
}

RESULTS_DIR = f"results/phase1_{CROP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 10
LR_STAGE1 = 3e-4 
LR_STAGE2 = 1e-5
N_FOLDS = 5
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ==========================================================
# MILD AUGMENTATION -- flip only
# ==========================================================

def build_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
    ], name="augmentation")


# ==========================================================
# LAZY IMAGE LOADING (the actual fix)
# ==========================================================
# Reads and decodes one image at a time, inside the tf.data
# pipeline, only when that batch is actually needed -- rather
# than decoding all images into RAM before training starts.

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)   # kept in [0,255] -- image_encoder's
                                      # own preprocessing layer handles
                                      # the correct EfficientNet scaling
    return img, label


def build_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ==========================================================
# LOAD METADATA ONLY (cheap -- no pixel data touched here)
# ==========================================================

def load_data():
    csv_path = DATASET_MAP[CROP]
    print(f"\nLoading {CROP} dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(df.shape)

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(df["crop_health"])
    num_classes = len(encoder.classes_)
    print("Classes:", list(encoder.classes_))
    print("\nClass distribution:")
    print(df["crop_health"].value_counts())

    # Cheap existence check (no image decoding) instead of the
    # previous full-decode validation loop
    print("\nVerifying image paths exist (fast check, no decoding)...")
    exists_mask = df["image_path"].apply(os.path.exists).values
    n_missing = (~exists_mask).sum()
    if n_missing > 0:
        print(f"  [warning] {n_missing} image paths not found, excluding them")

    paths = df.loc[exists_mask, "image_path"].values
    y = y_all[exists_mask]

    print(f"Usable images: {len(paths)} / {len(df)}")

    return paths, y, num_classes, encoder


# ==========================================================
# BUILD MODEL
# ==========================================================

def build_model(num_classes, dropout=0.30):
    augmentation = build_augmentation()

    image_encoder = build_image_encoder(
        embed_dim=256, dropout=dropout, trainable_backbone=False
    )
    raw_input = image_encoder.input
    augmented = augmentation(raw_input)
    embedding = image_encoder(augmented)

    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense1")(embedding)
    x = tf.keras.layers.BatchNormalization(name="head_bn1")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_drop1")(x)

    x = tf.keras.layers.Dense(64, activation="relu", name="head_dense2")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn2")(x)
    x = tf.keras.layers.Dropout(0.20, name="head_drop2")(x)

    output = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    name=f"{CROP}_health_output")(x)

    model = tf.keras.Model(inputs=raw_input, outputs=output,
                            name=f"AgroFedVision_{CROP.capitalize()}")
    return model


# ==========================================================
# TRAIN ONE FOLD
# ==========================================================

def train_one_fold(fold_idx, train_idx, val_idx, paths, y, num_classes):

    print(f"\n{'='*60}")
    print(f"{CROP.upper()} - FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'='*60}")

    # ----------------------------------------------------
    # Build datasets
    # ----------------------------------------------------
    train_ds = build_dataset(paths[train_idx], y[train_idx], shuffle=True)
    val_ds   = build_dataset(paths[val_idx], y[val_idx], shuffle=False)

    # ----------------------------------------------------
    # Build model
    # ----------------------------------------------------
    model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_STAGE1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ----------------------------------------------------
    # Callbacks
    # ----------------------------------------------------
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            RESULTS_DIR,
            f"fold_{fold_idx + 1}_best.keras"
        ),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks = [
        checkpoint,
        early_stop,
        reduce_lr,
    ]

    # ----------------------------------------------------
    # Stage 1 : Frozen Backbone
    # ----------------------------------------------------
    print("\nStage 1 : Training Classification Head")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks,
        verbose=1,
    )

    stage1_best = max(history1.history["val_accuracy"])

    print(f"\nStage 1 Best Validation Accuracy : {stage1_best:.4f}")

    # ----------------------------------------------------
    # Stage 2 : Fine Tuning
    # ----------------------------------------------------
    if stage1_best >= 0.35:

        print("\nStage 2 : Fine-Tuning EfficientNet")

        for layer in model.layers:
            if "efficientnetb0" in layer.name.lower():

                layer.trainable = True

                for sub in layer.layers[:-20]:
                    sub.trainable = False

                break

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LR_STAGE2
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_STAGE2,
            callbacks=callbacks,
            verbose=1,
        )

    # ----------------------------------------------------
    # Final Evaluation
    # ----------------------------------------------------
    loss, acc = model.evaluate(val_ds, verbose=0)

    print(f"\nFold {fold_idx + 1} Validation Accuracy : {acc:.4f}")

    # ----------------------------------------------------
    # Save Final Model
    # ----------------------------------------------------
    model.save(
        os.path.join(
            RESULTS_DIR,
            f"fold_{fold_idx + 1}_final.keras"
        )
    )

    return acc, loss


# ==========================================================
# MAIN
# ==========================================================

def main():
    paths, y, num_classes, encoder = load_data()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_accuracies, fold_losses = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        acc, loss = train_one_fold(fold_idx, train_idx, val_idx, paths, y, num_classes)
        fold_accuracies.append(acc)
        fold_losses.append(loss)

    fold_accuracies = np.array(fold_accuracies)
    fold_losses = np.array(fold_losses)

    print(f"\n{'='*50}")
    print(f"{CROP.upper()} -- 5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    for i, a in enumerate(fold_accuracies):
        print(f"Fold {i + 1}: {a:.4f}")
    print(f"\nMean Accuracy : {fold_accuracies.mean():.4f}")
    print(f"Std Accuracy  : {fold_accuracies.std():.4f}")
    print(f"Min / Max     : {fold_accuracies.min():.4f} / {fold_accuracies.max():.4f}")

    summary = {
        "crop": CROP,
        "num_classes": num_classes,
        "classes": list(encoder.classes_),
        "fold_accuracies": fold_accuracies.tolist(),
        "fold_losses": fold_losses.tolist(),
        "mean_accuracy": float(fold_accuracies.mean()),
        "std_accuracy": float(fold_accuracies.std()),
        "min_accuracy": float(fold_accuracies.min()),
        "max_accuracy": float(fold_accuracies.max()),
        "n_folds": N_FOLDS,
        "augmentation": ["RandomFlip(horizontal) only"],
    }
    with open(os.path.join(RESULTS_DIR, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    joblib.dump(encoder, os.path.join(RESULTS_DIR, "label_encoder.pkl"))

    plt.figure(figsize=(7, 4))
    plt.bar(range(1, N_FOLDS + 1), fold_accuracies, color="#2C5F2D")
    plt.axhline(fold_accuracies.mean(), color="#C98A2E", linestyle="--",
                label=f"Mean = {fold_accuracies.mean():.4f}")
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy")
    plt.title(f"5-Fold CV: {CROP.capitalize()} Classifier")
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