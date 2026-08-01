"""
==============================================================
AgroFedVision - Phase 1 Training
Crop Disease Classification

Author : Amit Mishra
Model  : EfficientNet-B0
Version: 2.0
==============================================================
"""

# ==========================================================
# Imports
# ==========================================================

import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ==========================================================
# Import Image Encoder
# ==========================================================

from image_encoder import build_image_encoder

# ==========================================================
# Configuration
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 5

LR_STAGE1 = 3e-4
LR_STAGE2 = 1e-5

CROP = "Guava"

CSV_FILE = "data/guava_dataset_dedup.csv"

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("AgroFedVision Phase 1")
print("=" * 60)

print("TensorFlow :", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU Detected :", gpus[0].name)
else:
    print("Running on CPU")

# ==========================================================
# Dataset Loader
# ==========================================================

def load_dataset():

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Dataset not found:\n{CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    print("\nDataset Loaded")
    print(df.head())

    print("\nShape :", df.shape)
    print("\nColumns")
    print(df.columns.tolist())

    # Detect label column
    if "crop_health" in df.columns:
        label_column = "crop_health"

    elif "label" in df.columns:
        label_column = "label"

    elif "original_label" in df.columns:
        label_column = "original_label"

    else:
        raise ValueError(
            "No valid label column found."
        )

    # Remove missing images
    exists = df["image_path"].apply(os.path.exists)

    df = df[exists].reset_index(drop=True)

    print("\nValid Images :", len(df))

    print("\nClass Distribution")
    print(df[label_column].value_counts())

    return df, label_column

# ==========================================================
# Encode Labels
# ==========================================================

def encode_labels(df, label_column):

    encoder = LabelEncoder()

    labels = encoder.fit_transform(df[label_column])

    print("\nClasses")

    for i, cls in enumerate(encoder.classes_):

        print(f"{i} -> {cls}")

    return labels, encoder

# ==========================================================
# Train / Validation / Test Split
# ==========================================================

def split_dataset(df, labels):

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(

        df["image_path"],

        labels,

        test_size=0.30,

        random_state=SEED,

        stratify=labels

    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(

        temp_paths,

        temp_labels,

        test_size=0.50,

        random_state=SEED,

        stratify=temp_labels

    )

    print("\nDataset Split")

    print(f"Train : {len(train_paths)}")

    print(f"Validation : {len(val_paths)}")

    print(f"Test : {len(test_paths)}")

    return (

        train_paths.tolist(),

        val_paths.tolist(),

        test_paths.tolist(),

        train_labels,

        val_labels,

        test_labels

    )

# ==========================================================
# Image Loader
# ==========================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(

        image,

        channels=3

    )

    image = tf.image.resize(

        image,

        IMG_SIZE

    )

    image = tf.cast(

        image,

        tf.float32

    )

    return image, label

# ==========================================================
# Data Augmentation
# ==========================================================

augmentation = tf.keras.Sequential(

    [

        tf.keras.layers.RandomFlip("horizontal"),

        tf.keras.layers.RandomRotation(0.08),

        tf.keras.layers.RandomZoom(0.10),

        tf.keras.layers.RandomContrast(0.10)

    ],

    name="augmentation"

)

# ==========================================================
# tf.data Pipeline
# ==========================================================

def create_dataset(paths, labels, training=False):

    ds = tf.data.Dataset.from_tensor_slices(

        (paths, labels)

    )

    ds = ds.map(

        load_image,

        num_parallel_calls=AUTOTUNE

    )

    if training:

        ds = ds.shuffle(

            1024,

            seed=SEED

        )

        ds = ds.map(

            lambda x, y: (augmentation(x, training=True), y),

            num_parallel_calls=AUTOTUNE

        )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(AUTOTUNE)

    return ds

# ==========================================================
# Compute Class Weights
# ==========================================================

def get_class_weights(labels):

    weights = compute_class_weight(

        class_weight="balanced",

        classes=np.unique(labels),

        y=labels

    )

    class_weights = {

        i: float(w)

        for i, w in enumerate(weights)

    }

    print("\nClass Weights")

    print(class_weights)

    return class_weights
# ==========================================================
# Build Classification Model
# ==========================================================

def build_model(num_classes):

    print("\nBuilding Model...")

    # Your Image Encoder
    image_encoder = build_image_encoder(
        embed_dim=256,
        dropout=0.30,
        trainable_backbone=False
    )

    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="input_image"
    )

    x = image_encoder(inputs)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.40)(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="prediction"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name=f"{CROP}_Classifier"
    )

    return model


# ==========================================================
# Callbacks
# ==========================================================

def build_callbacks():

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=os.path.join(
                RESULTS_DIR,
                "best_model.keras"
            ),

            monitor="val_accuracy",

            save_best_only=True,

            save_weights_only=False,

            verbose=1

        ),

        tf.keras.callbacks.EarlyStopping(

            monitor="val_accuracy",

            patience=4,

            restore_best_weights=True,

            verbose=1

        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=2,

            min_lr=1e-6,

            verbose=1

        ),

        tf.keras.callbacks.CSVLogger(

            os.path.join(
                RESULTS_DIR,
                "training_log.csv"
            )

        ),

        tf.keras.callbacks.TensorBoard(

            log_dir=os.path.join(
                RESULTS_DIR,
                "tensorboard"
            )

        )

    ]

    return callbacks


# ==========================================================
# Stage 1 Training
# ==========================================================

def train_stage1(

    model,

    train_ds,

    val_ds,

    class_weights

):

    print("\n" + "=" * 60)
    print("Stage 1 Training")
    print("=" * 60)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LR_STAGE1
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS_STAGE1,

        callbacks=build_callbacks(),

        class_weight=class_weights,

        verbose=1

    )

    return history


# ==========================================================
# Fine Tune EfficientNet
# ==========================================================
def unfreeze_backbone(model):

    image_encoder = model.get_layer("ImageEncoder")

    image_encoder.trainable = True

    unfreeze = False

    for layer in image_encoder.layers:

        if layer.name.startswith("block6"):
            unfreeze = True

        if layer.name.startswith("block7"):
            unfreeze = True

        layer.trainable = unfreeze

    print("Fine-tuning Block6 + Block7")


# ==========================================================
# Stage 2 Training
# ==========================================================

def train_stage2(

    model,

    train_ds,

    val_ds,

    class_weights

):

    print("\n" + "=" * 60)
    print("Stage 2 Fine Tuning")
    print("=" * 60)

    unfreeze_backbone(model)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LR_STAGE2
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS_STAGE2,

        callbacks=build_callbacks(),

        class_weight=class_weights,

        verbose=1

    )

    return history


# ==========================================================
# Complete Training Pipeline
# ==========================================================

def train_model(

    model,

    train_ds,

    val_ds,

    class_weights

):

    history1 = train_stage1(

        model,

        train_ds,

        val_ds,

        class_weights

    )

    history2 = train_stage2(

        model,

        train_ds,

        val_ds,

        class_weights

    )

    return history1, history2
# ==========================================================
# Imports Required For Evaluation
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model, test_ds, encoder):

    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(images, verbose=0)

        predictions = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predictions)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_,
        digits=4
    )

    print("\nClassification Report\n")
    print(report)

    with open(
        os.path.join(
            RESULTS_DIR,
            "classification_report.txt"
        ),
        "w"
    ) as f:

        f.write(report)

    metrics = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1)

    }

    with open(

        os.path.join(
            RESULTS_DIR,
            "metrics.json"
        ),

        "w"

    ) as f:

        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return cm


# ==========================================================
# Save Confusion Matrix
# ==========================================================

def save_confusion_matrix(cm, encoder):

    plt.figure(figsize=(8,7))

    plt.imshow(cm, cmap="Blues")

    plt.title("Confusion Matrix")

    plt.colorbar()

    ticks = np.arange(len(encoder.classes_))

    plt.xticks(
        ticks,
        encoder.classes_,
        rotation=45
    )

    plt.yticks(
        ticks,
        encoder.classes_
    )

    plt.xlabel("Predicted")

    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            RESULTS_DIR,
            "confusion_matrix.png"
        ),

        dpi=300

    )

    plt.close()


# ==========================================================
# Plot Training Curves
# ==========================================================

def plot_training(history1, history2):

    acc = history1.history["accuracy"]
    val_acc = history1.history["val_accuracy"]

    loss = history1.history["loss"]
    val_loss = history1.history["val_loss"]

    if history2 is not None:

        acc += history2.history["accuracy"]
        val_acc += history2.history["val_accuracy"]

        loss += history2.history["loss"]
        val_loss += history2.history["val_loss"]

    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        acc,
        label="Training"
    )

    plt.plot(
        epochs,
        val_acc,
        label="Validation"
    )

    plt.legend()

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            RESULTS_DIR,
            "accuracy_curve.png"
        ),

        dpi=300

    )

    plt.close()

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        loss,
        label="Training"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation"
    )

    plt.legend()

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            RESULTS_DIR,
            "loss_curve.png"
        ),

        dpi=300

    )

    plt.close()


# ==========================================================
# Main
# ==========================================================

def main():

    df, LABEL_COLUMN = load_dataset()

    labels, encoder = encode_labels(df, LABEL_COLUMN)

    (
        train_paths,
        val_paths,
        test_paths,
        train_labels,
        val_labels,
        test_labels

    ) = split_dataset(df, labels)

    train_ds = create_dataset(
        train_paths,
        train_labels,
        training=True
    )

    val_ds = create_dataset(
        val_paths,
        val_labels
    )

    test_ds = create_dataset(
        test_paths,
        test_labels
    )

    class_weights = get_class_weights(
        train_labels
    )

    model = build_model(
        len(encoder.classes_)
    )

    model.summary()

    history1, history2 = train_model(

        model,

        train_ds,

        val_ds,

        class_weights

    )

    best_model = tf.keras.models.load_model(

        os.path.join(
            RESULTS_DIR,
            "best_model.keras"
        )

    )

    cm = evaluate_model(

        best_model,

        test_ds,

        encoder

    )

    save_confusion_matrix(

        cm,

        encoder

    )

    plot_training(

        history1,

        history2

    )

    print("\n" + "="*60)
    print("Training Completed Successfully")
    print("="*60)

    print("\nResults Saved To:")

    print(RESULTS_DIR)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()