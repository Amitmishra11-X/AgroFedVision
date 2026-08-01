"""
==============================================================
AgroFedVision V4
Universal Paddy Disease Classification Pipeline
Author : Amit Mishra
TensorFlow : 2.21
==============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os

import pandas as pd
import random
import warnings

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

import seaborn as sns
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

warnings.filterwarnings("ignore")

# ==========================================================
# REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# GPU INFO
# ==========================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("=" * 60)
    print("GPU DETECTED")
    print("=" * 60)

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(gpu)

else:
    print("=" * 60)
    print("Running on CPU")
    print("=" * 60)

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# CHANGE ONLY THIS PATH

DATASET_PATH = r"D:\Download1\archive\paddy-disease-classification\train_images"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")

for folder in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    FIGURE_DIR,
    LOG_DIR,
    REPORT_DIR,
    PREDICTION_DIR,
]:
    os.makedirs(folder, exist_ok=True)

# ==========================================================
# TRAINING PARAMETERS
# ==========================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.20

STAGE1_EPOCHS = 10

STAGE2_EPOCHS = 40

LEARNING_RATE_STAGE1 = 1e-3

LEARNING_RATE_STAGE2 = 1e-5

LABEL_SMOOTHING = 0.10

AUTOTUNE = tf.data.AUTOTUNE

# ==========================================================
# LOAD DATASET
# ==========================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

CLASS_NAMES = train_dataset.class_names

NUM_CLASSES = len(CLASS_NAMES)

print("=" * 60)
print("Detected Classes")
print("=" * 60)

for i, cls in enumerate(CLASS_NAMES):
    print(f"{i} : {cls}")

print("=" * 60)
print("Total Classes :", NUM_CLASSES)

# ==========================================================
# PREPROCESS
# ==========================================================

# EfficientNetV2 preprocessing
train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
    num_parallel_calls=AUTOTUNE,
)

val_dataset = val_dataset.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
    num_parallel_calls=AUTOTUNE,
)

# ==========================================================
# EfficientNetV2 Preprocessing
# ==========================================================

train_dataset = train_dataset.map(
    lambda x, y: (
        preprocess_input(tf.cast(x, tf.float32)),
        y
    ),
    num_parallel_calls=AUTOTUNE,
)

val_dataset = val_dataset.map(
    lambda x, y: (
        preprocess_input(tf.cast(x, tf.float32)),
        y
    ),
    num_parallel_calls=AUTOTUNE,
)

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

# ==========================================================
# CLASS WEIGHTS
# ==========================================================

labels = []

for _, y in train_dataset:
    labels.extend(np.argmax(y.numpy(), axis=1))

labels = np.array(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels,
)

CLASS_WEIGHTS = {
    i: float(w)
    for i, w in enumerate(weights)
}

print("=" * 60)
print("CLASS WEIGHTS")
print("=" * 60)

for i, w in CLASS_WEIGHTS.items():
    print(CLASS_NAMES[i], ":", round(w, 3))

print("=" * 60)

print("PART 1 COMPLETED")
# ==========================================================
# DATA AUGMENTATION
# ==========================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal_and_vertical"
        ),

        tf.keras.layers.RandomRotation(
            0.15
        ),

        tf.keras.layers.RandomZoom(
            0.15
        ),

        tf.keras.layers.RandomContrast(
            0.20
        ),

        tf.keras.layers.RandomTranslation(
            0.10,
            0.10
        ),
    ],
    name="DataAugmentation",
)

# ==========================================================
# APPLY AUGMENTATION
# ==========================================================

def augment(images, labels):

    images = data_augmentation(
        images,
        training=True
    )

    return images, labels


train_dataset = train_dataset.map(
    augment,
    num_parallel_calls=AUTOTUNE
)

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

# ==========================================================
# BUILD MODEL
# ==========================================================

print()
print("=" * 60)
print("Building EfficientNetV2-S")
print("=" * 60)

base_model = EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3),
    pooling="avg",
)

# ==========================================================
# STAGE 1
# ==========================================================

base_model.trainable = False

inputs = tf.keras.Input(
    shape=(224,224,3),
    name="InputImage"
)

x = base_model(
    inputs,
    training=False
)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)

x = tf.keras.layers.Dense(
    1024,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(
        1e-4
    )
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(
    0.40
)(x)

x = tf.keras.layers.Dense(
    512,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(
        1e-4
    )
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    dtype="float32",
    name="Predictions"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="AgroFedVision_Paddy_V4"
)

print()
model.summary()

# ==========================================================
# COMPILE
# ==========================================================

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE_STAGE1,
    weight_decay=1e-4,
)

loss_function = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=LABEL_SMOOTHING
)

model.compile(

    optimizer=optimizer,

    loss=loss_function,

    metrics=[

        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )

    ]

)

print()
print("=" * 60)
print("MODEL COMPILED")
print("=" * 60)

# ==========================================================
# CALLBACKS
# ==========================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath=os.path.join(
        CHECKPOINT_DIR,
        "best_paddy_model.keras"
    ),

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1

)

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=8,

    restore_best_weights=True,

    verbose=1

)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=3,

    min_lr=1e-7,

    verbose=1

)

csv_logger = tf.keras.callbacks.CSVLogger(

    os.path.join(
        LOG_DIR,
        "training_log.csv"
    )

)

tensorboard = tf.keras.callbacks.TensorBoard(

    log_dir=LOG_DIR,

    histogram_freq=1

)

callbacks = [

    checkpoint,

    early_stop,

    reduce_lr,

    csv_logger,

    tensorboard

]

print()
print("=" * 60)
print("CALLBACKS READY")
print("=" * 60)

# ==========================================================
# VERIFY MODEL
# ==========================================================

print()

print("Model Name        :", model.name)

print("Backbone          : EfficientNetV2-S")

print("Classes           :", NUM_CLASSES)

print("Batch Size        :", BATCH_SIZE)

print("Stage 1 Epochs    :", STAGE1_EPOCHS)

print("Stage 2 Epochs    :", STAGE2_EPOCHS)

print("Learning Rate     :", LEARNING_RATE_STAGE1)

print()

print("=" * 60)
print("PART 2 COMPLETED")
print("=" * 60)
# ==========================================================
# STAGE 1 TRAINING
# ==========================================================

print()
print("=" * 70)
print("STAGE 1 : TRAINING CLASSIFICATION HEAD")
print("=" * 70)

history_stage1 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=STAGE1_EPOCHS,

    class_weight=CLASS_WEIGHTS,

    callbacks=callbacks,

    verbose=1,

)

print()
print("=" * 70)
print("STAGE 1 TRAINING COMPLETED")
print("=" * 70)

# ==========================================================
# LOAD BEST MODEL
# ==========================================================

print()
print("Loading Best Model...")

best_model_path = os.path.join(
    CHECKPOINT_DIR,
    "best_paddy_model.keras"
)

model = tf.keras.models.load_model(best_model_path)

print("Best Model Loaded Successfully")

# ==========================================================
# SAVE STAGE 1 MODEL
# ==========================================================

stage1_model_path = os.path.join(
    CHECKPOINT_DIR,
    "stage1_final.keras"
)

model.save(stage1_model_path)

print("Stage 1 Model Saved")
print(stage1_model_path)

# ==========================================================
# TRAINING SUMMARY
# ==========================================================

best_epoch = np.argmax(
    history_stage1.history["val_accuracy"]
) + 1

best_train_accuracy = max(
    history_stage1.history["accuracy"]
)

best_val_accuracy = max(
    history_stage1.history["val_accuracy"]
)

best_train_loss = min(
    history_stage1.history["loss"]
)

best_val_loss = min(
    history_stage1.history["val_loss"]
)

print()
print("=" * 70)
print("STAGE 1 SUMMARY")
print("=" * 70)

print(f"Best Epoch          : {best_epoch}")
print(f"Train Accuracy      : {best_train_accuracy:.4f}")
print(f"Validation Accuracy : {best_val_accuracy:.4f}")
print(f"Train Loss          : {best_train_loss:.4f}")
print(f"Validation Loss     : {best_val_loss:.4f}")

print("=" * 70)

# ==========================================================
# SAVE TRAINING CURVES
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history_stage1.history["accuracy"],
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    history_stage1.history["val_accuracy"],
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Stage 1 Accuracy")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "stage1_accuracy.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history_stage1.history["loss"],
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    history_stage1.history["val_loss"],
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Stage 1 Loss")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "stage1_loss.png"
    ),
    dpi=300
)

plt.close()

print()
print("Training curves saved successfully.")

# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

history_df = pd.DataFrame(
    history_stage1.history
)

history_df.to_csv(

    os.path.join(
        REPORT_DIR,
        "stage1_training_history.csv"
    ),

    index=False

)

print("Training history saved.")

print()

print("=" * 70)
print("PART 3 COMPLETED")
print("=" * 70)
# ==========================================================
# STAGE 2 : FINE TUNING
# ==========================================================

print()
print("=" * 70)
print("STAGE 2 : FINE TUNING")
print("=" * 70)

# ----------------------------------------------------------
# Unfreeze Entire Backbone
# ----------------------------------------------------------

base_model.trainable = True

# ----------------------------------------------------------
# Freeze ALL BatchNormalization Layers
# ----------------------------------------------------------

for layer in base_model.layers:

    if isinstance(layer, tf.keras.layers.BatchNormalization):

        layer.trainable = False

# ----------------------------------------------------------
# Fine Tune Only Last 25 Layers
# ----------------------------------------------------------

FINE_TUNE_AT = len(base_model.layers) - 25

for layer in base_model.layers[:FINE_TUNE_AT]:

    layer.trainable = False

print()
print("Total Backbone Layers :", len(base_model.layers))
print("Fine Tune Starts From :", FINE_TUNE_AT)

# ----------------------------------------------------------
# Recompile
# ----------------------------------------------------------

optimizer = tf.keras.optimizers.AdamW(

    learning_rate=LEARNING_RATE_STAGE2,

    weight_decay=1e-5

)

loss_function = tf.keras.losses.CategoricalCrossentropy(

    label_smoothing=LABEL_SMOOTHING

)

model.compile(

    optimizer=optimizer,

    loss=loss_function,

    metrics=[

        "accuracy",

        tf.keras.metrics.Precision(name="precision"),

        tf.keras.metrics.Recall(name="recall"),

        tf.keras.metrics.AUC(name="auc")

    ]

)

print()
print("Fine-Tuning Model Compiled")

# ==========================================================
# STAGE 2 TRAINING
# ==========================================================

history_stage2 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    initial_epoch=STAGE1_EPOCHS,

    epochs=STAGE1_EPOCHS + STAGE2_EPOCHS,

    class_weight=CLASS_WEIGHTS,

    callbacks=callbacks,

    verbose=1

)

print()
print("=" * 70)
print("FINE TUNING COMPLETED")
print("=" * 70)

# ==========================================================
# LOAD BEST MODEL
# ==========================================================

print()
print("Loading Best Fine-Tuned Model...")

model = tf.keras.models.load_model(

    os.path.join(
        CHECKPOINT_DIR,
        "best_paddy_model.keras"
    )

)

print("Best Model Loaded Successfully")

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

FINAL_MODEL_PATH = os.path.join(

    CHECKPOINT_DIR,

    "AgroFedVision_Paddy_V4.keras"

)

model.save(FINAL_MODEL_PATH)

print()
print("Final Model Saved")

print(FINAL_MODEL_PATH)

# ==========================================================
# STAGE 2 SUMMARY
# ==========================================================

best_epoch = np.argmax(

    history_stage2.history["val_accuracy"]

) + 1

best_accuracy = np.max(

    history_stage2.history["val_accuracy"]

)

best_loss = np.min(

    history_stage2.history["val_loss"]

)

print()
print("=" * 70)
print("FINAL TRAINING SUMMARY")
print("=" * 70)

print(f"Best Epoch           : {best_epoch}")

print(f"Best Accuracy        : {best_accuracy:.4f}")

print(f"Best Validation Loss : {best_loss:.4f}")

print("=" * 70)

# ==========================================================
# SAVE STAGE 2 HISTORY
# ==========================================================

history_stage2_df = pd.DataFrame(

    history_stage2.history

)

history_stage2_df.to_csv(

    os.path.join(

        REPORT_DIR,

        "stage2_training_history.csv"

    ),

    index=False

)

print()
print("Stage 2 History Saved")

# ==========================================================
# SAVE ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history_stage2.history["accuracy"],

    label="Train Accuracy",

    linewidth=2

)

plt.plot(

    history_stage2.history["val_accuracy"],

    label="Validation Accuracy",

    linewidth=2

)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Stage 2 Accuracy")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURE_DIR,

        "stage2_accuracy.png"

    ),

    dpi=300

)

plt.close()

# ==========================================================
# SAVE LOSS GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    history_stage2.history["loss"],

    label="Train Loss",

    linewidth=2

)

plt.plot(

    history_stage2.history["val_loss"],

    label="Validation Loss",

    linewidth=2

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Stage 2 Loss")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURE_DIR,

        "stage2_loss.png"

    ),

    dpi=300

)

plt.close()

print()
print("Stage 2 Graphs Saved")

print()
print("=" * 70)
print("PART 4 COMPLETED")
print("=" * 70)
# ==========================================================
# FINAL EVALUATION
# ==========================================================

print()
print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

evaluation = model.evaluate(
    val_dataset,
    verbose=1
)

print()

for name, value in zip(model.metrics_names, evaluation):
    print(f"{name:15s}: {value:.4f}")

# ==========================================================
# PREDICTIONS
# ==========================================================

y_true = []
y_pred = []

for images, labels in val_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_pred.extend(
        np.argmax(predictions, axis=1)
    )

    y_true.extend(
        np.argmax(labels.numpy(), axis=1)
    )

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    digits=4,

    output_dict=True

)

report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(
    REPORT_DIR,
    "classification_report.csv"
)

report_df.to_csv(report_path)

print()
print("Classification Report Saved")
print(report_path)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10,8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=CLASS_NAMES,

    yticklabels=CLASS_NAMES

)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURE_DIR,

        "confusion_matrix.png"

    ),

    dpi=300

)

plt.close()

print("Confusion Matrix Saved")

# ==========================================================
# PER CLASS ACCURACY
# ==========================================================

print()
print("=" * 70)
print("PER CLASS ACCURACY")
print("=" * 70)

per_class_accuracy = {}

for i, class_name in enumerate(CLASS_NAMES):

    total = np.sum(cm[i])

    correct = cm[i][i]

    accuracy = 100 * correct / total if total > 0 else 0

    per_class_accuracy[class_name] = accuracy

    print(f"{class_name:30s} {accuracy:.2f}%")

accuracy_df = pd.DataFrame(

    list(per_class_accuracy.items()),

    columns=["Class", "Accuracy"]

)

accuracy_df.to_csv(

    os.path.join(

        REPORT_DIR,

        "per_class_accuracy.csv"

    ),

    index=False

)

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

prediction_df = pd.DataFrame({

    "True_Label": [CLASS_NAMES[i] for i in y_true],

    "Predicted_Label": [CLASS_NAMES[i] for i in y_pred]

})

prediction_df.to_csv(

    os.path.join(

        PREDICTION_DIR,

        "predictions.csv"

    ),

    index=False

)

print()
print("Predictions Saved")

# ==========================================================
# SAVE COMPLETE MODEL
# ==========================================================

complete_model_path = os.path.join(

    CHECKPOINT_DIR,

    "AgroFedVision_Paddy_Final.keras"

)

model.save(complete_model_path)

print()
print("Complete Model Saved")

print(complete_model_path)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Backbone            : EfficientNetV2-S")

print("Image Size          :", IMAGE_SIZE)

print("Classes             :", NUM_CLASSES)

print("Stage 1 Epochs      :", STAGE1_EPOCHS)

print("Stage 2 Epochs      :", STAGE2_EPOCHS)

print()

print("Outputs Folder")

print(OUTPUT_DIR)

print()

print("Saved Files")

print("✓ Best Model")

print("✓ Final Model")

print("✓ Stage 1 History")

print("✓ Stage 2 History")

print("✓ Accuracy Graphs")

print("✓ Loss Graphs")

print("✓ Confusion Matrix")

print("✓ Classification Report")

print("✓ Predictions CSV")

print("✓ Per-Class Accuracy")

print()

print("=" * 70)
print("AgroFedVision V4 Training Finished")
print("=" * 70)