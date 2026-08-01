"""
train_image_sensor.py

AgroFedVision

Image + Sensor Ablation Model

Uses:
    Leaf Images
    Sensor Features

Predicts:
    Crop Health
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from image_encoder import build_image_encoder
from sensor_transformer import build_sensor_transformer

# ==========================================================
# CONFIG
# ==========================================================

CSV_FILE = "data/multimodal_dataset.csv"

RESULTS_DIR = "results/image_sensor"

IMG_SIZE = 224

BATCH_SIZE = 16

EPOCHS_STAGE1 = 25

EPOCHS_STAGE2 = 10

LR = 3e-4

RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================================
# UAV COLUMNS
# (Ignored in this experiment)
# ==========================================================

UAV_COLS = [

    "NDVI_Mean",

    "NDVI_Std",

    "NDRE_Mean",

    "NDRE_Std"

]

SKIP_COLS = [

    "image_path",

    "leaf_label",

    "crop_health"

] + UAV_COLS

# ==========================================================
# LOAD DATASET
# ==========================================================

print("\nLoading Dataset...")

df = pd.read_csv(CSV_FILE)

print(df.shape)

# ==========================================================
# LABEL ENCODER
# ==========================================================

encoder = LabelEncoder()

y = encoder.fit_transform(

    df["crop_health"]

)

num_classes = len(

    encoder.classes_

)

print("\nClasses")

for c in encoder.classes_:

    print(c)

# ==========================================================
# SENSOR FEATURES
# ==========================================================

sensor_cols = [

    c

    for c in df.columns

    if c not in SKIP_COLS

]

print("\nSensor Features")

print(sensor_cols)

scaler = MinMaxScaler()

X_sensor = scaler.fit_transform(

    df[sensor_cols]

).astype(np.float32)

# ==========================================================
# LOAD IMAGES
# ==========================================================

print("\nLoading Images...")

images = []

labels = []

sensor_data = []

for i, row in df.iterrows():

    try:

        img = load_img(

            row["image_path"],

            target_size=(IMG_SIZE, IMG_SIZE)

        )

        img = img_to_array(img)

        images.append(img)

        sensor_data.append(

            X_sensor[i]

        )

        labels.append(

            y[i]

        )

    except:

        pass

X_image = np.array(

    images,

    dtype=np.float32

)

X_sensor = np.array(

    sensor_data,

    dtype=np.float32

)

y = np.array(

    labels

)

print("\nLoaded Images :", len(X_image))

print("Sensor Shape  :", X_sensor.shape)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

idx = np.arange(

    len(y)

)

train_idx, test_idx = train_test_split(

    idx,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y

)

X_sensor_train = X_sensor[train_idx]
X_sensor_test = X_sensor[test_idx]

X_image_train = X_image[train_idx]
X_image_test = X_image[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# ==========================================================
# TF DATASET
# ==========================================================

train_ds = tf.data.Dataset.from_tensor_slices(

(

{

"sensor_input": X_sensor_train,

"image_input": X_image_train

},

y_train

)

)

train_ds = train_ds.shuffle(

len(y_train)

).batch(

BATCH_SIZE

).prefetch(

tf.data.AUTOTUNE

)

test_ds = tf.data.Dataset.from_tensor_slices(

(

{

"sensor_input": X_sensor_test,

"image_input": X_image_test

},

y_test

)

)

test_ds = test_ds.batch(

BATCH_SIZE

).prefetch(

tf.data.AUTOTUNE
)

print("\nDataset Ready.")
# ==========================================================
# BUILD IMAGE + SENSOR MODEL
# ==========================================================

print("\n======================================")
print("Building Image + Sensor Model...")
print("======================================")

# ----------------------------------------------------------
# SENSOR BRANCH
# ----------------------------------------------------------

sensor_encoder = build_sensor_transformer(
    num_features=X_sensor_train.shape[1],
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout=0.1
)

sensor_input = sensor_encoder.input
sensor_features = sensor_encoder.output

# ----------------------------------------------------------
# IMAGE BRANCH
# ----------------------------------------------------------

image_encoder = build_image_encoder(
    embed_dim=256,
    dropout=0.30,
    trainable_backbone=False
)

image_input = image_encoder.input
image_features = image_encoder.output

# ----------------------------------------------------------
# FEATURE FUSION
# ----------------------------------------------------------

fusion = tf.keras.layers.Concatenate(
    name="feature_fusion"
)(
    [
        sensor_features,
        image_features
    ]
)

print("\nFusion Feature Dimension:", fusion.shape)

# ----------------------------------------------------------
# CLASSIFICATION HEAD
# ----------------------------------------------------------

x = tf.keras.layers.Dense(
    256,
    activation="relu",
    name="head_dense1"
)(fusion)

x = tf.keras.layers.BatchNormalization(
    name="head_bn1"
)(x)

x = tf.keras.layers.Dropout(
    0.30,
    name="head_drop1"
)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="head_dense2"
)(x)

x = tf.keras.layers.BatchNormalization(
    name="head_bn2"
)(x)

x = tf.keras.layers.Dropout(
    0.30,
    name="head_drop2"
)(x)

output = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="crop_health_output"
)(x)

# ----------------------------------------------------------
# COMPLETE MODEL
# ----------------------------------------------------------

model = tf.keras.Model(

    inputs=[
        sensor_input,
        image_input
    ],

    outputs=output,

    name="AgroFedVision_ImageSensor"

)

print("\n======================================")
print("MODEL SUMMARY")
print("======================================")

model.summary()

print("\nTrainable Parameters:")

print(
    np.sum(
        [
            np.prod(v.shape)
            for v in model.trainable_weights
        ]
    )
)
# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LR
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# ==========================================================
# CALLBACKS
# ==========================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=6,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

        verbose=1

    ),

    tf.keras.callbacks.ModelCheckpoint(

        os.path.join(

            RESULTS_DIR,

            "best_image_sensor.keras"

        ),

        monitor="val_accuracy",

        save_best_only=True

    )

]

# ==========================================================
# STAGE 1
# ==========================================================

print("\n======================================")
print("Stage 1 : Frozen EfficientNet")
print("======================================")

history = model.fit(

    train_ds,

    validation_data=test_ds,

    epochs=EPOCHS_STAGE1,

    callbacks=callbacks,

    verbose=1

)

stage1_best = max(

    history.history["val_accuracy"]

)

print("\nStage 1 Best Accuracy :", stage1_best)

# ==========================================================
# STAGE 2
# ==========================================================

if stage1_best >= 0.35:

    print("\n======================================")
    print("Fine Tuning EfficientNet")
    print("======================================")

    image_encoder.trainable = True

    backbone = None

    for layer in image_encoder.layers:

        if "efficientnetb0" in layer.name.lower():

            backbone = layer

            break

    if backbone is not None:

        for l in backbone.layers[:-20]:

            l.trainable = False

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=1e-5

        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    history2 = model.fit(

        train_ds,

        validation_data=test_ds,

        epochs=EPOCHS_STAGE2,

        callbacks=callbacks,

        verbose=1

    )

else:

    history2 = None
# ==========================================================
# EVALUATION
# ==========================================================

print("\n======================================")
print("Evaluating Image + Sensor Model")
print("======================================")

loss, accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n======================================")
print("IMAGE + SENSOR RESULTS")
print("======================================")

print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")

# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    os.path.join(
        RESULTS_DIR,
        "image_sensor_model.keras"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        RESULTS_DIR,
        "sensor_scaler.pkl"
    )
)

joblib.dump(
    encoder,
    os.path.join(
        RESULTS_DIR,
        "label_encoder.pkl"
    )
)

joblib.dump(
    sensor_cols,
    os.path.join(
        RESULTS_DIR,
        "sensor_columns.pkl"
    )
)

# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

history_df = pd.DataFrame(history.history)

if history2 is not None:

    history2_df = pd.DataFrame(history2.history)

    history_df = pd.concat(
        [history_df, history2_df],
        ignore_index=True
    )

history_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "training_history.csv"
    ),

    index=False

)

# ==========================================================
# SAVE METRICS
# ==========================================================

metrics = {

    "accuracy": float(accuracy),

    "loss": float(loss),

    "epochs": len(history_df),

    "classes": list(encoder.classes_)

}

import json

with open(

    os.path.join(
        RESULTS_DIR,
        "metrics.json"
    ),

    "w"

) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# ==========================================================
# TRAINING CURVE
# ==========================================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.plot(
    history_df["accuracy"],
    label="Train"
)

plt.plot(
    history_df["val_accuracy"],
    label="Validation"
)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.subplot(1,2,2)

plt.plot(
    history_df["loss"],
    label="Train"
)

plt.plot(
    history_df["val_loss"],
    label="Validation"
)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "image_sensor_training_curve.png"
    ),

    dpi=300

)

plt.close()

print("\n======================================")
print("Saved Files")
print("======================================")

print("image_sensor_model.keras")
print("sensor_scaler.pkl")
print("label_encoder.pkl")
print("sensor_columns.pkl")
print("training_history.csv")
print("metrics.json")
print("image_sensor_training_curve.png")

print("\n======================================")
print("Image + Sensor Training Completed")
print("======================================")
