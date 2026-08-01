"""
train_sensor_only.py

Train AgroFedVision Sensor Transformer only.

Input:
    Sensor Features

Output:
    Crop Health

Author:
    AgroFedVision
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sensor_transformer import build_sensor_transformer

# ==========================================================
# CONFIG
# ==========================================================

CSV_FILE = "data/multimodal_dataset.csv"

RESULTS_DIR = "results/sensor_only"

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 3e-4

RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================================
# UAV Columns
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
# Load Dataset
# ==========================================================

print("\nLoading Dataset...")

df = pd.read_csv(CSV_FILE)

print(df.shape)

# ==========================================================
# Label Encoder
# ==========================================================

encoder = LabelEncoder()

y = encoder.fit_transform(df["crop_health"])

num_classes = len(encoder.classes_)

print("\nClasses")

for c in encoder.classes_:

    print(c)

# ==========================================================
# Sensor Columns
# ==========================================================

sensor_cols = [

    c

    for c in df.columns

    if c not in SKIP_COLS

]

print("\nSensor Features")

print(sensor_cols)

# ==========================================================
# Scale Sensor Features
# ==========================================================

scaler = MinMaxScaler()

X = scaler.fit_transform(

    df[sensor_cols]

).astype(np.float32)

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    stratify=y,

    random_state=RANDOM_STATE

)

print("\nTrain")

print(X_train.shape)

print("\nTest")

print(X_test.shape)

# ==========================================================
# TensorFlow Dataset
# ==========================================================

train_ds = tf.data.Dataset.from_tensor_slices(

    (

        X_train,

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

        X_test,

        y_test

    )

)

test_ds = test_ds.batch(

    BATCH_SIZE

).prefetch(

    tf.data.AUTOTUNE

)
# ==========================================================
# BUILD SENSOR TRANSFORMER
# ==========================================================

print("\n======================================")
print("Building Sensor Transformer...")
print("======================================")

sensor_encoder = build_sensor_transformer(
    num_features=X_train.shape[1],
    d_model=64,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout=0.1
)

inputs = sensor_encoder.input

features = sensor_encoder.output

# ==========================================================
# Classification Head
# ==========================================================

x = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="classifier_dense1"
)(features)

x = tf.keras.layers.BatchNormalization(
    name="classifier_bn1"
)(x)

x = tf.keras.layers.Dropout(
    0.30,
    name="classifier_drop1"
)(x)

x = tf.keras.layers.Dense(
    64,
    activation="relu",
    name="classifier_dense2"
)(x)

x = tf.keras.layers.BatchNormalization(
    name="classifier_bn2"
)(x)

x = tf.keras.layers.Dropout(
    0.30,
    name="classifier_drop2"
)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="crop_health_output"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="AgroFedVision_SensorOnly"
)

print("\nModel Summary\n")

model.summary()

# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
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
        filepath=os.path.join(
            RESULTS_DIR,
            "best_sensor_model.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True
    )

]

print("\n======================================")
print("Model Compiled Successfully")
print("======================================")
# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\n======================================")
print("Training Sensor Transformer")
print("======================================")

history = model.fit(

    train_ds,

    validation_data=test_ds,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1

)

# ==========================================================
# EVALUATE
# ==========================================================

print("\n======================================")
print("Evaluating Model")
print("======================================")

loss, accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n======================================")
print("SENSOR ONLY RESULTS")
print("======================================")

print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")

# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    os.path.join(
        RESULTS_DIR,
        "sensor_only_model.keras"
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

history_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "training_history.csv"
    ),

    index=False

)

# ==========================================================
# PLOT TRAINING CURVE
# ==========================================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.plot(
    history.history["accuracy"],
    label="Train"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.subplot(1,2,2)

plt.plot(
    history.history["loss"],
    label="Train"
)

plt.plot(
    history.history["val_loss"],
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
        "sensor_training_curve.png"
    ),

    dpi=300

)

plt.close()

# ==========================================================
# SAVE METRICS
# ==========================================================

metrics = {

    "accuracy": float(accuracy),

    "loss": float(loss),

    "epochs": len(history.history["loss"]),

    "num_sensor_features": len(sensor_cols),

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

print("\n======================================")
print("Saved Files")
print("======================================")

print("sensor_only_model.keras")

print("sensor_scaler.pkl")

print("label_encoder.pkl")

print("sensor_columns.pkl")

print("training_history.csv")

print("sensor_training_curve.png")

print("metrics.json")

print("\n======================================")
print("Sensor Only Training Completed")
print("======================================")