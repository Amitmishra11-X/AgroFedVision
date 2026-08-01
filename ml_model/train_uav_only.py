"""
train_uav_only.py

Train AgroFedVision UAV Only Model

Input:
    NDVI_Mean
    NDVI_Std
    NDRE_Mean
    NDRE_Std

Output:
    Crop Health
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# =============================================

CSV_FILE = "data/multimodal_dataset.csv"

RESULTS_DIR = "results/uav_only"

os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 16

EPOCHS = 30

LR = 3e-4

RANDOM_STATE = 42

# =============================================

print("\nLoading Dataset...")

df = pd.read_csv(CSV_FILE)

encoder = LabelEncoder()

y = encoder.fit_transform(df["crop_health"])

num_classes = len(encoder.classes_)

print("Classes:", encoder.classes_)

UAV_COLS = [

    "NDVI_Mean",

    "NDVI_Std",

    "NDRE_Mean",

    "NDRE_Std"

]

X = df[UAV_COLS].values.astype(np.float32)

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y

)

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
# BUILD UAV MODEL
# ==========================================================

print("\n======================================")
print("Building UAV Model...")
print("======================================")

inputs = tf.keras.Input(
    shape=(4,),
    name="uav_input"
)

# ------------------------------------
# Same UAV branch used in Fusion Model
# ------------------------------------

x = tf.keras.layers.Dense(
    32,
    activation="relu",
    name="uav_dense1"
)(inputs)

x = tf.keras.layers.BatchNormalization(
    name="uav_bn1"
)(x)

x = tf.keras.layers.Dropout(
    0.20
)(x)

x = tf.keras.layers.Dense(
    16,
    activation="relu",
    name="uav_dense2"
)(x)

x = tf.keras.layers.BatchNormalization(
    name="uav_bn2"
)(x)

x = tf.keras.layers.Dropout(
    0.20
)(x)

# ------------------------------------
# Classification Head
# ------------------------------------

x = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="classifier_dense1"
)(x)

x = tf.keras.layers.BatchNormalization(
    name="classifier_bn1"
)(x)

x = tf.keras.layers.Dropout(
    0.30
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
    0.30
)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="crop_health_output"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="AgroFedVision_UAVOnly"
)

model.summary()

# ==========================================================
# COMPILE
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

            "best_uav_model.keras"

        ),

        monitor="val_accuracy",

        save_best_only=True

    )

]

print("\n======================================")
print("Model Ready")
print("======================================")
# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\n======================================")
print("Training UAV Model")
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
print("Evaluating UAV Model")
print("======================================")

loss, accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n======================================")
print("UAV ONLY RESULTS")
print("======================================")

print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy:.4f}")

# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    os.path.join(
        RESULTS_DIR,
        "uav_only_model.keras"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        RESULTS_DIR,
        "uav_scaler.pkl"
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
    UAV_COLS,
    os.path.join(
        RESULTS_DIR,
        "uav_columns.pkl"
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
# TRAINING CURVE
# ==========================================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"], label="Train")

plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.subplot(1,2,2)

plt.plot(history.history["loss"], label="Train")

plt.plot(history.history["val_loss"], label="Validation")

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULTS_DIR,

        "uav_training_curve.png"

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

    "features": UAV_COLS,

    "classes": list(encoder.classes_)

}

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

print("uav_only_model.keras")

print("uav_scaler.pkl")

print("label_encoder.pkl")

print("uav_columns.pkl")

print("training_history.csv")

print("uav_training_curve.png")

print("metrics.json")

print("\n======================================")
print("UAV Only Training Completed")
print("======================================")