import os
import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt

# Project root
sys.path.append(os.path.abspath("."))

from ml_model.image_encoder import EfficientNetPreprocess

# -----------------------------
# Paths
# -----------------------------

MODEL_PATH = "results/agrofedvision_fusion_model.keras"
DATA_PATH = "data/multimodal_dataset.csv"

# -----------------------------
# Load Model
# -----------------------------

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "EfficientNetPreprocess": EfficientNetPreprocess
    }
)

print("Model loaded.")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Load Encoder
# -----------------------------

label_encoder = joblib.load(
    "results/fusion_label_encoder.pkl"
)

# -----------------------------
# Features
# -----------------------------

ignore_cols = [
    "image_path",
    "leaf_label",
    "crop_health"
]

uav_cols = [
    "NDVI_Mean",
    "NDVI_Std",
    "NDRE_Mean",
    "NDRE_Std"
]

feature_cols = [
    c for c in df.columns
    if c not in ignore_cols
]

sensor_cols = [
    c for c in feature_cols
    if c not in uav_cols
]

# -----------------------------
# Sensor Data
# -----------------------------

sensor_data = df[sensor_cols].values.astype(np.float32)

scaler = joblib.load(
    "results/fusion_sensor_scaler.pkl"
)

sensor_data = scaler.transform(sensor_data)

# -----------------------------
# UAV Data
# -----------------------------

uav_data = df[uav_cols].values.astype(np.float32)

# -----------------------------
# Image Data
# -----------------------------

IMG_SIZE = 224

images = []

print("Loading images...")

for path in df["image_path"]:

    img = load_img(
        path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img = img_to_array(img)

    images.append(img)

images = np.array(images)

# -----------------------------
# Labels
# -----------------------------

y_true = label_encoder.transform(
    df["crop_health"]
)

# -----------------------------
# Prediction
# -----------------------------

print("Predicting...")

y_prob = model.predict(
    {
        "sensor_input": sensor_data,
        "uav_input": uav_data,
        "image_input": images
    },
    verbose=1
)

y_pred = np.argmax(
    y_prob,
    axis=1
)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix\n")
print(cm)

# -----------------------------
# Plot
# -----------------------------

labels = label_encoder.classes_

plt.figure(figsize=(8,6))

plt.imshow(cm)

plt.title(
    "AgroFedVision Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    np.arange(len(labels)),
    labels,
    rotation=20
)

plt.yticks(
    np.arange(len(labels)),
    labels
)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")

plt.tight_layout()

os.makedirs(
    "results",
    exist_ok=True
)

save_path = "results/confusion_matrix.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSaved: {save_path}")

plt.show()