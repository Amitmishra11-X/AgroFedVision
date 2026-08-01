import os
import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add project root
sys.path.append(os.path.abspath("."))

from ml_model.image_encoder import EfficientNetPreprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = "results/agrofedvision_fusion_model.keras"
DATA_PATH = "data/multimodal_dataset.csv"

# --------------------------------------------------
# Load Model
# --------------------------------------------------

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "EfficientNetPreprocess": EfficientNetPreprocess
    }
)

print("Model Loaded Successfully")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(f"\nDataset Shape: {df.shape}")

# --------------------------------------------------
# Load Label Encoder
# --------------------------------------------------

label_encoder = joblib.load(
    "results/fusion_label_encoder.pkl"
)

# --------------------------------------------------
# Feature Columns
# --------------------------------------------------

ignore_cols = [
    "image_path",
    "leaf_label",
    "crop_health"
]

feature_cols = [
    c for c in df.columns
    if c not in ignore_cols
]

uav_cols = [
    "NDVI_Mean",
    "NDVI_Std",
    "NDRE_Mean",
    "NDRE_Std"
]

sensor_cols = [
    c for c in feature_cols
    if c not in uav_cols
]

print("\nSensor Features:", len(sensor_cols))
print("UAV Features:", len(uav_cols))

# --------------------------------------------------
# Prepare Inputs
# --------------------------------------------------

sensor_data = df[sensor_cols].values.astype(np.float32)
uav_data = df[uav_cols].values.astype(np.float32)

# Scale sensors
scaler = joblib.load(
    "results/fusion_sensor_scaler.pkl"
)

sensor_data = scaler.transform(sensor_data)

# --------------------------------------------------
# Images
# --------------------------------------------------

IMG_SIZE = 224

images = []

print("\nLoading Images...")

for path in df["image_path"]:

    img = load_img(
        path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img = img_to_array(img)

    images.append(img)

images = np.array(images)

print("Images Loaded:", len(images))

# --------------------------------------------------
# Labels
# --------------------------------------------------

y_true = label_encoder.transform(
    df["crop_health"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

print("\nRunning Prediction...")

y_prob = model.predict(
    {
        "sensor_input": sensor_data,
        "uav_input": uav_data,
        "image_input": images
    },
    batch_size=16,
    verbose=1
)

y_pred = np.argmax(y_prob, axis=1)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print("\n" + "="*60)
print("AGROFEDVISION EVALUATION")
print("="*60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# --------------------------------------------------
# Per-Class Metrics
# --------------------------------------------------

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# Confusion Matrix


cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix\n")
print(cm)

print("\nClass Labels:")

for i, label in enumerate(label_encoder.classes_):
    print(i, label)

print("\n" + "="*60)
print("Evaluation Complete")
print("="*60)