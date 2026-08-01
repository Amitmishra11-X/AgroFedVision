import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import sys

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

sys.path.append("ml_model")

from image_encoder import EfficientNetPreprocess

# =====================================================
# CONFIG
# =====================================================

DATASET = "data/multimodal_dataset.csv"

CENTRAL_MODEL = (
    "results/agrofedvision_fusion_model.keras"
)

FED_MODEL = (
    "federated_multimodal/global_model/global_round_3.keras"
)

IMG_SIZE = 224

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(DATASET)

sensor_cols = joblib.load(
    "results/fusion_sensor_cols.pkl"
)

scaler = joblib.load(
    "results/fusion_sensor_scaler.pkl"
)

label_encoder = joblib.load(
    "results/fusion_label_encoder.pkl"
)

uav_cols = [
    "NDVI_Mean",
    "NDVI_Std",
    "NDRE_Mean",
    "NDRE_Std"
]

print("Loading Images...")

images = []

for path in df["image_path"]:

    try:

        img = load_img(
            path,
            target_size=(224,224)
        )

        img = img_to_array(img)

        images.append(img)

    except:

        images.append(
            np.zeros(
                (224,224,3),
                dtype=np.float32
            )
        )

X_img = np.array(
    images,
    dtype=np.float32
)

X_sensor = scaler.transform(
    df[sensor_cols]
)

X_uav = df[uav_cols].values

y_true = label_encoder.transform(
    df["crop_health"]
)

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(model_path, name):

    print("\n" + "="*60)
    print(name)
    print("="*60)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "EfficientNetPreprocess":
            EfficientNetPreprocess
        }
    )

    probs = model.predict(
        {
            "sensor_input": X_sensor,
            "uav_input": X_uav,
            "image_input": X_img
        },
        verbose=1
    )

    y_pred = np.argmax(
        probs,
        axis=1
    )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    print(
        f"\nAccuracy: {acc:.4f}"
    )

    print(
        "\nClassification Report\n"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=
            label_encoder.classes_
        )
    )

    return acc

# =====================================================
# RUN
# =====================================================

central_acc = evaluate_model(
    CENTRAL_MODEL,
    "CENTRALIZED MODEL"
)

fed_acc = evaluate_model(
    FED_MODEL,
    "FEDPROX MODEL"
)

print("\n" + "="*60)
print("COMPARISON")
print("="*60)

print(
    f"Centralized : {central_acc:.4f}"
)

print(
    f"FedProx     : {fed_acc:.4f}"
)

print(
    f"Difference  : {(fed_acc-central_acc):.4f}"
)