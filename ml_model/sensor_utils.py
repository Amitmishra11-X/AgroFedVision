"""
sensor_utils.py

AgroFedVision -- Sensor Prediction (reusable version)

Wraps the trained sensor-only model (from train_sensor_only.py) into
a single-reading prediction function, so predict.py can attach sensor
context to an image prediction.

IMPORTANT: the sensor-only model was validated at ~41.83% accuracy in
the original ablation study -- well above the 33.3% random baseline
for 3 classes, but a much weaker signal than the image branch. It is
surfaced here as supplementary context alongside the image prediction
(consistent with the "fuse predictions, not raw data" architecture),
NOT as an equally-weighted vote. predict.py should present it as
"environmental context", not as a competing diagnosis.
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sensor_recommendations import analyze_sensor_readings

SENSOR_MODEL_DIR = "results/sensor_only"


def load_sensor_artifacts():
    model_path = os.path.join(SENSOR_MODEL_DIR, "sensor_only_model.keras")
    scaler_path = os.path.join(SENSOR_MODEL_DIR, "sensor_scaler.pkl")
    encoder_path = os.path.join(SENSOR_MODEL_DIR, "label_encoder.pkl")
    columns_path = os.path.join(SENSOR_MODEL_DIR, "sensor_columns.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path, columns_path]):
        return None  # sensor model not trained/available -- caller treats as "no context"

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    sensor_cols = joblib.load(columns_path)

    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "sensor_cols": sensor_cols,
    }


def predict_sensor(sensor_reading: dict, artifacts=None):
    """
    sensor_reading: dict mapping sensor column name -> value, e.g.
        {"N": 40, "P": 30, "K": 20, "temperature": 27.5, ...}
    Must include all columns the model was trained on (sensor_cols) --
    missing columns will raise a clear error rather than silently
    guessing a default value.
    """
    if artifacts is None:
        artifacts = load_sensor_artifacts()
    if artifacts is None:
        return None  # no trained sensor model available

    sensor_cols = artifacts["sensor_cols"]
    missing = [c for c in sensor_cols if c not in sensor_reading]
    if missing:
        raise ValueError(f"Sensor reading is missing required fields: {missing}")

    x = pd.DataFrame(
    [sensor_reading],
    columns=sensor_cols
)
    x_scaled = artifacts["scaler"].transform(x)

    probs = artifacts["model"].predict(x_scaled, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_class = artifacts["encoder"].inverse_transform([predicted_idx])[0]
    analysis = analyze_sensor_readings(sensor_reading)

    return {
    "predicted_class": predicted_class,
    "confidence": float(probs[predicted_idx]),
    "class_probabilities": {
        cls: float(p)
        for cls, p in zip(
            artifacts["encoder"].classes_,
            probs
        )
    },

    "analysis": analysis,

    "note":
        "Sensor prediction is supplementary environmental context "
        "used alongside image and UAV analysis."
}