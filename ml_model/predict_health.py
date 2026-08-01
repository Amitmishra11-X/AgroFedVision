"""
predict_health.py — Single prediction from all three modalities.

Usage:
    python ml_model\\predict_health.py ^
        --leaf path\\to\\leaf.jpg ^
        --sensor N=90,P=42,K=43,temperature=20.8,humidity=82,ph=6.5,rainfall=202.9 ^
        --uav NDVI_Mean=0.22,NDVI_Std=0.18,NDRE_Mean=0.11,NDRE_Std=0.17

Output:
    ============================================================
    AGROFEDVISION — CROP HEALTH ASSESSMENT
    ============================================================
    Prediction   : High_Risk
    Confidence   : 87.3%

    All probabilities:
      Healthy          :  8.4%
      Moderate_Risk    :  4.3%
      High_Risk        : 87.3%   <-- predicted

    Recommendation:
      High Risk detected. Investigate for disease or severe deficiency.
      Apply targeted treatment and re-assess within 7 days.
    ============================================================
"""

import sys
import os
import argparse
import numpy as np
import joblib
import tensorflow as tf
from image_encoder import EfficientNetPreprocess

from tensorflow.keras.preprocessing.image import load_img, img_to_array

RESULTS_DIR = "results"

ADVICE = {
    "Healthy":       "Crop appears healthy. Maintain current irrigation and nutrition schedule.",
    "Moderate_Risk": "Moderate risk detected — likely nutrient deficiency or early-stage stress. "
                     "Check soil NPK levels and review irrigation. Monitor closely for 1 week.",
    "High_Risk":     "High risk detected — possible active disease or severe deficiency. "
                     "Apply targeted treatment immediately and re-assess within 7 days. "
                     "Consider isolating affected plants to prevent spread.",
}


def parse_kv_arg(s):
    result = {}
    for pair in s.split(","):
        k, v = pair.split("=")
        result[k.strip()] = float(v.strip())
    return result


def predict(leaf_path, sensor_dict, uav_dict):
    model = tf.keras.models.load_model(
    os.path.join(
        RESULTS_DIR,
        "agrofedvision_fusion_model.keras"
    ),
    custom_objects={
        "EfficientNetPreprocess": EfficientNetPreprocess
    }
)
    label_enc   = joblib.load(os.path.join(RESULTS_DIR, "fusion_label_encoder.pkl"))
    scaler      = joblib.load(os.path.join(RESULTS_DIR, "fusion_sensor_scaler.pkl"))
    sensor_cols = joblib.load(os.path.join(RESULTS_DIR, "fusion_sensor_cols.pkl"))
    uav_cols    = joblib.load(os.path.join(RESULTS_DIR, "fusion_uav_cols.pkl"))

    # Sensor
    sensor_vec = np.array([[sensor_dict.get(c, 0.0) for c in sensor_cols]], dtype=np.float32)
    sensor_vec = scaler.transform(sensor_vec)

    # UAV
    uav_vec = np.array([[uav_dict.get(c, 0.0) for c in uav_cols]], dtype=np.float32)

    # Image
    img = load_img(leaf_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    probs = model.predict(
        {"sensor_input": sensor_vec,
         "uav_input":    uav_vec,
         "image_input":  img},
        verbose=0
    )[0]

    top_idx  = int(np.argmax(probs))
    top_cls  = label_enc.classes_[top_idx]
    top_conf = float(probs[top_idx]) * 100

    print("\n" + "="*60)
    print("   AGROFEDVISION — CROP HEALTH ASSESSMENT")
    print("="*60)
    print(f"Prediction   : {top_cls}")
    print(f"Confidence   : {top_conf:.1f}%")
    print("\nAll probabilities:")
    for i, cls in enumerate(label_enc.classes_):
        marker = "   <-- predicted" if i == top_idx else ""
        print(f"  {cls:20s}: {probs[i]*100:5.1f}%{marker}")
    print(f"\nRecommendation:\n  {ADVICE[top_cls]}")
    print("="*60 + "\n")

    return top_cls, top_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf",   required=True)
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--uav",    required=True)
    args = parser.parse_args()

    predict(
        leaf_path=args.leaf,
        sensor_dict=parse_kv_arg(args.sensor),
        uav_dict=parse_kv_arg(args.uav)
    )
