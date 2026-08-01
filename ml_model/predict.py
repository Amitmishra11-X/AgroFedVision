"""
predict.py

AgroFedVision Prediction Orchestrator

Runs image prediction, optional UAV analysis,
optional sensor analysis and decision fusion.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import tensorflow as tf

from gradcam import generate_gradcam
from decision_fusion import fuse_predictions

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Required so TensorFlow can deserialize the custom preprocessing layer
import image_encoder

from crop_registry import (
    get_crop_config,
    list_available_crops
)

from recommendations import get_recommendation
from uav_utils import extract_uav_features
from sensor_utils import (
    predict_sensor,
    load_sensor_artifacts
)

# ==========================================================
# Global Cache
# ==========================================================

_MODEL_CACHE = {}
_ENCODER_CACHE = {}
_SENSOR_ARTIFACTS = None


# ==========================================================
# Load Model
# ==========================================================

def _load_model_and_encoder(crop):

    if crop not in _MODEL_CACHE:

        config = get_crop_config(crop)

        print(f"\nLoading {crop} model...")

        _MODEL_CACHE[crop] = tf.keras.models.load_model(
            config["model_path"]
        )

        _ENCODER_CACHE[crop] = joblib.load(
            config["encoder_path"]
        )

    return (
        _MODEL_CACHE[crop],
        _ENCODER_CACHE[crop]
    )


# ==========================================================
# Image Prediction
# ==========================================================

def predict_image(crop, image_path):

    config = get_crop_config(crop)

    model, encoder = _load_model_and_encoder(crop)

    img_size = config["img_size"]

    img = tf.io.read_file(image_path)

    img = tf.image.decode_jpeg(
        img,
        channels=3
    )

    img = tf.image.resize(
        img,
        [img_size, img_size]
    )

    img = tf.cast(
        img,
        tf.float32
    )

    img = tf.expand_dims(
        img,
        axis=0
    )

    probs = model.predict(
        img,
        verbose=0
    )[0]

    idx = int(np.argmax(probs))

    predicted_class = encoder.inverse_transform(
        [idx]
    )[0]

    return {

        "predicted_class": predicted_class,

        "confidence": float(
            probs[idx]
        ),

        "class_probabilities": {

            cls: float(p)

            for cls, p in zip(
                encoder.classes_,
                probs
            )

        }

    }


# ==========================================================
# Complete Prediction Pipeline
# ==========================================================

def predict_and_recommend(

        crop,
        image_path,
        uav_capture_folder=None,
        uav_image_id=None,
        sensor_reading=None
):

    global _SENSOR_ARTIFACTS

    # --------------------------------------------------
    # Image Prediction
    # --------------------------------------------------

    result = predict_image(
        crop,
        image_path
    )

    config = get_crop_config(crop)

    model, _ = _load_model_and_encoder(crop)

    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------

    recommendation = get_recommendation(

        crop,

        result["predicted_class"]

    )

    # --------------------------------------------------
    # UAV
    # --------------------------------------------------

    uav_context = None

    if uav_capture_folder and uav_image_id:

        uav_context = extract_uav_features(

            uav_capture_folder,

            uav_image_id

        )
        print("\nDEBUG UAV")
        print(uav_context)

        if uav_context is None:

            print(

                "[Predict] UAV data not available."

                " Proceeding with Image + Sensor."

            )

    # --------------------------------------------------
    # Sensor
    # --------------------------------------------------

    sensor_context = None

    if sensor_reading is not None:

        if _SENSOR_ARTIFACTS is None:

            _SENSOR_ARTIFACTS = load_sensor_artifacts()

        if _SENSOR_ARTIFACTS is None:

            print(

                "[Predict] Sensor model not found."

            )

        else:

            sensor_context = predict_sensor(

                sensor_reading,

                artifacts=_SENSOR_ARTIFACTS

            )

    # --------------------------------------------------
    # Decision Fusion
    # --------------------------------------------------

    fusion = fuse_predictions(

        image_result={

            "prediction":
                result["predicted_class"],

            "confidence":
                result["confidence"],

            "recommendation":
                recommendation

        },

        sensor_result=sensor_context,

        uav_result=uav_context

    )

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    gradcam_path = None

    try:

        gradcam_path = generate_gradcam(

            model=model,

            image_path=image_path,

            output_dir="results/gradcam"

        )

    except Exception:

        gradcam_path = None

    # --------------------------------------------------
    # Final Report
    # --------------------------------------------------

    report = {

        "crop": crop,

        "image_path": image_path,

        "prediction":
            result["predicted_class"],

        "confidence":
            result["confidence"],

        "class_probabilities":
            result["class_probabilities"],

        "recommendation":
            recommendation,

        "uav_context":
            uav_context,

        "sensor_context":
            sensor_context,

        "fusion":
            fusion,

        "gradcam":
            gradcam_path

    }

    return report
# ==========================================================
# Print Prediction Report
# ==========================================================

def print_report(report):

    print("\n" + "=" * 60)
    print("AgroFedVision Prediction Report")
    print("=" * 60)

    print(f"Crop       : {report['crop']}")
    print(f"Image      : {report['image_path']}")
    print(f"Prediction : {report['prediction']}")
    print(f"Confidence : {report['confidence']:.2%}")

    # --------------------------------------------------
    # Class Probabilities
    # --------------------------------------------------

    print("\nClass Probabilities")
    print("-" * 40)

    for cls, p in report["class_probabilities"].items():

        print(f"{cls:25s}: {p:.2%}")

    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------

    print("\nRecommendation")
    print("-" * 40)

    print(report["recommendation"]["summary"])

    for action in report["recommendation"]["actions"]:

        print(f"  • {action}")

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    if report.get("gradcam"):

        print("\nGrad-CAM")
        print("-" * 40)
        print(report["gradcam"])

    # --------------------------------------------------
    # UAV
    # --------------------------------------------------

    if report.get("uav_context"):

        u = report["uav_context"]

        print("\nUAV Analysis")
        print("-" * 40)

        print(f"NDVI Mean : {u['NDVI_Mean']:.3f}")
        print(f"NDVI Std  : {u['NDVI_Std']:.3f}")

        print(f"NDRE Mean : {u['NDRE_Mean']:.3f}")
        print(f"NDRE Std  : {u['NDRE_Std']:.3f}")

        print(f"Field Status : {u['field_health_flag']}")

    # --------------------------------------------------
    # Sensor
    # --------------------------------------------------

    if report.get("sensor_context"):

        s = report["sensor_context"]
        a = s["analysis"]

        print("\nSensor Analysis")
        print("-" * 40)

        print(f"Predicted Class : {s['predicted_class']}")
        print(f"Confidence      : {s['confidence']:.2%}")

        print("\nSoil Status")

        print(f" Nitrogen    : {a['nitrogen_status']}")
        print(f" Phosphorus  : {a['phosphorus_status']}")
        print(f" Potassium   : {a['potassium_status']}")
        print(f" Soil pH     : {a['soil_ph']}")
        print(f" Moisture    : {a['moisture_status']}")
        print(f" Temperature : {a['temperature_status']}")
        print(f" Humidity    : {a['humidity_status']}")

        print(f"\nSensor Health Score : {a['sensor_health_score']}/100")

        print("\nRecommendations")

        for item in a["fertilizer"]:

            print(f"  • {item}")

        print(f"  • {a['irrigation']}")

    # --------------------------------------------------
    # Decision Fusion
    # --------------------------------------------------

    if report.get("fusion"):

        f = report["fusion"]

        print("\n" + "=" * 60)
        print("Decision Fusion Report")
        print("=" * 60)

        print(f"Prediction Mode : {f['prediction_mode']}")
        print(f"Overall Status  : {f['overall_status']}")
        print(f"Risk Level      : {f['risk_level']}")
        print(f"Health Score    : {f['overall_health_score']:.2f}/100")

        print("\nIndividual Scores")
        print("-" * 40)

        if f["image_score"] is not None:
            print(f"Image  : {f['image_score']:.2f}")

        if f["sensor_score"] is not None:
            print(f"Sensor : {f['sensor_score']:.2f}")

        if f["uav_score"] is not None:
            print(f"UAV    : {f['uav_score']:.2f}")

        print("\nFusion Weights")
        print("-" * 40)

        for name, value in f["weights"].items():

            print(f"{name.capitalize():10s}: {value*100:.1f}%")

        print("\nFinal Recommendations")
        print("-" * 40)

        for rec in f["recommendations"]:

            print(f"  • {rec}")

    print("\n" + "=" * 60)
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AgroFedVision Prediction"
    )

    # --------------------------------------------------
    # Required Inputs
    # --------------------------------------------------

    parser.add_argument(
        "--crop",
        required=True,
        help=f"Available Crops: {list_available_crops()}"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to leaf image"
    )

    # --------------------------------------------------
    # Optional UAV Inputs
    # --------------------------------------------------

    parser.add_argument(
        "--uav-folder",
        default=None,
        help="Folder containing multispectral UAV images"
    )

    parser.add_argument(
        "--uav-id",
        default=None,
        help="Example: IMG_0041"
    )

    # --------------------------------------------------
    # Optional Sensor Inputs
    # --------------------------------------------------

    parser.add_argument("--N", type=float)
    parser.add_argument("--P", type=float)
    parser.add_argument("--K", type=float)

    parser.add_argument("--temperature", type=float)
    parser.add_argument("--humidity", type=float)
    parser.add_argument("--ph", type=float)
    parser.add_argument("--rainfall", type=float)

    parser.add_argument(
        "--soil-moisture",
        dest="soil_moisture",
        type=float
    )

    parser.add_argument(
        "--soil-type",
        dest="soil_type",
        type=float
    )

    parser.add_argument(
        "--sunlight-exposure",
        dest="sunlight_exposure",
        type=float
    )

    parser.add_argument(
        "--wind-speed",
        dest="wind_speed",
        type=float
    )

    parser.add_argument(
        "--co2-concentration",
        dest="co2_concentration",
        type=float
    )

    parser.add_argument(
        "--organic-matter",
        dest="organic_matter",
        type=float
    )

    parser.add_argument(
        "--irrigation-frequency",
        dest="irrigation_frequency",
        type=float
    )

    parser.add_argument(
        "--crop-density",
        dest="crop_density",
        type=float
    )

    parser.add_argument(
        "--pest-pressure",
        dest="pest_pressure",
        type=float
    )

    parser.add_argument(
        "--fertilizer-usage",
        dest="fertilizer_usage",
        type=float
    )

    parser.add_argument(
        "--growth-stage",
        dest="growth_stage",
        type=float
    )

    parser.add_argument(
        "--urban-area-proximity",
        dest="urban_area_proximity",
        type=float
    )

    parser.add_argument(
        "--water-source-type",
        dest="water_source_type",
        type=float
    )

    parser.add_argument(
        "--frost-risk",
        dest="frost_risk",
        type=float
    )

    parser.add_argument(
        "--water-usage-efficiency",
        dest="water_usage_efficiency",
        type=float
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Build Sensor Dictionary
    # --------------------------------------------------

    sensor_reading = {

        "N": args.N,
        "P": args.P,
        "K": args.K,

        "temperature": args.temperature,
        "humidity": args.humidity,
        "ph": args.ph,
        "rainfall": args.rainfall,

        "soil_moisture": args.soil_moisture,
        "soil_type": args.soil_type,
        "sunlight_exposure": args.sunlight_exposure,
        "wind_speed": args.wind_speed,
        "co2_concentration": args.co2_concentration,
        "organic_matter": args.organic_matter,
        "irrigation_frequency": args.irrigation_frequency,
        "crop_density": args.crop_density,
        "pest_pressure": args.pest_pressure,
        "fertilizer_usage": args.fertilizer_usage,
        "growth_stage": args.growth_stage,
        "urban_area_proximity": args.urban_area_proximity,
        "water_source_type": args.water_source_type,
        "frost_risk": args.frost_risk,
        "water_usage_efficiency": args.water_usage_efficiency,
    }

    # Disable sensor branch if no values were supplied

    if all(v is None for v in sensor_reading.values()):
        sensor_reading = None

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    report = predict_and_recommend(

        crop=args.crop,

        image_path=args.image,

        uav_capture_folder=args.uav_folder,

        uav_image_id=args.uav_id,

        sensor_reading=sensor_reading

    )

    # --------------------------------------------------
    # Print Report
    # --------------------------------------------------

    print_report(report)