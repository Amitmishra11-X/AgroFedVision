"""
predict.py

AgroFedVision Prediction Orchestrator

Runs image prediction, optional UAV analysis,
optional sensor analysis and decision fusion.
"""

import os
import sys
import glob
import argparse
import joblib
import numpy as np
import tensorflow as tf
from crop_interpreter import interpret_prediction
from plant_statistics import PlantStatistics
from uav_statistics import UAVStatistics
from gradcam import generate_gradcam
from decision_fusion import fuse_predictions
from explainable_fusion import ExplainableFusion

from explainability_utils import save_explainability

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Required so TensorFlow can deserialize the custom preprocessing layer
import image_encoder

from crop_registry import (
    get_crop_config,
    list_available_crops
)

from recommendations import get_recommendation
from uav_utils import (
    extract_uav_features,
    batch_extract
)
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
# Image Prediction (Single + Multiple Images)
# ==========================================================

def predict_images(crop, image_paths):

    # -----------------------------------------
    # Convert single image to list
    # -----------------------------------------

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    config = get_crop_config(crop)

    model, encoder = _load_model_and_encoder(crop)

    img_size = config["img_size"]

    # -----------------------------------------
    # Plant Statistics Object
    # -----------------------------------------

    stats = PlantStatistics()

    # -----------------------------------------
    # Process Every Image
    # -----------------------------------------

    for image_path in image_paths:

        img = tf.io.read_file(image_path)

        img = tf.image.decode_image(
            img,
            channels=3,
            expand_animations=False
        )

        img = tf.image.resize(
            img,
            [img_size, img_size]
        )

        img = tf.cast(img, tf.float32)

        img = tf.expand_dims(
            img,
            axis=0
        )

        probs = model.predict(
            img,
            verbose=0
        )[0]

        confidence = float(
            np.max(probs)
        )

        predicted = encoder.inverse_transform(

            [np.argmax(probs)]

        )[0]

        # -----------------------------------------
        # Store inside PlantStatistics
        # -----------------------------------------

        stats.add_prediction(

            image_path=image_path,

            predicted_class=predicted,

            confidence=confidence,

            probabilities=probs

        )

    # -----------------------------------------
    # Average Probability
    # -----------------------------------------

    avg_probs = stats.average_probabilities()

    pred_idx = int(

        np.argmax(avg_probs)

    )

    predicted_class = encoder.inverse_transform(

        [pred_idx]

    )[0]

    confidence = float(

        avg_probs[pred_idx]

    )

    # -----------------------------------------
    # Top-3 Predictions
    # -----------------------------------------

    sorted_idx = np.argsort(

        avg_probs

    )[::-1]

    top_predictions = []

    for idx in sorted_idx[:3]:

        top_predictions.append({

            "class":

                encoder.inverse_transform([idx])[0],

            "confidence":

                float(avg_probs[idx])

        })

    # -----------------------------------------
    # Plant Statistics
    # -----------------------------------------

    plant_stats = stats.summary()

    # -----------------------------------------
    # Return
    # -----------------------------------------

    return {

        "crop": crop,

        "predicted_class": predicted_class,

        "confidence": confidence,

        "top_predictions": top_predictions,

        "class_probabilities": {

            cls: float(prob)

            for cls, prob in zip(

                encoder.classes_,

                avg_probs

            )

        },

        "best_image":

            plant_stats["best_image"],

        "num_images":

            plant_stats["total_images"],

        "per_image_results":

            plant_stats["per_image_results"],

        "plant_statistics":

            plant_stats

    }
# ==========================================================
# UAV Prediction (Multiple Captures)
# ==========================================================

def predict_uav_capture(capture_folder):

    stats = UAVStatistics()

    _, captures = batch_extract(
        capture_folder,
        output_csv="results/uav_features.csv"
    )

    for capture in captures:

        stats.add_image(

            image_name=capture["image_id"],

            ndvi_mean=capture["NDVI_Mean"],

            ndre_mean=capture["NDRE_Mean"]

        )

    return stats.summary()

# ==========================================================
# Complete Prediction Pipeline
# ==========================================================

def predict_and_recommend(

        crop,
        image_paths,
        uav_capture_folder=None,
        uav_image_id=None,
        sensor_reading=None
):

    global _SENSOR_ARTIFACTS

    # --------------------------------------------------
    # Image Prediction
    # --------------------------------------------------

    result = predict_images(
        crop,
        image_paths
    )
    
    interpreted = interpret_prediction(

    crop,

    result["predicted_class"]

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

    if uav_capture_folder:
        try:
            summary = predict_uav_capture(uav_capture_folder)
            uav_context = {
                "NDVI_Mean": summary["ndvi_mean"],
                "NDVI_Std": summary["ndvi_std"],
                "NDRE_Mean": summary.get("ndre_mean"),
                "NDRE_Std": summary.get("ndre_std"),
                "field_health_flag": "Computed from Multiple UAV Captures",
                "uav_statistics": summary,
            }
        except Exception as e:
            print(f"[Predict] UAV Error: {e}")


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
                recommendation,
                
            "plant_statistics":
                result["plant_statistics"]


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

            image_path=result["best_image"],

            output_dir="results/gradcam"

)

    except Exception:

        gradcam_path = None

    # --------------------------------------------------
    # Final Report
    # --------------------------------------------------

    report = {

        "crop": crop,

        "image_path": result["best_image"],

        "num_images": result["num_images"],

        "per_image_results": result["per_image_results"],
        
        
        "plant_statistics": result["plant_statistics"],

        "prediction":
            result["predicted_class"],
        "interpreted_prediction":
            interpreted,

        "confidence":
            result["confidence"],
        "top_predictions":
            result["top_predictions"],

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

    # --------------------------------------------------
    # Basic Information
    # --------------------------------------------------

    print(f"Crop       : {report['crop']}")
    print(f"Image      : {report['image_path']}")
    print(f"Images     : {report['num_images']}")
    print(f"Prediction : {report['prediction']}")
    ip = report["interpreted_prediction"]

    print(f"Disease    : {ip['disease']}")
    print(f"Risk       : {ip['risk']}")
    print(f"Severity   : {ip['severity']}")
    print(f"Health     : {ip['health_state']}")
    ps = report["plant_statistics"]
    print(f"Plant Health Index : {ps['health_index']:.1f}/100")
    print(f"Reliability Score  : {ps['reliability_score']:.2%}")
    print(f"Mixed Disease      : {ps['mixed_disease']}")
    print(f"Confidence : {report['confidence']:.2%}")
    # --------------------------------------------------
# Per Image Results
# --------------------------------------------------

    if report.get("per_image_results"):

        print("\nPer Image Results")
        print("-" * 40)

    for item in report["per_image_results"]:

        print(
            f"{os.path.basename(item['image'])} "
            f"-> {item['prediction']} "
            f"({item['confidence']:.2%})"
        )
    print("\nPlant Statistics")
    print("-"*40)

    print(f"Healthy Leaves        : {ps['healthy']}")

    print(f"Moderate Risk Leaves  : {ps['moderate']}")

    print(f"High Risk Leaves      : {ps['high']}")

    print(f"Healthy %             : {ps['healthy_percent']:.1f}%")

    print(f"Moderate %            : {ps['moderate_percent']:.1f}%")

    print(f"High %                : {ps['high_percent']:.1f}%")

    print(f"Majority Class        : {ps['majority_class']}")

    print(f"Severity Score        : {ps['severity_percentage']:.1f}%")

    # --------------------------------------------------
    # Top Predictions
    # --------------------------------------------------

    if report.get("top_predictions"):

        print("\nTop Predictions")
        print("-" * 40)

        for item in report["top_predictions"]:

            print(
                f"{item['class']:25s}: "
                f"{item['confidence']:.2%}"
            )

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
    # UAV Analysis
    # --------------------------------------------------

    if report.get("uav_context"):

        u = report["uav_context"]

        print("\nUAV Analysis")
        print("-" * 40)

        print(f"NDVI Mean     : {u['NDVI_Mean']:.3f}")
        print(f"NDVI Std      : {u['NDVI_Std']:.3f}")
        print(f"NDRE Mean     : {u['NDRE_Mean']:.3f}")
        print(f"NDRE Std      : {u['NDRE_Std']:.3f}")
        print(f"Field Status  : {u['field_health_flag']}")
    if "uav_statistics" in u:

        us = u["uav_statistics"]

        print("\nField Statistics")
        print("-" * 40)

        print(f"Captures Processed : {us['total_images']}")

        print(f"Best Capture       : {us['best_image']}")

        print(f"Average NDVI       : {us['ndvi_mean']:.3f}")

        print(f"NDVI Std           : {us['ndvi_std']:.3f}")

        print(f"Minimum NDVI       : {us['ndvi_min']:.3f}")

        print(f"Maximum NDVI       : {us['ndvi_max']:.3f}")

    if "ndre_mean" in us:
        print(f"Average NDRE       : {us['ndre_mean']:.3f}")

    # --------------------------------------------------
    # Sensor Analysis
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

        if f.get("image_score") is not None:
            print(f"Image  : {f['image_score']:.2f}")

        if f.get("sensor_score") is not None:
            print(f"Sensor : {f['sensor_score']:.2f}")

        if f.get("uav_score") is not None:
            print(f"UAV    : {f['uav_score']:.2f}")

        print("\nFusion Weights")
        print("-" * 40)

        for name, value in f["weights"].items():

            print(f"{name.capitalize():10s}: {value * 100:.1f}%")

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

# --------------------------------------------------
# Image Inputs
# --------------------------------------------------

    parser.add_argument(
        "--image",
        help="Single leaf image"
)

    parser.add_argument(
        "--images",
        nargs="+",
        help="Multiple leaf images"
)

    parser.add_argument(
        "--image-folder",
        dest="image_folder",
        help="Folder containing leaf images"
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
    "--uav",
    type=str,
    help="Single UAV image"
    )
    parser.add_argument(
    "--uav-images",
    nargs="+",
    help="Multiple UAV images"
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
    uav_paths = []

if args.uav:
    uav_paths.append(args.uav)

if args.uav_images:
    uav_paths.extend(args.uav_images)

if args.uav_folder:
    for ext in ("*.tif", "*.tiff"):
        uav_paths.extend(
            glob.glob(
                os.path.join(args.uav_folder, ext)
            )
        )
# --------------------------------------------------
# Build Image List
# --------------------------------------------------

image_paths = []

# Single image
if args.image:
    image_paths.append(args.image)

# Multiple images
if args.images:
    image_paths.extend(args.images)

# Folder
if args.image_folder:

    supported = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff"
    )

    for ext in supported:
        image_paths.extend(
            glob.glob(
                os.path.join(
                    args.image_folder,
                    ext
                )
            )
        )

image_paths = sorted(image_paths)

# Validation
if len(image_paths) == 0:

    parser.error(
        "Provide at least one of:\n"
        "--image\n"
        "--images\n"
        "--image-folder"
    )

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

if all(v is None for v in sensor_reading.values()):
    sensor_reading = None

# --------------------------------------------------
# Prediction
# --------------------------------------------------

report = predict_and_recommend(

    crop=args.crop,

    image_paths=image_paths,

    uav_capture_folder=args.uav_folder,

    uav_image_id=args.uav_id,

    sensor_reading=sensor_reading

)

# --------------------------------------------------
# Print Report
# --------------------------------------------------

print_report(report)