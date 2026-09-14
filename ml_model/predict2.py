"""
predict.py

AgroFedVision Prediction Orchestrator

Runs image prediction, optional UAV analysis,
optional sensor analysis and decision fusion.
"""

import os
import pandas as pd
import sys
import glob
import argparse
import joblib
import numpy as np
from tensorboard import summary
import tensorflow as tf
from crop_interpreter import interpret_prediction
from plant_statistics import PlantStatistics
from uav_statistics import UAVStatistics
from gradcam import generate_gradcam
from decision_fusion import fuse_predictions
from explainable_fusion import ExplainableFusion
from shap_explainer import AgroFedSHAP, save_shap_result

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
# Unified Fusion Model
# ==========================================================

_FUSION_MODEL = None
_FUSION_ENCODER = None
_FUSION_SENSOR_SCALER = None
_FUSION_UAV_SCALER = None
_FUSION_SENSOR_COLS = None


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
# Load Unified Fusion Model
# ==========================================================

def _load_fusion_artifacts():

    global _FUSION_MODEL
    global _FUSION_ENCODER
    global _FUSION_SENSOR_SCALER
    global _FUSION_UAV_SCALER
    global _FUSION_SENSOR_COLS

    if _FUSION_MODEL is None:

        print(
            "\nLoading unified AgroFedVision fusion model..."
        )

        _FUSION_MODEL = tf.keras.models.load_model(
            "results/agrofedvision_fusion_model.keras"
        )

        _FUSION_ENCODER = joblib.load(
            "results/fusion_label_encoder.pkl"
        )

        _FUSION_SENSOR_SCALER = joblib.load(
            "results/fusion_sensor_scaler.pkl"
        )

        _FUSION_UAV_SCALER = joblib.load(
            "results/fusion_uav_scaler.pkl"
        )

        _FUSION_SENSOR_COLS = joblib.load(
            "results/fusion_sensor_cols.pkl"
        )

        print(
            "Fusion model loaded successfully."
        )

        print(
            f"Sensor features: "
            f"{len(_FUSION_SENSOR_COLS)}"
        )

        print(
            "UAV features: 4"
        )

    return (
        _FUSION_MODEL,
        _FUSION_ENCODER,
        _FUSION_SENSOR_SCALER,
        _FUSION_UAV_SCALER,
        _FUSION_SENSOR_COLS
    )
# ==========================================================
# Build Fusion Inputs For SHAP
# ==========================================================

def _prepare_shap_inputs(
    sensor_reading,
    uav_context,
    image_path
):

    (
        fusion_model,
        encoder,
        sensor_scaler,
        uav_scaler,
        sensor_cols
    ) = _load_fusion_artifacts()

    # ------------------------------------------------------
    # Sensor
    # ------------------------------------------------------

    sensor_values = []

    for column in sensor_cols:

        value = sensor_reading.get(
            column,
            0.0
        )

        if value is None:
            value = 0.0

        sensor_values.append(
            float(value)
        )

    sensor_raw = np.asarray(
        sensor_values,
        dtype=np.float32
    ).reshape(1, -1)

    sensor_df = pd.DataFrame(
    sensor_raw,
    columns=sensor_cols
    )
    sensor_scaled = (
    sensor_scaler
    .transform(sensor_df)
    .astype(np.float32)
)

    # ------------------------------------------------------
    # UAV
    # ------------------------------------------------------

    uav_values = [

        float(
            uav_context.get(
                "NDVI_Mean",
                0.0
            ) or 0.0
        ),

        float(
            uav_context.get(
                "NDVI_Std",
                0.0
            ) or 0.0
        ),

        float(
            uav_context.get(
                "NDRE_Mean",
                0.0
            ) or 0.0
        ),

        float(
            uav_context.get(
                "NDRE_Std",
                0.0
            ) or 0.0
        )

    ]

    uav_raw = np.asarray(
        uav_values,
        dtype=np.float32
    ).reshape(1, -1)

    uav_cols = [
    "NDVI_Mean",
    "NDVI_Std",
    "NDRE_Mean",
    "NDRE_Std"
]
    uav_df = pd.DataFrame(
    uav_raw,
    columns=uav_cols
)
    uav_scaled = (
    uav_scaler
    .transform(uav_df)
    .astype(np.float32)
)

    # ------------------------------------------------------
    # Image
    #
    # IMPORTANT:
    # Match current predict_images() preprocessing.
    # No /255 here.
    # ------------------------------------------------------

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [224, 224]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image.numpy()

    image = image[
        np.newaxis,
        ...
    ]

    return (
        sensor_scaled,
        uav_scaled,
        image
    )
# ==========================================================
# Generate SHAP Explanation
# ==========================================================

def generate_shap_explanation(
    sensor_reading,
    uav_context,
    image_path
):

    if sensor_reading is None:
        return None

    if uav_context is None:
        return None

    try:

        (
            fusion_model,
            fusion_encoder,
            sensor_scaler,
            uav_scaler,
            sensor_cols
        ) = _load_fusion_artifacts()

        (
            sensor_scaled,
            uav_scaled,
            image
        ) = _prepare_shap_inputs(
            sensor_reading,
            uav_context,
            image_path
        )

        # --------------------------------------------------
        # Background
        #
        # First working version:
        # zero in scaled feature space.
        # --------------------------------------------------

        # --------------------------------------------------
        # SHAP background: representative samples from the
        # fusion training data.
        # --------------------------------------------------

        fusion_dataset_path = "data/fused_dataset.csv"

        if not os.path.exists(fusion_dataset_path):
            raise FileNotFoundError(
                f"Fusion dataset not found: {fusion_dataset_path}"
            )

        fusion_df = pd.read_csv(fusion_dataset_path)

        missing_sensor_cols = [
            col for col in sensor_cols
            if col not in fusion_df.columns
        ]

        if missing_sensor_cols:
            raise ValueError(
                "Missing sensor columns in fused_dataset.csv: "
                + ", ".join(missing_sensor_cols)
            )

        background_sensor_raw = (
            fusion_df[sensor_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .values
            .astype(np.float32)
        )

        background_sensor_df = pd.DataFrame(
            background_sensor_raw,
            columns=sensor_cols
        )

        background_sensor = sensor_scaler.transform(
            background_sensor_df
        ).astype(np.float32)

        uav_cols = [
            "NDVI_Mean", "NDVI_Std", "NDRE_Mean", "NDRE_Std"
        ]

        missing_uav_cols = [
            col for col in uav_cols
            if col not in fusion_df.columns
        ]

        if missing_uav_cols:
            raise ValueError(
                "Missing UAV columns in fused_dataset.csv: "
                + ", ".join(missing_uav_cols)
            )

        background_uav_raw = (
            fusion_df[uav_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .values
            .astype(np.float32)
        )

        background_uav_df = pd.DataFrame(
            background_uav_raw,
            columns=uav_cols
        )

        background_uav = uav_scaler.transform(
            background_uav_df
        ).astype(np.float32)

        max_background_samples = 50
        if len(background_sensor) > max_background_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                len(background_sensor),
                size=max_background_samples,
                replace=False
            )
            background_sensor = background_sensor[indices]
            background_uav = background_uav[indices]

        print(
            f"[SHAP] Background samples: {len(background_sensor)}"
        )
        # --------------------------------------------------
        # SHAP Engine
        # --------------------------------------------------

        shap_engine = AgroFedSHAP(

            model=fusion_model,

            sensor_cols=sensor_cols,

            class_names=(
                fusion_encoder.classes_
            ),

            background_sensor=(
                background_sensor
            ),

            background_uav=(
                background_uav
            ),

            background_image=image

        )

        shap_result = shap_engine.explain(

            sensor=sensor_scaled,

            uav=uav_scaled,

            image=image,

            top_k=10

        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        output_path = (
            "results/shap_explanation.json"
        )

        save_shap_result(
            shap_result,
            output_path
        )

        print(
            "\nSHAP explanation saved:"
        )

        print(
            f"  {output_path}"
        )

        return shap_result

    except Exception as e:

        print(
            f"\n[SHAP] Explanation failed: {e}"
        )

        return None

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

def predict_uav_capture(capture_folder=None, uav_paths=None):
    """
    Analyze UAV multispectral captures.

    Supports:
    1. A complete UAV capture folder
    2. Explicit UAV image paths
    """

    stats = UAVStatistics()

    # --------------------------------------------------
    # Explicit UAV image paths
    # --------------------------------------------------
    if uav_paths:
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp(prefix="agrofedvision_uav_")

        try:
            copied_paths = []

            for path in uav_paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"UAV image not found: {path}"
                    )

                destination = os.path.join(
                    temp_dir,
                    os.path.basename(path)
                )

                shutil.copy2(path, destination)
                copied_paths.append(destination)

            _, captures = batch_extract(
                temp_dir,
                output_csv="results/uav_features.csv"
            )

        finally:
            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

    # --------------------------------------------------
    # UAV folder
    # --------------------------------------------------
    elif capture_folder:
        _, captures = batch_extract(
            capture_folder,
            output_csv="results/uav_features.csv"
        )

    else:
        raise ValueError(
            "No UAV folder or UAV images were provided."
        )

    # --------------------------------------------------
    # UAV statistics
    # --------------------------------------------------
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
        uav_image_paths=None,
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

    if uav_capture_folder or uav_image_paths:
        try:
            summary = predict_uav_capture(
                capture_folder=uav_capture_folder,
                uav_paths=uav_image_paths
            )
            uav_context = {
            "NDVI_Mean": summary["ndvi_mean"],
            "NDVI_Std": summary["ndvi_std"],
            "NDRE_Mean": summary.get("ndre_mean"),
            "NDRE_Std": summary.get("ndre_std"),
            "field_health_flag":
                "Computed from UAV Multispectral Captures",
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
    # =====================================================
# Explainable AI
# =====================================================

    explainer = ExplainableFusion()
    ps = result["plant_statistics"]
    
    explainer.add_module(

        name="Image",

        score=fusion.get("image_score"),

        confidence=ps["reliability_score"],

        summary=(
            f"{ps['moderate']} Moderate Risk leaves, "
            f"{ps['high']} High Risk leaves detected."
    ),

        evidence={

    "healthy": ps["healthy"],

    "moderate": ps["moderate"],

    "high": ps["high"],

    "majority_class": ps["majority_class"],

    "severity_percentage": ps["severity_percentage"],

    "health_index": ps["health_index"],

    "mixed_disease": ps["mixed_disease"]
},     recommendation=(
            "Inspect symptomatic leaves "
            "and monitor disease progression."
    )
)
    if sensor_context:

        analysis = sensor_context.get("analysis", {})

        explainer.add_module(

            name="Sensor",

            score=fusion.get("sensor_score"),

            confidence=None,

            summary=(
                f"Sensor Health Score "
                f"{analysis.get('sensor_health_score','N/A')}"
            ),

            evidence=analysis,

            recommendation=(
                "Maintain balanced irrigation "
                "and nutrient levels."
            )
        )

    if uav_context:

        us = uav_context["uav_statistics"]

        explainer.add_module(

            name="UAV",

            score=fusion.get("uav_score"),

            confidence=None,

            summary=(
                f"{us['total_images']} captures analysed."
            ),

            evidence={

                "Average NDVI": us["ndvi_mean"],

                "Average NDRE": us["ndre_mean"],

                "Best Capture": us["best_image"],

                "NDVI Std": us["ndvi_std"]
            },

            recommendation=(
                "Inspect regions having lower vegetation vigor."
            )
        )
    # =====================================================
    # SHAP explainability
    # =====================================================

    shap_result = None

    # SHAP explains structured inputs of the unified fusion model.
    if sensor_context is not None and uav_context is not None:
        shap_result = generate_shap_explanation(
            sensor_reading=sensor_reading,
            uav_context=uav_context,
            image_path=result["best_image"]
        )

        if shap_result is not None:
            explainer.add_shap_explanation(shap_result)
    else:
        print(
            "[SHAP] Skipped: both sensor and UAV context are required "
            "for the current unified SHAP explanation."
        )

    # =====================================================
    # generate explainability report
    # =====================================================

    explainability = explainer.generate(

        overall_status=
            fusion["overall_status"],

        risk_level=
            fusion["risk_level"],

        health_score=
            fusion["overall_health_score"]

    )

    save_explainability(
        explainability
    )

    

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    gradcam_path = None

    try:
        # IMPORTANT: explain the final unified AgroFedVision_V2 model,
        # not the separate crop image classifier.
        (
            fusion_model,
            fusion_encoder,
            fusion_sensor_scaler,
            fusion_uav_scaler,
            fusion_sensor_cols
        ) = _load_fusion_artifacts()

        if sensor_reading is not None and uav_context is not None:
            grad_sensor, grad_uav, _ = _prepare_shap_inputs(
                sensor_reading=sensor_reading,
                uav_context=uav_context,
                image_path=result["best_image"]
            )
        else:
            grad_sensor = None
            grad_uav = None

        gradcam_path = generate_gradcam(
            model=fusion_model,
            image_path=result["best_image"],
            sensor=grad_sensor,
            uav=grad_uav,
            save_dir="results/gradcam"
        )

        print("\nGrad-CAM generated:")
        print(f"  {gradcam_path}")

    except Exception as e:
        # Never hide the actual explainability error.
        print(
            f"\n[Grad-CAM] Explanation failed: "
            f"{type(e).__name__}: {e}"
        )
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
        
        "explainability": explainability,

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
    """Print a structured prediction report to stdout."""

    if report.get("explainability"):

        exp = report["explainability"]

        print("\n")
        print("=" * 60)
        print("Explainable AI Report")
        print("=" * 60)

        print(f"Overall Status : {exp['overall_status']}")
        print(f"Risk Level     : {exp['risk_level']}")
        print(f"Health Score   : {exp['health_score']:.2f}/100")

        for module in exp["modules"]:

            print("\n----------------------------------------")
            print(module["module"])
            print("----------------------------------------")

        if module["score"] is not None:
            print(f"Score        : {module['score']:.2f}")

        if module["confidence"] is not None:
            print(f"Confidence   : {module['confidence']:.2%}")

        print(f"Summary      : {module['summary']}")

        print("\nEvidence")

        for k, v in module["evidence"].items():
            print(f"  {k:<20}: {v}")

        if module["recommendation"]:
            print("\nRecommendation")
            print(f"  • {module['recommendation']}")

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
    # SHAP Explanation
    # --------------------------------------------------

    exp = report.get("explainability", {})
    shap_data = exp.get("shap_explanation")

    if shap_data:

        print("\nSHAP Explanation")
        print("-" * 40)

        print(
            f"Explained Class : "
            f"{shap_data['explained_class']}"
        )

        print(
            f"Probability     : "
            f"{shap_data['predicted_probability']:.2%}"
        )

        contribution = shap_data["modality_contribution"]

        print(
            f"\nSensor Contribution : "
            f"{contribution['sensor']:.2f}%"
        )

        print(
            f"UAV Contribution    : "
            f"{contribution['uav']:.2f}%"
        )

        print("\nTop Influential Features")
        print("-" * 40)

        for item in shap_data["top_features"]:

            print(
                f"{item['feature']:25s} "
                f"{item['shap_value']:+.6f} "
                f"({item['direction']})"
            )

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

        us = u.get("uav_statistics")
        if us is not None:
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

    # --------------------------------------------------
    # Parse Arguments
    # --------------------------------------------------

    args = parser.parse_args()

    # --------------------------------------------------
    # Build UAV Image List
    # --------------------------------------------------

    uav_paths = []

    if args.uav:
        uav_paths.append(args.uav)

    if args.uav_images:
        uav_paths.extend(args.uav_images)

    if args.uav_folder:

        for ext in ("*.tif", "*.tiff"):

            uav_paths.extend(
                glob.glob(
                    os.path.join(
                        args.uav_folder,
                        ext
                    )
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

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

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

        uav_image_paths=uav_paths,

        uav_image_id=args.uav_id,

        sensor_reading=sensor_reading

    )

    # --------------------------------------------------
    # Print Report
    # --------------------------------------------------

    print_report(report)