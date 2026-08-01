"""
uav_quality_v2.py
-----------------------------------------
AgroFedVision UAV Reliability Engine

Uses:
    - adaptive_nutrient_report.txt
    - fertilizer_recommendation_report.txt
    - NDVI map
    - NDRE map
    - Disease hotspot map

Outputs:
    results/uav_quality.csv
    results/uav_quality.json
    results/uav_quality_report.txt
    results/uav_quality_dashboard.png
"""

from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd

from uav_parser import (
    parse_adaptive_report,
    parse_prescription_report
)

from uav_dashboard import (
    create_quality_dashboard,
    create_pie_chart,
    create_reliability_gauge,
    create_radar
)

# ----------------------------------------------------
# Paths
# ----------------------------------------------------

RESULTS = Path("results")

ADAPTIVE_REPORT = RESULTS / "adaptive_nutrient_report.txt"
PRESCRIPTION_REPORT = RESULTS / "fertilizer_recommendation_report.txt"

NDVI_IMAGE = RESULTS / "ndvi_map_uav.png"
NDRE_IMAGE = RESULTS / "ndre_map_uav.png"
DISEASE_IMAGE = RESULTS / "disease_hotspots.png"

# ----------------------------------------------------
# Helper
# ----------------------------------------------------

def image_quality(path):

    if not path.exists():

        return {
            "brightness":0,
            "contrast":0,
            "coverage":0
        }

    img = cv2.imread(str(path))

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(np.mean(gray))

    contrast = float(np.std(gray))

    coverage = float(
        np.count_nonzero(gray)/gray.size
    )

    return {

        "brightness":brightness,

        "contrast":contrast,

        "coverage":coverage

    }

# ----------------------------------------------------
# Reliability
# ----------------------------------------------------

def compute_reliability(
    adaptive,
    prescription,
    ndvi,
    ndre,
    disease
):

    # -----------------------------
    # Pipeline completeness
    # -----------------------------

    files = [
        ADAPTIVE_REPORT,
        PRESCRIPTION_REPORT,
        NDVI_IMAGE,
        NDRE_IMAGE,
        DISEASE_IMAGE
    ]

    pipeline = (
        sum(f.exists() for f in files)
        /
        len(files)
    )*100

    # -----------------------------
    # Vegetation
    # -----------------------------

    vegetation = adaptive["healthy"]

    # -----------------------------
    # Nutrient Balance
    # -----------------------------

    nutrient = 100 - (

        adaptive["nitrogen"]

        +

        adaptive["phosphorus"]

        +

        adaptive["potassium"]

    )/3

    nutrient = max(0,min(100,nutrient))

    # -----------------------------
    # Prescription
    # -----------------------------

    prescription_score = 100

    if len(prescription["recommendations"])==0:

        prescription_score = 40

    # -----------------------------
    # Image quality
    # -----------------------------

    image_score = np.mean([

        ndvi["coverage"]*100,

        ndre["coverage"]*100,

        disease["coverage"]*100

    ])

    image_score = min(image_score,100)

    # -----------------------------
    # Final
    # -----------------------------

    reliability = (

        0.20*pipeline +

        0.30*vegetation +

        0.20*nutrient +

        0.15*prescription_score +

        0.15*image_score

    )

    return {

        "Pipeline":round(pipeline,2),

        "Vegetation":round(vegetation,2),

        "Nutrient":round(nutrient,2),

        "Prescription":round(prescription_score,2),

        "Image":round(image_score,2),

        "Reliability":round(reliability,2)

    }

# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main():

    print()

    print("Loading UAV Reports...")

    adaptive = parse_adaptive_report(
        ADAPTIVE_REPORT
    )

    prescription = parse_prescription_report(
        PRESCRIPTION_REPORT
    )

    print("Reading Images...")

    ndvi = image_quality(
        NDVI_IMAGE
    )

    ndre = image_quality(
        NDRE_IMAGE
    )

    disease = image_quality(
        DISEASE_IMAGE
    )

    metrics = compute_reliability(

        adaptive,

        prescription,

        ndvi,

        ndre,

        disease

    )

    print()

    print("======================================")

    print("UAV QUALITY REPORT")

    print("======================================")

    for k,v in metrics.items():

        print(f"{k:15s}: {v:.2f}")

    # ----------------------------------

    pd.DataFrame([metrics]).to_csv(

        RESULTS/"uav_quality.csv",

        index=False

    )

    with open(

        RESULTS/"uav_quality.json",

        "w"

    ) as f:

        json.dump(

            metrics,

            f,

            indent=4

        )

    # ----------------------------------

    with open(

        RESULTS/"uav_quality_report.txt",

        "w"

    ) as f:

        f.write(

            "AGROFEDVISION UAV QUALITY REPORT\n"

        )

        f.write(

            "="*40+"\n\n"

        )

        for k,v in metrics.items():

            f.write(

                f"{k:15s}: {v:.2f}\n"

            )

    # ----------------------------------

    dashboard = {

        "Pipeline":metrics["Pipeline"],

        "Healthy":adaptive["healthy"],

        "Prescription":metrics["Prescription"],

        "Vegetation":metrics["Vegetation"],

        "Image":metrics["Image"]

    }

    create_quality_dashboard(

        dashboard

    )

    create_pie_chart({

        "Healthy":adaptive["healthy"],

        "Nitrogen":adaptive["nitrogen"],

        "Phosphorus":adaptive["phosphorus"],

        "Potassium":adaptive["potassium"]

    })

    create_reliability_gauge(

        metrics["Reliability"]

    )

    create_radar(

        dashboard

    )

    print()

    print("Saved")

    print("results/uav_quality.csv")

    print("results/uav_quality.json")

    print("results/uav_quality_report.txt")

    print("results/uav_quality_dashboard.png")

    print("results/uav_quality_radar.png")

    print("results/uav_reliability_gauge.png")

if __name__=="__main__":

    main()