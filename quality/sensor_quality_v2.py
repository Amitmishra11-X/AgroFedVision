"""
sensor_quality_v2.py

Research Version

Outputs

1. Dataset Quality
2. Feature Quality
3. Sensor Reliability
4. CSV Report
5. Visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = "data/Crop_recommendationV2.csv"

RESULTS = "results"

os.makedirs(RESULTS, exist_ok=True)

print("Loading Sensor Dataset...")

df = pd.read_csv(DATA)

numeric = df.select_dtypes(include=np.number)

# --------------------------------------------------------
# Agronomic Valid Ranges
# --------------------------------------------------------

VALID_RANGES = {

    "N": (0,200),

    "P": (0,200),

    "K": (0,250),

    "temperature": (-5,60),

    "humidity": (0,100),

    "ph": (3,10),

    "rainfall": (0,600),

    "soil_moisture": (0,100),

    "sunlight_exposure": (0,24),

    "wind_speed": (0,100),

    "co2_concentration": (200,2000),

    "organic_matter": (0,20),

    "irrigation_frequency": (0,10),

    "crop_density": (0,100),

    "pest_pressure": (0,100),

    "fertilizer_usage": (0,500),

    "growth_stage": (0,6),

    "urban_area_proximity": (0,100),

    "water_source_type": (0,5),

    "frost_risk": (0,100),

    "water_usage_efficiency": (0,5)
}

feature_scores = []

print("\nEvaluating Features...\n")

for column in numeric.columns:

    data = numeric[column]

    missing = data.isnull().sum()

    missing_ratio = missing / len(data)

    q1 = data.quantile(.25)

    q3 = data.quantile(.75)

    iqr = q3-q1

    lower = q1-1.5*iqr

    upper = q3+1.5*iqr

    outliers = ((data<lower)|(data>upper)).sum()

    outlier_ratio = outliers/len(data)

    invalid = 0

    if column in VALID_RANGES:

        low,high = VALID_RANGES[column]

        invalid=((data<low)|(data>high)).sum()

    invalid_ratio=invalid/len(data)

    cv=data.std()/(abs(data.mean())+1e-6)

    noise=min(cv,1)

    score=1-(
        0.30*missing_ratio+
        0.25*outlier_ratio+
        0.25*invalid_ratio+
        0.20*noise
    )

    score=max(score,0)

    feature_scores.append({

        "Feature":column,

        "Quality":round(score,3),

        "Missing":missing,

        "Outliers":outliers,

        "Invalid":invalid,

        "Noise":round(noise,3)

    })

report=pd.DataFrame(feature_scores)

overall=report["Quality"].mean()

print("="*45)

print("OVERALL SENSOR QUALITY")

print("="*45)

print(f"\nOverall Quality : {overall:.3f}")

if overall>0.90:

    status="Excellent"

elif overall>0.75:

    status="Good"

elif overall>0.60:

    status="Moderate"

else:

    status="Poor"

print("Status :",status)

print("\nTop Reliable Sensors")

print(report.sort_values("Quality",ascending=False).head())

print("\nWeakest Sensors")

print(report.sort_values("Quality").head())

report.to_csv(

    os.path.join(

        RESULTS,

        "sensor_feature_quality.csv"

    ),

    index=False

)

plt.figure(figsize=(12,5))

plt.bar(

    report["Feature"],

    report["Quality"]

)

plt.xticks(rotation=90)

plt.ylim(0,1)

plt.ylabel("Quality Score")

plt.title("Sensor Feature Quality")

plt.tight_layout()

plt.savefig(

    os.path.join(

        RESULTS,

        "sensor_feature_quality.png"

    ),

    dpi=300

)

plt.close()

print("\nSaved")

print("results/sensor_feature_quality.csv")

print("results/sensor_feature_quality.png")