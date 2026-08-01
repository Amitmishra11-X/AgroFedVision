"""
===========================================================
AgroFedVision

Sensor Quality Assessment v3

Author : Amit Mishra

Purpose
-------
Evaluates agricultural sensor reliability before multimodal fusion.

Outputs

• Overall Dataset Quality
• Feature Reliability
• Missing Values
• Invalid Values
• Outliers
• Noise
• Reliability Ranking
• CSV Report
• TXT Report
• JSON Report
• Visualization

===========================================================
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

DATA_PATH = "data/Crop_recommendationV2.csv"

RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

print("\nLoading Sensor Dataset...")

df = pd.read_csv(DATA_PATH)

numeric_df = df.select_dtypes(include=np.number)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

# ----------------------------------------------------
# AGRONOMIC VALID RANGES
# ----------------------------------------------------

VALID_RANGES = {

    "N": (0,200),

    "P": (0,200),

    "K": (0,250),

    "temperature": (-10,60),

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

# ----------------------------------------------------
# RELIABILITY BAR
# ----------------------------------------------------

def reliability_bar(score, length=10):

    score = max(0, min(score, 1))

    filled = int(round(score * length))

    return "█"*filled + "░"*(length-filled)

# ----------------------------------------------------
# QUALITY LABEL
# ----------------------------------------------------

def quality_label(score):

    if score >= 0.95:
        return "Excellent"

    elif score >= 0.85:
        return "Good"

    elif score >= 0.70:
        return "Moderate"

    else:
        return "Poor"

# ----------------------------------------------------
# FEATURE ANALYSIS
# ----------------------------------------------------

feature_results = []

print("\nEvaluating Sensor Features...\n")

for column in numeric_df.columns:

    values = numeric_df[column]

    total = len(values)

    # -----------------------
    # Missing
    # -----------------------

    missing = values.isnull().sum()

    missing_ratio = missing / total

    # -----------------------
    # Invalid
    # -----------------------

    invalid = 0

    if column in VALID_RANGES:

        low, high = VALID_RANGES[column]

        invalid = ((values < low) | (values > high)).sum()

    invalid_ratio = invalid / total

    # -----------------------
    # Outliers
    # -----------------------

    q1 = values.quantile(.25)

    q3 = values.quantile(.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr

    upper = q3 + 1.5 * iqr

    outliers = ((values < lower) | (values > upper)).sum()

    outlier_ratio = outliers / total

    # -----------------------
    # Noise
    # -----------------------

    mean = abs(values.mean()) + 1e-6

    cv = values.std() / mean

    noise = min(cv,1)

    # -----------------------
    # Reliability Score
    # -----------------------

    score = (
        1
        - 0.30 * missing_ratio
        - 0.25 * invalid_ratio
        - 0.25 * outlier_ratio
        - 0.20 * noise
    )

    score = max(score,0)

    feature_results.append({

        "Feature":column,

        "Quality":round(score,3),

        "Missing":missing,

        "Invalid":invalid,

        "Outliers":outliers,

        "Noise":round(noise,3),

        "Status":quality_label(score)

    })

# ----------------------------------------------------
# CREATE DATAFRAME
# ----------------------------------------------------

report = pd.DataFrame(feature_results)

report = report.sort_values(
    "Quality",
    ascending=False
).reset_index(drop=True)

overall_quality = report["Quality"].mean()

overall_status = quality_label(overall_quality)
# ----------------------------------------------------
# OVERALL DATASET QUALITY
# ----------------------------------------------------

print("\n" + "=" * 60)
print("SENSOR RELIABILITY REPORT")
print("=" * 60)

print(f"\nOverall Dataset Quality : {overall_quality:.3f}")
print(f"Overall Status          : {overall_status}")

print("\n" + "-" * 60)
print("FEATURE RELIABILITY")
print("-" * 60)

# ----------------------------------------------------
# PRINT FEATURE REPORT
# ----------------------------------------------------

for _, row in report.iterrows():

    score = float(row["Quality"])

    bar = reliability_bar(score)

    print(
        f"{row['Feature']:<24}"
        f"{bar} "
        f"{score*100:6.2f}%   "
        f"{row['Status']}"
    )

# ----------------------------------------------------
# TOP RELIABLE SENSORS
# ----------------------------------------------------

print("\n" + "-" * 60)
print("TOP 5 MOST RELIABLE SENSORS")
print("-" * 60)

top5 = report.head(5)

for i, (_, row) in enumerate(top5.iterrows(), start=1):

    print(
        f"{i}. "
        f"{row['Feature']} "
        f"({row['Quality']*100:.2f}%)"
    )

# ----------------------------------------------------
# WEAKEST SENSORS
# ----------------------------------------------------

print("\n" + "-" * 60)
print("SENSORS NEEDING ATTENTION")
print("-" * 60)

bottom5 = report.tail(5)

for i, (_, row) in enumerate(bottom5.iterrows(), start=1):

    print(
        f"{i}. "
        f"{row['Feature']} "
        f"({row['Quality']*100:.2f}%)"
    )

# ----------------------------------------------------
# SUMMARY STATISTICS
# ----------------------------------------------------

excellent = (report["Status"] == "Excellent").sum()

good = (report["Status"] == "Good").sum()

moderate = (report["Status"] == "Moderate").sum()

poor = (report["Status"] == "Poor").sum()

print("\n" + "-" * 60)
print("SUMMARY")
print("-" * 60)

print(f"Excellent Sensors : {excellent}")
print(f"Good Sensors      : {good}")
print(f"Moderate Sensors  : {moderate}")
print(f"Poor Sensors      : {poor}")

# ----------------------------------------------------
# SAVE CSV
# ----------------------------------------------------

csv_path = os.path.join(
    RESULT_DIR,
    "sensor_feature_quality.csv"
)

report.to_csv(
    csv_path,
    index=False
)

# ----------------------------------------------------
# SAVE JSON
# ----------------------------------------------------

json_report = {

    "overall_quality": round(float(overall_quality),3),

    "status": overall_status,

    "feature_quality":
        report.to_dict(orient="records")

}

json_path = os.path.join(
    RESULT_DIR,
    "sensor_quality.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_report,
        f,
        indent=4
    )

# ----------------------------------------------------
# SAVE TXT REPORT
# ----------------------------------------------------

txt_path = os.path.join(
    RESULT_DIR,
    "sensor_reliability_report.txt"
)

with open(
    txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("="*60 + "\n")

    f.write("SENSOR RELIABILITY REPORT\n")

    f.write("="*60 + "\n\n")

    f.write(
        f"Overall Dataset Quality : "
        f"{overall_quality:.3f}\n"
    )

    f.write(
        f"Overall Status : "
        f"{overall_status}\n\n"
    )

    f.write("-"*60 + "\n")

    f.write("FEATURE RELIABILITY\n")

    f.write("-"*60 + "\n\n")

    for _, row in report.iterrows():

        score = float(row["Quality"])

        f.write(
            f"{row['Feature']:<24}"
            f"{score*100:6.2f}%   "
            f"{row['Status']}\n"
        )

    f.write("\n")

    f.write("-"*60 + "\n")

    f.write("TOP RELIABLE SENSORS\n")

    f.write("-"*60 + "\n")

    for i, (_, row) in enumerate(
            top5.iterrows(),
            start=1):

        f.write(
            f"{i}. "
            f"{row['Feature']} "
            f"({row['Quality']*100:.2f}%)\n"
        )

    f.write("\n")

    f.write("-"*60 + "\n")

    f.write("NEEDS ATTENTION\n")

    f.write("-"*60 + "\n")

    for i, (_, row) in enumerate(
            bottom5.iterrows(),
            start=1):

        f.write(
            f"{i}. "
            f"{row['Feature']} "
            f"({row['Quality']*100:.2f}%)\n"
        )

print("\nSaved Files")
print("-"*60)

print(csv_path)
print(json_path)
print(txt_path)
# ----------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------

plt.style.use("ggplot")

# ----------------------------------------------------
# Figure 1: Feature Reliability
# ----------------------------------------------------

plt.figure(figsize=(12,8))

colors = []

for q in report["Quality"]:

    if q >= 0.95:
        colors.append("green")

    elif q >= 0.85:
        colors.append("limegreen")

    elif q >= 0.70:
        colors.append("orange")

    else:
        colors.append("red")

plt.barh(
    report["Feature"],
    report["Quality"],
    color=colors
)

plt.xlim(0,1)

plt.xlabel("Reliability Score")

plt.title("Sensor Reliability Ranking")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "sensor_feature_quality.png"
    ),
    dpi=300
)

plt.close()

# ----------------------------------------------------
# Figure 2: Quality Distribution
# ----------------------------------------------------

status_count = report["Status"].value_counts()

plt.figure(figsize=(7,7))

plt.pie(
    status_count.values,
    labels=status_count.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sensor Quality Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "sensor_quality_distribution.png"
    ),
    dpi=300
)

plt.close()

# ----------------------------------------------------
# Figure 3: Top 10 Sensors
# ----------------------------------------------------

top10 = report.head(10)

plt.figure(figsize=(12,6))

plt.bar(
    top10["Feature"],
    top10["Quality"],
)

plt.ylim(0,1)

plt.xticks(rotation=45)

plt.ylabel("Reliability")

plt.title("Top Reliable Sensors")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "top_sensor_reliability.png"
    ),
    dpi=300
)

plt.close()

# ----------------------------------------------------
# AI Fusion Weights
# ----------------------------------------------------

sensor_weights = {}

total_quality = report["Quality"].sum()

for _, row in report.iterrows():

    sensor_weights[row["Feature"]] = (
        float(row["Quality"]) /
        total_quality
    )

# Save JSON

with open(
    os.path.join(
        RESULT_DIR,
        "sensor_weights.json"
    ),
    "w"
) as f:

    json.dump(
        sensor_weights,
        f,
        indent=4
    )

# Save NumPy Array

weight_array = np.array(
    list(sensor_weights.values()),
    dtype=np.float32
)

np.save(
    os.path.join(
        RESULT_DIR,
        "sensor_weights.npy"
    ),
    weight_array
)

# ----------------------------------------------------
# AI Summary
# ----------------------------------------------------

print("\n" + "="*60)
print("AI FUSION READINESS")
print("="*60)

print("\nHighest Confidence Sensors")

for _, row in report.head(5).iterrows():

    print(
        f"{row['Feature']:<22}"
        f"{row['Quality']:.3f}"
    )

print("\nLowest Confidence Sensors")

for _, row in report.tail(5).iterrows():

    print(
        f"{row['Feature']:<22}"
        f"{row['Quality']:.3f}"
    )

print("\nFusion Weight Sum :", weight_array.sum())

print("\nSaved Additional Files")

print("------------------------------------------")

print("sensor_feature_quality.png")

print("sensor_quality_distribution.png")

print("top_sensor_reliability.png")

print("sensor_weights.json")

print("sensor_weights.npy")

print("\nSensor Quality Module Finished Successfully.")