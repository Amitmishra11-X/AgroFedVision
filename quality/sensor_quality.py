"""
sensor_quality.py

AgroFedVision
Sensor Data Quality Assessment

Evaluates

1. Missing values
2. Duplicate rows
3. Invalid values
4. Outliers
5. Noise
6. Completeness
7. Overall quality score

Run

python quality/sensor_quality.py
"""

import os
import numpy as np
import pandas as pd

# ===========================================================
# CONFIG
# ===========================================================

CSV_PATH = "data/Crop_recommendationV2.csv"

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===========================================================
# LOAD DATA
# ===========================================================

print("\nLoading Sensor Dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

# ===========================================================
# REMOVE LABEL COLUMN
# ===========================================================

numeric_df = df.select_dtypes(include=np.number)

# ===========================================================
# 1. MISSING VALUES
# ===========================================================

missing_values = numeric_df.isnull().sum().sum()

missing_percent = (
    missing_values /
    (numeric_df.shape[0] * numeric_df.shape[1])
)

# ===========================================================
# 2. DUPLICATES
# ===========================================================

duplicates = df.duplicated().sum()

duplicate_percent = duplicates / len(df)

# ===========================================================
# 3. INVALID VALUES
# ===========================================================

invalid = 0

for col in numeric_df.columns:

    invalid += (numeric_df[col] < 0).sum()

invalid_percent = invalid / (
    numeric_df.shape[0] *
    numeric_df.shape[1]
)

# ===========================================================
# 4. OUTLIERS (IQR)
# ===========================================================

outlier_count = 0

total_values = 0

for col in numeric_df.columns:

    q1 = numeric_df[col].quantile(0.25)
    q3 = numeric_df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_count += (
        ((numeric_df[col] < lower) |
         (numeric_df[col] > upper))
    ).sum()

    total_values += len(numeric_df)

outlier_percent = outlier_count / total_values

# ===========================================================
# 5. NOISE SCORE
# ===========================================================

noise_score = 0

for col in numeric_df.columns:

    cv = numeric_df[col].std() / (
        numeric_df[col].mean() + 1e-6
    )

    noise_score += cv

noise_score /= len(numeric_df.columns)

# ===========================================================
# QUALITY SCORES
# ===========================================================

completeness = max(0, 1 - missing_percent)

consistency = max(0, 1 - duplicate_percent)

validity = max(
    0,
    1 - invalid_percent - outlier_percent
)

noise_quality = max(
    0,
    1 - min(noise_score, 1)
)

overall = np.mean([
    completeness,
    consistency,
    validity,
    noise_quality
])

# ===========================================================
# QUALITY LABEL
# ===========================================================

if overall >= 0.90:
    status = "Excellent"

elif overall >= 0.75:
    status = "Good"

elif overall >= 0.60:
    status = "Moderate"

else:
    status = "Poor"

# ===========================================================
# PRINT REPORT
# ===========================================================

print("\n===================================")
print("SENSOR QUALITY REPORT")
print("===================================")

print(f"\nOverall Score : {overall:.3f}")

print(f"Status        : {status}")

print("\nChecks")
print("-----------------------------------")

print(f"Missing Values      : {missing_values}")
print(f"Duplicate Rows      : {duplicates}")
print(f"Invalid Values      : {invalid}")
print(f"Outlier Percentage  : {outlier_percent*100:.2f}%")
print(f"Noise Score         : {noise_score:.3f}")

print("\nQuality Breakdown")
print("-----------------------------------")

print(f"Completeness : {completeness:.3f}")
print(f"Consistency  : {consistency:.3f}")
print(f"Validity     : {validity:.3f}")
print(f"Noise        : {noise_quality:.3f}")

# ===========================================================
# SAVE REPORT
# ===========================================================

report_path = os.path.join(
    RESULT_DIR,
    "sensor_quality_report.txt"
)

with open(report_path, "w") as f:

    f.write("SENSOR QUALITY REPORT\n")
    f.write("=============================\n\n")

    f.write(f"Overall Score : {overall:.3f}\n")
    f.write(f"Status : {status}\n\n")

    f.write(f"Missing Values : {missing_values}\n")
    f.write(f"Duplicate Rows : {duplicates}\n")
    f.write(f"Invalid Values : {invalid}\n")
    f.write(f"Outlier Percentage : {outlier_percent*100:.2f}%\n")
    f.write(f"Noise Score : {noise_score:.3f}\n\n")

    f.write("Quality Breakdown\n")

    f.write(f"Completeness : {completeness:.3f}\n")
    f.write(f"Consistency : {consistency:.3f}\n")
    f.write(f"Validity : {validity:.3f}\n")
    f.write(f"Noise : {noise_quality:.3f}\n")

print(f"\nSaved:\n{report_path}")