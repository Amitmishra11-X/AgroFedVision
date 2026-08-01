import os
import sys
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = "data/multimodal_dataset.csv"

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# --------------------------------------------------
# Feature Selection
# --------------------------------------------------

ignore_cols = [
    "image_path",
    "leaf_label",
    "crop_health"
]

features = [
    c for c in df.columns
    if c not in ignore_cols
]

X = df[features]

# --------------------------------------------------
# Scale Features
# --------------------------------------------------

scaler = joblib.load(
    "results/fusion_sensor_scaler.pkl"
)

sensor_cols = joblib.load(
    "results/fusion_sensor_cols.pkl"
)

uav_cols = [
    "NDVI_Mean",
    "NDVI_Std",
    "NDRE_Mean",
    "NDRE_Std"
]

sensor_scaled = scaler.transform(
    X[sensor_cols]
)

sensor_df = pd.DataFrame(
    sensor_scaled,
    columns=sensor_cols
)

for col in uav_cols:
    sensor_df[col] = X[col].values

X_final = sensor_df

print("Features Used:", len(X_final.columns))

# --------------------------------------------------
# Load Label Encoder
# --------------------------------------------------

label_encoder = joblib.load(
    "results/fusion_label_encoder.pkl"
)

# --------------------------------------------------
# Train Explainability Model
# --------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

y = label_encoder.transform(
    df["crop_health"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

print("RF Accuracy:",
      rf.score(X_test, y_test))

# --------------------------------------------------
# SHAP
# --------------------------------------------------

print("Calculating SHAP...")

explainer = shap.TreeExplainer(rf)

sample = X_test.iloc[:100]

shap_values = explainer.shap_values(sample)

# --------------------------------------------------
# Summary Plot
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.tight_layout()

os.makedirs(
    "results",
    exist_ok=True
)

plt.savefig(
    "results/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# Bar Plot
# --------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print("results/shap_summary.png")
print("results/shap_bar.png")

# --------------------------------------------------
# Top Features
# --------------------------------------------------

importance = np.abs(
    shap_values[0]
).mean(axis=0)

top_idx = np.argsort(
    importance
)[::-1][:10]

print("\nTop 10 Important Features:\n")

for i in top_idx:
    print(
        f"{X_final.columns[i]} : "
        f"{importance[i]:.4f}"
    )