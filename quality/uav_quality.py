"""
uav_quality.py
--------------------------------------
AgroFedVision

Evaluates UAV multispectral image quality
and generates a reliability score.

Author: AgroFedVision
"""

import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ============================================================
# CHANGE THESE PATHS
# ============================================================

RED_PATH = r"data/uav/red.tif"
GREEN_PATH = r"data/uav/green.tif"
BLUE_PATH = r"data/uav/blue.tif"
RED_EDGE_PATH = r"data/uav/rededge.tif"
NIR_PATH = r"data/uav/nir.tif"

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# ============================================================


def load_band(path):

    with rasterio.open(path) as src:
        img = src.read(1).astype(np.float32)

    return img


def normalize(img):

    img = img - img.min()

    if img.max() != 0:
        img = img / img.max()

    return img


def laplacian_score(gray):

    return cv2.Laplacian(gray, cv2.CV_64F).var()


print("\nLoading UAV Bands...\n")

bands = {}

paths = {
    "Blue": BLUE_PATH,
    "Green": GREEN_PATH,
    "Red": RED_PATH,
    "RedEdge": RED_EDGE_PATH,
    "NIR": NIR_PATH,
}

# ---------------------------------------------------

missing = 0

for name, path in paths.items():

    if Path(path).exists():

        bands[name] = load_band(path)

    else:

        missing += 1

        print(f"{name} band missing.")

# ---------------------------------------------------

if len(bands) < 5:

    print("\nNot enough bands.")
    quit()

# ---------------------------------------------------

red = normalize(bands["Red"])
nir = normalize(bands["NIR"])
rededge = normalize(bands["RedEdge"])

# ============================================================
# NDVI
# ============================================================

ndvi = (nir - red) / (nir + red + 1e-6)

# ============================================================
# NDRE
# ============================================================

ndre = (nir - rededge) / (nir + rededge + 1e-6)

vegetation = ndvi > 0.2

veg_percent = vegetation.mean() * 100

# ============================================================
# RGB Preview
# ============================================================

rgb = np.dstack([
    normalize(bands["Red"]),
    normalize(bands["Green"]),
    normalize(bands["Blue"])
])

rgb8 = (rgb * 255).astype(np.uint8)

gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)

brightness = gray.mean()

contrast = gray.std()

sharpness = laplacian_score(gray)

dynamic_range = gray.max() - gray.min()

# ============================================================
# Quality Scores
# ============================================================

brightness_score = min(brightness / 128, 1)

contrast_score = min(contrast / 60, 1)

sharpness_score = min(sharpness / 500, 1)

dynamic_score = min(dynamic_range / 255, 1)

vegetation_score = min(veg_percent / 60, 1)

band_score = 1 - (missing / 5)

overall = np.mean([
    brightness_score,
    contrast_score,
    sharpness_score,
    dynamic_score,
    vegetation_score,
    band_score
])

# ============================================================

if overall > 0.90:
    status = "Excellent"

elif overall > 0.80:
    status = "Good"

elif overall > 0.65:
    status = "Moderate"

else:
    status = "Poor"

# ============================================================

print("=" * 55)
print(" UAV QUALITY REPORT")
print("=" * 55)

print()

print(f"Band Completeness : {band_score:.2f}")

print(f"Brightness        : {brightness:.2f}")

print(f"Contrast          : {contrast:.2f}")

print(f"Sharpness         : {sharpness:.2f}")

print(f"Dynamic Range     : {dynamic_range}")

print(f"Vegetation Cover  : {veg_percent:.2f}%")

print()

print(f"NDVI Mean         : {ndvi.mean():.3f}")

print(f"NDVI Std          : {ndvi.std():.3f}")

print(f"NDRE Mean         : {ndre.mean():.3f}")

print(f"NDRE Std          : {ndre.std():.3f}")

print()

print(f"Overall Score     : {overall:.3f}")

print(f"Status            : {status}")

# ============================================================
# Save CSV
# ============================================================

report = pd.DataFrame({

    "Metric":[
        "Band Completeness",
        "Brightness",
        "Contrast",
        "Sharpness",
        "Dynamic Range",
        "Vegetation Coverage",
        "NDVI Mean",
        "NDVI Std",
        "NDRE Mean",
        "NDRE Std",
        "Overall Score"
    ],

    "Value":[
        band_score,
        brightness,
        contrast,
        sharpness,
        dynamic_range,
        veg_percent,
        ndvi.mean(),
        ndvi.std(),
        ndre.mean(),
        ndre.std(),
        overall
    ]
})

report.to_csv(
    RESULTS / "uav_quality_report.csv",
    index=False
)

# ============================================================
# Radar Plot
# ============================================================

labels = [
    "Bands",
    "Brightness",
    "Contrast",
    "Sharpness",
    "Vegetation",
    "Dynamic"
]

values = [
    band_score,
    brightness_score,
    contrast_score,
    sharpness_score,
    vegetation_score,
    dynamic_score
]

angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)

values = np.concatenate((values,[values[0]]))
angles = np.concatenate((angles,[angles[0]]))

plt.figure(figsize=(6,6))

ax = plt.subplot(111,polar=True)

ax.plot(angles,values,linewidth=2)

ax.fill(angles,values,alpha=0.25)

ax.set_xticks(angles[:-1])

ax.set_xticklabels(labels)

plt.title("UAV Quality Assessment")

plt.savefig(
    RESULTS / "uav_quality_radar.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("\nSaved")

print("results/uav_quality_report.csv")

print("results/uav_quality_radar.png")