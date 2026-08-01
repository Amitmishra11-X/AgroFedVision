import os
import cv2
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_ID = "0030"

BASE = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"

RED = os.path.join(BASE, f"IMG_{IMAGE_ID}_3.tif")
REDEDGE = os.path.join(BASE, f"IMG_{IMAGE_ID}_4.tif")
NIR = os.path.join(BASE, f"IMG_{IMAGE_ID}_5.tif")

# =====================================================
# LOAD BANDS
# =====================================================

print("Loading Bands...")

red = tiff.imread(RED).astype(np.float32)
rededge = tiff.imread(REDEDGE).astype(np.float32)
nir = tiff.imread(NIR).astype(np.float32)

# =====================================================
# NDVI + NDRE
# =====================================================

ndvi = (nir - red) / (nir + red + 1e-8)
ndre = (nir - rededge) / (nir + rededge + 1e-8)

veg = ndvi > 0.20

# =====================================================
# DEFICIENCY CLASSIFICATION
# =====================================================

prescription_map = np.zeros(
    ndvi.shape,
    dtype=np.uint8
)

# 1 = Healthy
prescription_map[
    veg &
    (ndvi > 0.45) &
    (ndre > 0.10)
] = 1

# 2 = Nitrogen
prescription_map[
    veg &
    (ndvi < 0.35) &
    (ndre < 0.00)
] = 2

# 3 = Phosphorus
prescription_map[
    veg &
    (ndvi >= 0.35) &
    (ndvi < 0.45) &
    (ndre < 0.05)
] = 3

# 4 = Potassium
prescription_map[
    veg &
    (ndvi < 0.45) &
    (ndre >= 0.05)
] = 4

# =====================================================
# STATISTICS
# =====================================================

total = np.sum(veg)

healthy = np.sum(prescription_map == 1)
nitrogen = np.sum(prescription_map == 2)
phosphorus = np.sum(prescription_map == 3)
potassium = np.sum(prescription_map == 4)

print("\n===================================")
print("FERTILIZER PRESCRIPTION REPORT")
print("===================================")

print(f"Healthy Area     : {(healthy/total)*100:.2f}%")
print(f"Nitrogen Area    : {(nitrogen/total)*100:.2f}%")
print(f"Phosphorus Area  : {(phosphorus/total)*100:.2f}%")
print(f"Potassium Area   : {(potassium/total)*100:.2f}%")

# =====================================================
# RECOMMENDATIONS
# =====================================================

recommendations = []

if nitrogen > 0:
    recommendations.append(
        "Apply Urea (Nitrogen fertilizer)"
    )

if phosphorus > 0:
    recommendations.append(
        "Apply DAP (Phosphorus fertilizer)"
    )

if potassium > 0:
    recommendations.append(
        "Apply MOP (Potassium fertilizer)"
    )

# =====================================================
# RGB MAP
# =====================================================

rgb = np.zeros(
    (
        prescription_map.shape[0],
        prescription_map.shape[1],
        3
    ),
    dtype=np.uint8
)

# Healthy = Green
rgb[prescription_map == 1] = [0,255,0]

# Nitrogen = Red
rgb[prescription_map == 2] = [255,0,0]

# Phosphorus = Yellow
rgb[prescription_map == 3] = [255,255,0]

# Potassium = Blue
rgb[prescription_map == 4] = [0,0,255]

# =====================================================
# SAVE MAP
# =====================================================

cv2.imwrite(
    "results/prescription_map.png",
    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
)

# =====================================================
# SAVE REPORT
# =====================================================

with open(
    "results/fertilizer_recommendation_report.txt",
    "w"
) as f:

    f.write("AGROFEDVISION\n")
    f.write("FERTILIZER PRESCRIPTION REPORT\n\n")

    f.write(
        f"Healthy Area: {(healthy/total)*100:.2f}%\n"
    )

    f.write(
        f"Nitrogen Deficiency: {(nitrogen/total)*100:.2f}%\n"
    )

    f.write(
        f"Phosphorus Deficiency: {(phosphorus/total)*100:.2f}%\n"
    )

    f.write(
        f"Potassium Deficiency: {(potassium/total)*100:.2f}%\n\n"
    )

    f.write("Recommendations:\n")

    for r in recommendations:
        f.write(f"- {r}\n")

# =====================================================
# VISUALIZATION
# =====================================================

plt.figure(figsize=(12,8))
plt.imshow(rgb)
plt.title("Prescription Map")
plt.axis("off")

plt.savefig(
    "results/prescription_map_plot.png",
    bbox_inches="tight"
)

plt.close()

print("\nRecommendations")

for r in recommendations:
    print("-", r)

print("\nSaved:")
print("results/prescription_map.png")
print("results/prescription_map_plot.png")
print("results/fertilizer_recommendation_report.txt")