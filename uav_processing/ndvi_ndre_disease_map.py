import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# =====================================================
# UAV DATA PATH
# =====================================================

BASE = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"

BLUE = os.path.join(BASE, "IMG_0045_1.tif")
GREEN = os.path.join(BASE, "IMG_0045_2.tif")
RED = os.path.join(BASE, "IMG_0045_3.tif")
REDEDGE = os.path.join(BASE, "IMG_0045_4.tif")
NIR = os.path.join(BASE, "IMG_0045_5.tif")

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD BANDS
# =====================================================

print("Loading Bands...")

blue = tiff.imread(BLUE).astype(np.float32)
green = tiff.imread(GREEN).astype(np.float32)
red = tiff.imread(RED).astype(np.float32)
rededge = tiff.imread(REDEDGE).astype(np.float32)
nir = tiff.imread(NIR).astype(np.float32)

print("Band Shape :", red.shape)
print("Band Type  :", red.dtype)

# =====================================================
# NDVI
# =====================================================

print("\nComputing NDVI...")

ndvi = (nir - red) / (nir + red + 1e-8)

print("\nNDVI Statistics")
print("Min  :", np.min(ndvi))
print("Max  :", np.max(ndvi))
print("Mean :", np.mean(ndvi))
print("Std  :", np.std(ndvi))

# =====================================================
# NDRE
# =====================================================

print("\nComputing NDRE...")

ndre = (nir - rededge) / (nir + rededge + 1e-8)

print("\nNDRE Statistics")
print("Min  :", np.min(ndre))
print("Max  :", np.max(ndre))
print("Mean :", np.mean(ndre))
print("Std  :", np.std(ndre))

# =====================================================
# SAVE NDVI MAP
# =====================================================

plt.figure(figsize=(10, 8))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar(label="NDVI")
plt.title("NDVI Map")
plt.axis("off")

plt.savefig(
    "results/ndvi_map_uav.png",
    bbox_inches="tight"
)

plt.close()

# =====================================================
# SAVE NDRE MAP
# =====================================================

plt.figure(figsize=(10, 8))
plt.imshow(ndre, cmap="RdYlGn")
plt.colorbar(label="NDRE")
plt.title("NDRE Map")
plt.axis("off")

plt.savefig(
    "results/ndre_map_uav.png",
    bbox_inches="tight"
)

plt.close()

# =====================================================
# VEGETATION MASK
# =====================================================

vegetation_mask = ndvi > 0.15

# =====================================================
# DISEASE RISK CLASSIFICATION
# =====================================================

risk_map = np.full(
    ndvi.shape,
    255,
    dtype=np.uint8
)

# Healthy
risk_map[
    vegetation_mask &
    (ndvi >= 0.55)
] = 0

# Moderate
risk_map[
    vegetation_mask &
    (ndvi >= 0.35) &
    (ndvi < 0.55)
] = 1

# High Risk
risk_map[
    vegetation_mask &
    (ndvi < 0.35)
] = 2

# =====================================================
# RGB VISUALIZATION
# =====================================================

rgb = np.zeros(
    (
        risk_map.shape[0],
        risk_map.shape[1],
        3
    ),
    dtype=np.uint8
)

# Background
rgb[risk_map == 255] = [0, 0, 0]

# Healthy
rgb[risk_map == 0] = [0, 255, 0]

# Moderate
rgb[risk_map == 1] = [255, 255, 0]

# High Risk
rgb[risk_map == 2] = [255, 0, 0]

cv2.imwrite(
    "results/disease_risk_map.png",
    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
)

# =====================================================
# FIELD SUMMARY
# =====================================================

valid_pixels = np.sum(
    vegetation_mask
)

healthy = np.sum(
    risk_map == 0
)

moderate = np.sum(
    risk_map == 1
)

high = np.sum(
    risk_map == 2
)

print("\n===================================")
print("FIELD HEALTH SUMMARY")
print("===================================")

print(
    f"Vegetation Pixels : {valid_pixels:,}"
)

print(
    f"Healthy       : {(healthy/valid_pixels)*100:.2f}%"
)

print(
    f"Moderate Risk : {(moderate/valid_pixels)*100:.2f}%"
)

print(
    f"High Risk     : {(high/valid_pixels)*100:.2f}%"
)

print("\nSaved Files")

print("results/ndvi_map_uav.png")
print("results/ndre_map_uav.png")
print("results/disease_risk_map.png")

print("\nDone.")