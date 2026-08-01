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
# NDVI
# =====================================================

ndvi = (nir - red) / (nir + red + 1e-8)

# =====================================================
# NDRE
# =====================================================

ndre = (nir - rededge) / (nir + rededge + 1e-8)

# =====================================================
# VEGETATION MASK
# =====================================================

veg = ndvi > 0.20

# =====================================================
# CLASS MAP
# =====================================================

# 0 = background
# 1 = healthy
# 2 = nitrogen
# 3 = phosphorus
# 4 = potassium

class_map = np.zeros(
    ndvi.shape,
    dtype=np.uint8
)

# Healthy

class_map[
    veg &
    (ndvi > 0.45) &
    (ndre > 0.10)
] = 1

# Nitrogen Deficiency

class_map[
    veg &
    (ndvi < 0.35) &
    (ndre < 0.00)
] = 2

# Phosphorus Deficiency

class_map[
    veg &
    (ndvi >= 0.35) &
    (ndvi < 0.45) &
    (ndre < 0.05)
] = 3

# Potassium Deficiency

class_map[
    veg &
    (ndvi < 0.45) &
    (ndre >= 0.05)
] = 4

# =====================================================
# STATISTICS
# =====================================================

total = np.sum(veg)

healthy = np.sum(class_map == 1)
nitrogen = np.sum(class_map == 2)
phosphorus = np.sum(class_map == 3)
potassium = np.sum(class_map == 4)

print("\n================================")
print("NUTRIENT DEFICIENCY REPORT")
print("================================")

print(f"Healthy     : {(healthy/total)*100:.2f}%")
print(f"Nitrogen    : {(nitrogen/total)*100:.2f}%")
print(f"Phosphorus  : {(phosphorus/total)*100:.2f}%")
print(f"Potassium   : {(potassium/total)*100:.2f}%")

# =====================================================
# RGB MAP
# =====================================================

rgb = np.zeros(
    (class_map.shape[0],
     class_map.shape[1],
     3),
    dtype=np.uint8
)

# Healthy → Green
rgb[class_map == 1] = [0,255,0]

# Nitrogen → Red
rgb[class_map == 2] = [255,0,0]

# Phosphorus → Yellow
rgb[class_map == 3] = [255,255,0]

# Potassium → Blue
rgb[class_map == 4] = [0,0,255]

# =====================================================
# SAVE
# =====================================================

cv2.imwrite(
    "results/nutrient_deficiency_map.png",
    cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )
)

plt.figure(figsize=(12,8))
plt.imshow(rgb)
plt.title("Nutrient Deficiency Map")
plt.axis("off")

plt.savefig(
    "results/nutrient_deficiency_map_plot.png",
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print("results/nutrient_deficiency_map.png")
print("results/nutrient_deficiency_map_plot.png")