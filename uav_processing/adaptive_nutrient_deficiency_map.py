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

OUTPUT_IMAGE = "results/adaptive_nutrient_deficiency_map.png"
OUTPUT_REPORT = "results/adaptive_nutrient_report.txt"

# =====================================================
# LOAD BANDS
# =====================================================

print("Loading Bands...")

red = tiff.imread(RED).astype(np.float32)
rededge = tiff.imread(REDEDGE).astype(np.float32)
nir = tiff.imread(NIR).astype(np.float32)

# =====================================================
# COMPUTE INDICES
# =====================================================

ndvi = (nir - red) / (nir + red + 1e-8)
ndre = (nir - rededge) / (nir + rededge + 1e-8)

# =====================================================
# VEGETATION MASK
# =====================================================

veg = ndvi > 0.20

print("\nVegetation Pixels:", np.sum(veg))

# =====================================================
# ADAPTIVE THRESHOLDS
# =====================================================

healthy_ndvi = np.percentile(ndvi[veg], 75)
healthy_ndre = np.percentile(ndre[veg], 75)

low_ndvi = np.percentile(ndvi[veg], 25)
low_ndre = np.percentile(ndre[veg], 25)

print("\nAdaptive Thresholds")
print("-----------------------------")
print(f"Healthy NDVI > {healthy_ndvi:.4f}")
print(f"Healthy NDRE > {healthy_ndre:.4f}")
print(f"Low NDVI     < {low_ndvi:.4f}")
print(f"Low NDRE     < {low_ndre:.4f}")

# =====================================================
# CLASS MAP
# =====================================================

# 0 Background
# 1 Healthy
# 2 Nitrogen
# 3 Phosphorus
# 4 Potassium

class_map = np.zeros(ndvi.shape, dtype=np.uint8)

# Healthy

healthy_mask = (
    veg &
    (ndvi >= healthy_ndvi) &
    (ndre >= healthy_ndre)
)

class_map[healthy_mask] = 1

# Nitrogen Deficiency

nitrogen_mask = (
    veg &
    (ndvi <= low_ndvi) &
    (ndre <= low_ndre)
)

class_map[nitrogen_mask] = 2

# Phosphorus Deficiency

phosphorus_mask = (
    veg &
    (class_map == 0) &
    (ndre <= low_ndre)
)

class_map[phosphorus_mask] = 3

# Potassium Deficiency

potassium_mask = (
    veg &
    (class_map == 0) &
    (ndvi <= healthy_ndvi)
)

class_map[potassium_mask] = 4

# Remaining vegetation becomes healthy

class_map[(veg) & (class_map == 0)] = 1

# =====================================================
# STATISTICS
# =====================================================

total = np.sum(veg)

healthy = np.sum(class_map == 1)
nitrogen = np.sum(class_map == 2)
phosphorus = np.sum(class_map == 3)
potassium = np.sum(class_map == 4)

print("\n======================================")
print("ADAPTIVE NUTRIENT DEFICIENCY REPORT")
print("======================================")

print(f"Healthy      : {healthy/total*100:.2f}%")
print(f"Nitrogen     : {nitrogen/total*100:.2f}%")
print(f"Phosphorus   : {phosphorus/total*100:.2f}%")
print(f"Potassium    : {potassium/total*100:.2f}%")

# =====================================================
# RGB VISUALIZATION
# =====================================================

rgb = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)

# Healthy
rgb[class_map == 1] = [0,255,0]

# Nitrogen
rgb[class_map == 2] = [255,0,0]

# Phosphorus
rgb[class_map == 3] = [255,255,0]

# Potassium
rgb[class_map == 4] = [0,0,255]

# =====================================================
# SAVE IMAGE
# =====================================================

cv2.imwrite(
    OUTPUT_IMAGE,
    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
)
#classify accuracy that is the main thing to be done locate as far it 
plt.figure(figsize=(12,8))
plt.imshow(rgb)
plt.title("Adaptive Nutrient Deficiency Map")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    "results/adaptive_nutrient_deficiency_map_plot.png",
    dpi=300
)

plt.close()

# =====================================================
# SAVE REPORT
# =====================================================

with open(OUTPUT_REPORT, "w") as f:

    f.write("AGROFEDVISION\n")
    f.write("===============================\n")
    f.write("Adaptive Nutrient Report\n\n")

    f.write(f"Healthy      : {healthy/total*100:.2f}%\n")
    f.write(f"Nitrogen     : {nitrogen/total*100:.2f}%\n")
    f.write(f"Phosphorus   : {phosphorus/total*100:.2f}%\n")
    f.write(f"Potassium    : {potassium/total*100:.2f}%\n\n")

    f.write("Recommendations\n")
    f.write("-----------------\n")

    if nitrogen > 0:
        f.write("- Increase Nitrogen fertilizer (Urea)\n")

    if phosphorus > 0:
        f.write("- Increase Phosphorus fertilizer (DAP)\n")

    if potassium > 0:
        f.write("- Increase Potassium fertilizer (MOP)\n")

    f.write("\nAdaptive thresholds\n")
    f.write(f"Healthy NDVI : {healthy_ndvi:.4f}\n")
    f.write(f"Healthy NDRE : {healthy_ndre:.4f}\n")
    f.write(f"Low NDVI     : {low_ndvi:.4f}\n")
    f.write(f"Low NDRE     : {low_ndre:.4f}\n")

print("\nSaved Files")
print("--------------------------------")
print(OUTPUT_IMAGE)
print("results/adaptive_nutrient_deficiency_map_plot.png")
print(OUTPUT_REPORT)
print("--------------------------------")
print("Done.")