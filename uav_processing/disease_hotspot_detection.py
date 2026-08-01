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

OUTPUT_FILE = "results/disease_hotspots.png"

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
# STRESS INDEX
# =====================================================

stress_index = (
    (1.0 - ndvi) * 0.7 +
    (1.0 - ndre) * 0.3
)

print("\nStress Index Statistics")

print("Min  :", np.min(stress_index))
print("Max  :", np.max(stress_index))
print("Mean :", np.mean(stress_index))
print("Std  :", np.std(stress_index))

# =====================================================
# VEGETATION MASK
# =====================================================

vegetation_mask = ndvi > 0.20

vegetation_pixels = np.sum(vegetation_mask)

print("\nVegetation Pixels:", vegetation_pixels)

# =====================================================
# PERCENTILE ANALYSIS
# =====================================================

print("\nPercentile Analysis")

for p in [70, 75, 80, 85, 90, 95]:

    value = np.percentile(
        stress_index[vegetation_mask],
        p
    )

    print(f"P{p}: {value:.6f}")

# =====================================================
# ADAPTIVE THRESHOLD
# =====================================================

threshold = np.percentile(
    stress_index[vegetation_mask],
    85
)

print(
    f"\nAdaptive Threshold: {threshold:.6f}"
)

# =====================================================
# HOTSPOT MASK
# =====================================================

hotspot_mask = (
    vegetation_mask &
    (stress_index > threshold)
)

hotspot_mask = hotspot_mask.astype(np.uint8)

print(
    "Hotspot Pixels:",
    np.sum(hotspot_mask)
)

# =====================================================
# MORPHOLOGICAL CLEANUP
# =====================================================

kernel = np.ones(
    (5, 5),
    np.uint8
)

hotspot_mask = cv2.morphologyEx(
    hotspot_mask,
    cv2.MORPH_OPEN,
    kernel
)

hotspot_mask = cv2.morphologyEx(
    hotspot_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# =====================================================
# CONNECTED COMPONENTS
# =====================================================

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    hotspot_mask,
    connectivity=8
)

# =====================================================
# BACKGROUND IMAGE
# =====================================================

background = cv2.normalize(
    ndvi,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

background = cv2.cvtColor(
    background,
    cv2.COLOR_GRAY2RGB
)

# =====================================================
# HOTSPOT ANALYSIS
# =====================================================

hotspot_count = 0

print("\nDetected Hotspots")

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area < 100:
        continue

    hotspot_count += 1

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    cx, cy = centroids[i]

    cv2.rectangle(
        background,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    cv2.putText(
        background,
        f"H{hotspot_count}",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    print(
        f"\nHotspot {hotspot_count}"
    )

    print(
        f"Area     : {area} pixels"
    )

    print(
        f"Location : ({int(cx)}, {int(cy)})"
    )

# =====================================================
# SAVE VISUALIZATION
# =====================================================

plt.figure(figsize=(12, 8))

plt.imshow(background)

plt.title(
    f"Disease Hotspots Detected: {hotspot_count}"
)

plt.axis("off")

plt.savefig(
    OUTPUT_FILE,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# FINAL REPORT
# =====================================================

print("\n=================================")
print("Disease Hotspot Detection Report")
print("=================================")

print(
    f"Total Hotspots: {hotspot_count}"
)

print(
    f"Saved: {OUTPUT_FILE}"
)

print("=================================")