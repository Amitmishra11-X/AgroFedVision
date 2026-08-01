"""
AgroFedVision
Image Quality Assessment

Evaluates leaf image quality before multimodal fusion.

Outputs

• Blur
• Brightness
• Contrast
• Sharpness
• Noise
• Overall Quality Score
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

IMAGE_ROOT = r"D:\Download1\Multi_Crop_Leaves_Disease\Multi_Crop_Leaves_Disease"

RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

# ----------------------------------------------------
# FIND IMAGES
# ----------------------------------------------------

image_paths = []

for root, _, files in os.walk(IMAGE_ROOT):

    for file in files:

        if file.lower().endswith((".jpg",".jpeg",".png")):

            image_paths.append(os.path.join(root,file))

print(f"\nImages Found : {len(image_paths)}")

# ----------------------------------------------------
# QUALITY FUNCTIONS
# ----------------------------------------------------

def blur_score(gray):

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

def brightness_score(gray):

    return gray.mean()

def contrast_score(gray):

    return gray.std()

def sharpness_score(gray):

    gx = cv2.Sobel(gray,cv2.CV_64F,1,0)

    gy = cv2.Sobel(gray,cv2.CV_64F,0,1)

    return np.mean(np.sqrt(gx**2+gy**2))

def noise_score(gray):

    blur = cv2.GaussianBlur(gray,(5,5),0)

    noise = gray.astype(np.float32)-blur.astype(np.float32)

    return np.std(noise)

# ----------------------------------------------------
# ANALYZE
# ----------------------------------------------------

results=[]

print("\nEvaluating Images...\n")

for path in image_paths:

    image=cv2.imread(path)

    if image is None:
        continue

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    blur=blur_score(gray)

    bright=brightness_score(gray)

    contrast=contrast_score(gray)

    sharp=sharpness_score(gray)

    noise=noise_score(gray)

    # -----------------------
    # Normalize
    # -----------------------

    blur_n=min(blur/300,1)

    bright_n=1-abs(bright-128)/128

    contrast_n=min(contrast/80,1)

    sharp_n=min(sharp/60,1)

    noise_n=1-min(noise/40,1)

    quality=np.mean([
        blur_n,
        bright_n,
        contrast_n,
        sharp_n,
        noise_n
    ])

    if quality>=0.90:
        status="Excellent"

    elif quality>=0.80:
        status="Good"

    elif quality>=0.65:
        status="Moderate"

    else:
        status="Poor"

    results.append({

        "Image":os.path.basename(path),

        "Blur":round(blur,2),

        "Brightness":round(bright,2),

        "Contrast":round(contrast,2),

        "Sharpness":round(sharp,2),

        "Noise":round(noise,2),

        "Quality":round(quality,3),

        "Status":status

    })

report=pd.DataFrame(results)

report=report.sort_values(
    "Quality",
    ascending=False
)

# ----------------------------------------------------
# REPORT
# ----------------------------------------------------

overall=report["Quality"].mean()

print("="*55)
print("IMAGE QUALITY REPORT")
print("="*55)

print(f"\nImages Evaluated : {len(report)}")

print(f"Average Quality  : {overall:.3f}")

print("\nTop 5 Images")

print(report.head())

print("\nLowest Quality Images")

print(report.tail())

# ----------------------------------------------------
# SAVE
# ----------------------------------------------------

report.to_csv(
    os.path.join(
        RESULT_DIR,
        "image_quality.csv"
    ),
    index=False
)

plt.figure(figsize=(12,8))

plt.hist(
    report["Quality"],
    bins=20
)

plt.xlabel("Quality Score")

plt.ylabel("Images")

plt.title("Image Quality Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "image_quality_distribution.png"
    ),
    dpi=300
)

plt.close()

print("\nSaved")

print("results/image_quality.csv")

print("results/image_quality_distribution.png")