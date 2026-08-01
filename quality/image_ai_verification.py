"""
AI Image Verification
AgroFedVision

Version 1
Uses pretrained EfficientNetB0
No training required.
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input,
    decode_predictions
)

from tensorflow.keras.preprocessing import image

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

IMAGE_FOLDER = r"data/leaf_images"

OUTPUT_CSV = "results/image_ai_verification.csv"

IMG_SIZE = 224

# ------------------------------------------------

print("\nLoading EfficientNetB0...")

model = EfficientNetB0(
    weights="imagenet"
)

# ------------------------------------------------


def image_metrics(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    brightness = gray.mean()

    contrast = gray.std()

    return blur, brightness, contrast


# ------------------------------------------------

rows = []

extensions = (".jpg", ".jpeg", ".png", ".bmp")

files = []

for root, _, fs in os.walk(IMAGE_FOLDER):
    for f in fs:
        if f.lower().endswith(extensions):
            files.append(os.path.join(root, f))

print(f"\nImages Found : {len(files)}")

# ------------------------------------------------

for path in files:

    try:

        img_cv = cv2.imread(path)

        blur, bright, contrast = image_metrics(img_cv)

        img = image.load_img(
            path,
            target_size=(IMG_SIZE, IMG_SIZE)
        )

        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        preds = model.predict(
            x,
            verbose=0
        )

        top = decode_predictions(preds, top=1)[0][0]

        imagenet_label = top[1]

        confidence = float(top[2])

        # ------------------------------
        # Reliability Score
        # ------------------------------

        score = confidence

        if blur < 80:
            score *= 0.8

        if bright < 40 or bright > 220:
            score *= 0.9

        if contrast < 20:
            score *= 0.9

        score = min(score, 1)

        if score > 0.90:
            status = "Excellent"

        elif score > 0.75:
            status = "Good"

        elif score > 0.55:
            status = "Moderate"

        else:
            status = "Poor"

        rows.append({

            "Image": os.path.basename(path),

            "ImageNet_Label": imagenet_label,

            "Confidence": round(confidence,3),

            "Blur": round(blur,2),

            "Brightness": round(bright,2),

            "Contrast": round(contrast,2),

            "Reliability": round(score,3),

            "Status": status

        })

    except:

        continue

# ------------------------------------------------

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n===================================")
print("AI IMAGE VERIFICATION REPORT")
print("===================================\n")

print(df.head(10))

print("\nSaved")

print(OUTPUT_CSV)