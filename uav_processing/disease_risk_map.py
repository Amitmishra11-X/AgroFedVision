import os
import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

INPUT_TIF = "data/uav/sample.tif"

OUTPUT_MAP = "results/disease_risk_map.png"


def normalize_band(band):

    band = band.astype(np.float32)

    return (
        band - np.min(band)
    ) / (
        np.max(band) - np.min(band) + 1e-8
    )


def compute_ndvi(nir, red):

    return (
        nir - red
    ) / (
        nir + red + 1e-8
    )


def classify_risk(ndvi):

    risk_map = np.zeros(
        ndvi.shape,
        dtype=np.uint8
    )

    # High Risk
    risk_map[ndvi < 0.20] = 2

    # Moderate Risk
    risk_map[
        (ndvi >= 0.20) &
        (ndvi < 0.50)
    ] = 1

    # Healthy
    risk_map[ndvi >= 0.50] = 0

    return risk_map


def create_rgb_risk_map(risk_map):

    rgb = np.zeros(
        (
            risk_map.shape[0],
            risk_map.shape[1],
            3
        ),
        dtype=np.uint8
    )

    # Green
    rgb[risk_map == 0] = [0,255,0]

    # Yellow
    rgb[risk_map == 1] = [255,255,0]

    # Red
    rgb[risk_map == 2] = [255,0,0]

    return rgb


def main():

    print("Loading UAV Image...")

    img = tiff.imread(
        INPUT_TIF
    )

    print(
        "Shape:",
        img.shape
    )

    # ------------------------------------------------
    # Expected:
    #
    # Band 0 -> Blue
    # Band 1 -> Green
    # Band 2 -> Red
    # Band 3 -> NIR
    #
    # Adjust if needed
    # ------------------------------------------------

    if len(img.shape) == 3:

        blue  = normalize_band(img[:,:,0])
        green = normalize_band(img[:,:,1])
        red   = normalize_band(img[:,:,2])
        nir   = normalize_band(img[:,:,3])

    else:

        raise ValueError(
            "Expected multi-band UAV image"
        )

    print(
        "Computing NDVI..."
    )

    ndvi = compute_ndvi(
        nir,
        red
    )

    risk_map = classify_risk(
        ndvi
    )

    rgb_risk = create_rgb_risk_map(
        risk_map
    )

    plt.figure(
        figsize=(10,8)
    )

    plt.imshow(
        rgb_risk
    )

    plt.title(
        "Disease Risk Map"
    )

    plt.axis("off")

    plt.savefig(
        OUTPUT_MAP,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved:",
        OUTPUT_MAP
    )

    total_pixels = risk_map.size

    healthy = np.sum(
        risk_map == 0
    )

    moderate = np.sum(
        risk_map == 1
    )

    high = np.sum(
        risk_map == 2
    )

    print("\nField Statistics")

    print(
        f"Healthy      : {healthy/total_pixels*100:.2f}%"
    )

    print(
        f"Moderate Risk: {moderate/total_pixels*100:.2f}%"
    )

    print(
        f"High Risk    : {high/total_pixels*100:.2f}%"
    )


if __name__ == "__main__":
    main()