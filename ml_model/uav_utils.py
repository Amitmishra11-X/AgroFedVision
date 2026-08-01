"""
uav_utils.py

AgroFedVision -- UAV Feature Extraction (reusable version)

Wraps the exact NDVI/NDRE computation from the original UAV scripts
into a callable function, so it can be used both for batch CSV
generation (as before) and for single-capture lookups inside
predict.py.

Band convention (confirmed from the original scripts):
    band_3 = Red
    band_4 = NIR
    band_5 = Red Edge

    NDVI = (NIR - Red) / (NIR + Red)
    NDRE = (NIR - RedEdge) / (NIR + RedEdge)
"""

import os
import numpy as np
import tifffile as tiff


def extract_uav_features(capture_folder, image_id):
    """
    Given a folder containing <image_id>_1.tif ... <image_id>_5.tif
    (5-band capture), returns NDVI/NDRE statistics for that capture.

    Returns None if any band file is missing or unreadable, rather
    than raising -- callers (like predict.py) treat missing UAV data
    as "no UAV context available", not a hard failure.
    """
    try:
        band3 = tiff.imread(os.path.join(capture_folder, f"{image_id}_3.tif")).astype(float)  # Red
        band4 = tiff.imread(os.path.join(capture_folder, f"{image_id}_4.tif")).astype(float)  # NIR
        band5 = tiff.imread(os.path.join(capture_folder, f"{image_id}_5.tif")).astype(float)  # RedEdge
    except Exception as e:
        print(f"[uav_utils] Could not read bands for {image_id}: {e}")
        return None

    ndvi = (band4 - band3) / (band4 + band3 + 1e-10)
    ndre = (band4 - band5) / (band4 + band5 + 1e-10)

    ndvi_mean = float(np.mean(ndvi))
    ndre_mean = float(np.mean(ndre))

    return {
        "image_id": image_id,
        "NDVI_Mean": ndvi_mean,
        "NDVI_Std": float(np.std(ndvi)),
        "NDRE_Mean": ndre_mean,
        "NDRE_Std": float(np.std(ndre)),
        "field_health_flag": _interpret_ndvi(ndvi_mean),
    }


def _interpret_ndvi(ndvi_mean):
    """
    Rough, commonly-used NDVI interpretation bands for vegetation
    vigor. These thresholds are standard remote-sensing rules of
    thumb, not something this project derived empirically -- treat
    as a coarse qualitative flag, not a validated health score.
    """
    if ndvi_mean < 0.2:
        return "Low vegetation vigor (bare soil / sparse cover / stress)"
    elif ndvi_mean < 0.5:
        return "Moderate vegetation vigor"
    else:
        return "Healthy, dense vegetation"


def batch_extract(capture_folder, output_csv="uav_features.csv"):
    """
    Batch version -- same behavior as the original standalone script,
    but reusing extract_uav_features() so there's only one place the
    NDVI/NDRE math lives.
    """
    import pandas as pd

    files = sorted(f for f in os.listdir(capture_folder) if f.endswith("_1.tif"))
    image_ids = [f.replace("_1.tif", "") for f in files]

    results = []
    for image_id in image_ids:
        result = extract_uav_features(capture_folder, image_id)
        if result is not None:
            results.append(result)
            print("Processed:", image_id)
        else:
            print("Skipped:", image_id)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    return df, results


if __name__ == "__main__":
    # Same default path as the original script -- update if needed
    base = (r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial"
            r"\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000")
    batch_extract(base)