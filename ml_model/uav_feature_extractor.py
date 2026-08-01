"""
uav_feature_extractor.py

AgroFedVision V2

Extracts per-image UAV features from the
TiHAN Agriculture Multispectral Dataset.

Output:
    data/uav_expanded_features.csv
"""

import os
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff

from glob import glob
from tqdm import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT_FOLDER = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial"

OUTPUT_CSV = "data/uav_expanded_features.csv"

IMAGE_EXT = "*.tif"

# ==========================================================
# FIND ALL TIFF FILES
# ==========================================================

print("\nSearching UAV Dataset...")

all_files = glob(

    os.path.join(

        ROOT_FOLDER,

        "**",

        IMAGE_EXT

    ),

    recursive=True

)

print(f"Total TIFF files : {len(all_files)}")

# ==========================================================
# GROUP INTO CAPTURES
#
# Example:
#
# IMG_0118_1.tif
# IMG_0118_2.tif
# IMG_0118_3.tif
# IMG_0118_4.tif
# IMG_0118_5.tif
#
# becomes
#
# IMG_0118
#
# ==========================================================

captures = {}

for file in all_files:

    filename = os.path.basename(file)

    name = os.path.splitext(filename)[0]

    tokens = name.split("_")

    if len(tokens) < 3:

        continue

    capture_name = "_".join(tokens[:-1])

    band = tokens[-1]

    if capture_name not in captures:

        captures[capture_name] = {}

    captures[capture_name][band] = file

print(f"Total UAV Captures : {len(captures)}")

# ==========================================================
# KEEP ONLY COMPLETE 5-BAND CAPTURES
# ==========================================================

complete = {}

for key in captures:

    if len(captures[key]) == 5:

        complete[key] = captures[key]

captures = complete

print(f"Complete Captures : {len(captures)}")

# ==========================================================
# BAND READER
# ==========================================================

def read_band(path):

    img = tiff.imread(path)

    img = img.astype(np.float32)

    return img

# ==========================================================
# SAFE DIVISION
# ==========================================================

def safe_divide(a, b):

    return np.divide(

        a,

        b + 1e-8

    )

# ==========================================================
# FEATURE LIST
# ==========================================================

records = []

print("\nDataset Ready.")

print(f"Processing {len(captures)} UAV samples...")
# ==========================================================
# PROCESS EACH UAV CAPTURE
# ==========================================================

for capture_name in tqdm(

    sorted(captures.keys())

):

    try:

        # --------------------------------------------------
        # READ FIVE BANDS
        # --------------------------------------------------

        band1 = read_band(captures[capture_name]["1"])

        band2 = read_band(captures[capture_name]["2"])

        band3 = read_band(captures[capture_name]["3"])

        band4 = read_band(captures[capture_name]["4"])

        band5 = read_band(captures[capture_name]["5"])

        # --------------------------------------------------
        # BAND ASSIGNMENT
        #
        # Verify with dataset documentation if needed.
        # --------------------------------------------------

        BLUE = band1

        GREEN = band2

        RED = band3

        RED_EDGE = band4

        NIR = band5

        # --------------------------------------------------
        # VEGETATION INDICES
        # --------------------------------------------------

        NDVI = safe_divide(

            NIR - RED,

            NIR + RED

        )

        NDRE = safe_divide(

            NIR - RED_EDGE,

            NIR + RED_EDGE

        )

        GNDVI = safe_divide(

            NIR - GREEN,

            NIR + GREEN

        )

        SAVI = 1.5 * safe_divide(

            NIR - RED,

            NIR + RED + 0.5

        )

        MSAVI = (

            2 * NIR + 1 -

            np.sqrt(

                (2 * NIR + 1) ** 2 -

                8 * (NIR - RED)

            )

        ) / 2

        OSAVI = 1.16 * safe_divide(

            NIR - RED,

            NIR + RED + 0.16

        )

        CIGREEN = safe_divide(

            NIR,

            GREEN

        ) - 1

        CIREDGE = safe_divide(

            NIR,

            RED_EDGE

        ) - 1

        # --------------------------------------------------
        # REMOVE INVALID VALUES
        # --------------------------------------------------

        indices = {

            "NDVI": NDVI,

            "NDRE": NDRE,

            "GNDVI": GNDVI,

            "SAVI": SAVI,

            "MSAVI": MSAVI,

            "OSAVI": OSAVI,

            "CIGREEN": CIGREEN,

            "CIREDGE": CIREDGE

        }

        for name in indices:

            indices[name] = np.nan_to_num(

                indices[name],

                nan=0.0,

                posinf=0.0,

                neginf=0.0

            )

        NDVI = indices["NDVI"]

        NDRE = indices["NDRE"]

        GNDVI = indices["GNDVI"]

        SAVI = indices["SAVI"]

        MSAVI = indices["MSAVI"]

        OSAVI = indices["OSAVI"]

        CIGREEN = indices["CIGREEN"]

        CIREDGE = indices["CIREDGE"]
                # ==========================================================
        # BASIC STATISTICS
        # ==========================================================

        feature = {}

        feature["Capture"] = capture_name

        def stats(prefix, img):

            feature[f"{prefix}_Mean"] = float(np.mean(img))
            feature[f"{prefix}_Std"] = float(np.std(img))
            feature[f"{prefix}_Min"] = float(np.min(img))
            feature[f"{prefix}_Max"] = float(np.max(img))
            feature[f"{prefix}_Median"] = float(np.median(img))

            feature[f"{prefix}_P25"] = float(np.percentile(img, 25))
            feature[f"{prefix}_P75"] = float(np.percentile(img, 75))

        # ----------------------------------------------------------
        # Statistics for all vegetation indices
        # ----------------------------------------------------------

        stats("NDVI", NDVI)
        stats("NDRE", NDRE)
        stats("GNDVI", GNDVI)
        stats("SAVI", SAVI)
        stats("MSAVI", MSAVI)
        stats("OSAVI", OSAVI)
        stats("CIGREEN", CIGREEN)
        stats("CIREDGE", CIREDGE)

        # ==========================================================
        # VEGETATION COVERAGE
        # ==========================================================

        vegetation_mask = NDVI > 0.30

        feature["Vegetation_Coverage"] = float(
            np.mean(vegetation_mask)
        )

        healthy_mask = NDVI > 0.60

        feature["Healthy_Percentage"] = float(
            np.mean(healthy_mask)
        )

        stressed_mask = NDVI < 0.20

        feature["Stress_Percentage"] = float(
            np.mean(stressed_mask)
        )

        # ==========================================================
        # BRIGHTNESS
        # ==========================================================

        feature["Brightness"] = float(

            np.mean(

                (RED + GREEN + BLUE) / 3

            )

        )

        # ==========================================================
        # CONTRAST
        # ==========================================================

        feature["Contrast"] = float(

            np.std(

                (RED + GREEN + BLUE) / 3

            )

        )

        # ==========================================================
        # BLUR SCORE
        # ==========================================================

        rgb = np.stack(

            [RED, GREEN, BLUE],

            axis=-1

        )

        rgb = cv2.normalize(

            rgb,

            None,

            0,

            255,

            cv2.NORM_MINMAX

        ).astype(np.uint8)

        gray = cv2.cvtColor(

            rgb,

            cv2.COLOR_RGB2GRAY

        )

        feature["Blur_Score"] = float(

            cv2.Laplacian(

                gray,

                cv2.CV_64F

            ).var()

        )

        # ==========================================================
        # CANOPY DENSITY
        # ==========================================================

        feature["Canopy_Density"] = float(

            np.mean(

                NDVI > 0.50

            )

        )

        # ==========================================================
        # HOTSPOT AREA
        # ==========================================================

        hotspot = NDRE < 0.15

        feature["Disease_Hotspot"] = float(

            np.mean(

                hotspot

            )

        )

        # ==========================================================
        # SAVE FEATURES
        # ==========================================================

        records.append(feature)

    except Exception as e:

        print(

            f"Skipping {capture_name}:",

            e

        )

        continue
# ==========================================================
# CREATE DATAFRAME
# ==========================================================

print("\nCreating Feature DataFrame...")

df = pd.DataFrame(records)

print(df.head())

print("\nTotal UAV Samples :", len(df))

print("Total Features    :", len(df.columns))

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

df.drop_duplicates(

    inplace=True

)

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

df.replace(

    [np.inf, -np.inf],

    np.nan,

    inplace=True

)

df.fillna(

    0,

    inplace=True

)

# ==========================================================
# FEATURE SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Feature": df.columns,

    "Minimum": df.min(numeric_only=False),

    "Maximum": df.max(numeric_only=False)

})

summary.to_csv(

    "data/uav_feature_summary.csv",

    index=False

)

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

corr = df.select_dtypes(

    include=np.number

).corr()

corr.to_csv(

    "data/uav_feature_correlation.csv"

)

# ==========================================================
# NORMALIZATION
# ==========================================================

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numeric_cols = df.select_dtypes(

    include=np.number

).columns

df[numeric_cols] = scaler.fit_transform(

    df[numeric_cols]

)

# ==========================================================
# SAVE SCALER
# ==========================================================

import joblib

joblib.dump(

    scaler,

    "data/uav_feature_scaler.pkl"

)

joblib.dump(

    list(df.columns),

    "data/uav_feature_columns.pkl"

)

# ==========================================================
# SAVE CSV
# ==========================================================

df.to_csv(

    OUTPUT_CSV,

    index=False

)

# ==========================================================
# SAVE NUMPY
# ==========================================================

np.save(

    "data/uav_expanded_features.npy",

    df.select_dtypes(

        include=np.number

    ).values

)

# ==========================================================
# VALIDATION
# ==========================================================

print("\n======================================")

print("VALIDATION")

print("======================================")

print("Rows      :", len(df))

print("Columns   :", len(df.columns))

print("NaN Count :", df.isnull().sum().sum())

print("Duplicates:", df.duplicated().sum())

assert df.isnull().sum().sum() == 0

assert df.duplicated().sum() == 0

# ==========================================================
# FINAL REPORT
# ==========================================================

print("\n======================================")

print("UAV FEATURE EXTRACTION COMPLETED")

print("======================================")

print("Generated Files")

print("--------------------------------------")

print("uav_expanded_features.csv")

print("uav_expanded_features.npy")

print("uav_feature_scaler.pkl")

print("uav_feature_columns.pkl")

print("uav_feature_summary.csv")

print("uav_feature_correlation.csv")

print("\nTotal Features :", len(df.columns))

print("Total Samples  :", len(df))

print("\nReady for AgroFedVision V2.")
