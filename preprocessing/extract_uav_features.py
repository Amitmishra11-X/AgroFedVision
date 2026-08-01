import os
import tifffile as tiff
import numpy as np
import pandas as pd

base = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"

files = sorted([f for f in os.listdir(base) if f.endswith("_1.tif")])

results = []

for file in files:

    image_id = file.replace("_1.tif", "")

    try:
        band1 = tiff.imread(os.path.join(base, image_id + "_1.tif")).astype(float)
        band2 = tiff.imread(os.path.join(base, image_id + "_2.tif")).astype(float)
        band3 = tiff.imread(os.path.join(base, image_id + "_3.tif")).astype(float)
        band4 = tiff.imread(os.path.join(base, image_id + "_4.tif")).astype(float)
        band5 = tiff.imread(os.path.join(base, image_id + "_5.tif")).astype(float)

        # same bands used in  NDVI script
        ndvi = (band4 - band3) / (band4 + band3 + 1e-10)

        # same bands used in  NDRE script
        ndre = (band4 - band5) / (band4 + band5 + 1e-10)

        results.append([
            image_id,
            np.mean(ndvi),
            np.std(ndvi),
            np.mean(ndre),
            np.std(ndre)
        ])

        print("Processed:", image_id)

    except Exception as e:
        print("Skipped:", image_id, e)

df = pd.DataFrame(
    results,
    columns=[
        "Image_ID",
        "NDVI_Mean",
        "NDVI_Std",
        "NDRE_Mean",
        "NDRE_Std"
    ]
)

df.to_csv("uav_features.csv", index=False)

print("\nSaved: uav_features.csv")
print(df.head())