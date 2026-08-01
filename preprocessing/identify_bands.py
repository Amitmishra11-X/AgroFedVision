import tifffile as tiff
import numpy as np

base_path = r"PASTE_THE_FULL_000_FOLDER_PATH_HERE"

image_id = "0155"   # Use a file number you know exists

for i in range(1, 6):
    img = tiff.imread(f"{base_path}\\IMG_{image_id}_{i}.tif")

    print(f"\nBand {i}")
    print("Mean :", np.mean(img))
    print("Min  :", np.min(img))
    print("Max  :", np.max(img))