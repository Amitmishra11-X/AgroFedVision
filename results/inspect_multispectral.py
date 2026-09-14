# inspect_multispectral.py

import tifffile as tiff
import os

base = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"

for i in range(1,6):

    path = os.path.join(
        base,
        f"IMG_0000_{i}.tif"
    )

    img = tiff.imread(path)

    print("\n", path)
    print("Shape:", img.shape)
    print("Dtype:", img.dtype)
    print("Min:", img.min())
    print("Max:", img.max())