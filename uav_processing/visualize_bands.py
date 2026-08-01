import tifffile as tiff
import matplotlib.pyplot as plt
import os

BASE = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"

for i in range(1,6):

    img = tiff.imread(
        os.path.join(BASE, f"IMG_0000_{i}.tif")
    )

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.title(f"Band {i}")
    plt.axis("off")

    plt.savefig(
        f"results/band_{i}.png",
        bbox_inches="tight"
    )

    plt.close()

print("Saved band_1.png ... band_5.png")