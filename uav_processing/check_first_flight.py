import tifffile as tiff
import matplotlib.pyplot as plt
import os

BASE = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"

for img_id in [23,24,25,30,40]:

    path = os.path.join(
        BASE,
        f"IMG_{img_id:04d}_1.tif"
    )

    img = tiff.imread(path)

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.title(f"IMG_{img_id:04d}")
    plt.axis("off")

    plt.savefig(
        f"results/IMG_{img_id:04d}.png",
        bbox_inches="tight"
    )

    plt.close()

print("Saved previews")