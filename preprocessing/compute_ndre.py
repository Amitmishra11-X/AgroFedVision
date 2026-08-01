import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

base = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"


nir = tiff.imread(base + r"\IMG_0155_4.tif").astype(float)
rededge = tiff.imread(base + r"\IMG_0155_5.tif").astype(float)

ndre = (nir - rededge) / (nir + rededge + 1e-10)

print("NDRE Min:", np.min(ndre))
print("NDRE Max:", np.max(ndre))
print("NDRE Mean:", np.mean(ndre))

plt.figure(figsize=(10,6))
plt.imshow(ndre, cmap="RdYlGn")
plt.colorbar(label="NDRE")
plt.title("NDRE Crop Health Map")
plt.savefig("ndre_map.png", dpi=300)
plt.show()