import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

base = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"

red = tiff.imread(base + r"\IMG_0155_3.tif").astype(float)
nir = tiff.imread(base + r"\IMG_0155_4.tif").astype(float)

ndvi = (nir - red) / (nir + red + 1e-10)

print("NDVI Min:", np.min(ndvi))
print("NDVI Max:", np.max(ndvi))
print("NDVI Mean:", np.mean(ndvi))

#plt.imshow(ndvi, cmap="RdYlGn")
#plt.colorbar()
#plt.title("NDVI")
#plt.show()
plt.figure(figsize=(10,6))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar(label="NDVI")
plt.title("NDVI Crop Health Map")
plt.savefig("results/ndvi_map.png", dpi=300)
plt.show()