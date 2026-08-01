import tifffile as tiff
import numpy as np

base = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"

b1 = tiff.imread(base + r"\IMG_0155_1.tif")
b2 = tiff.imread(base + r"\IMG_0155_2.tif")
b3 = tiff.imread(base + r"\IMG_0155_3.tif")
b4 = tiff.imread(base + r"\IMG_0155_4.tif")
b5 = tiff.imread(base + r"\IMG_0155_5.tif")

cube = np.stack([b1,b2,b3,b4,b5], axis=-1)

print("Cube Shape:", cube.shape)

print("Band 1 Mean:", np.mean(b1))
print("Band 2 Mean:", np.mean(b2))
print("Band 3 Mean:", np.mean(b3))
print("Band 4 Mean:", np.mean(b4))
print("Band 5 Mean:", np.mean(b5))