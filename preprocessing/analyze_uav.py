import matplotlib
matplotlib.use("TkAgg")

import tifffile as tiff
import matplotlib.pyplot as plt

img = tiff.imread(r"D:\Download1\IMG_0041_2.tif")

plt.figure(figsize=(8,6))
plt.imshow(img, cmap="gray")
plt.colorbar()
plt.title("Band Visualization")
plt.show()