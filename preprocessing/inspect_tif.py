import tifffile as tiff
import numpy as np

import tifffile as tiff
import numpy as np

img = tiff.imread(
    r"D:\Download1\IMG_0041_2.tif"
)

print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Min:", np.min(img))
print("Max:", np.max(img))

print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Min:", np.min(img))
print("Max:", np.max(img))