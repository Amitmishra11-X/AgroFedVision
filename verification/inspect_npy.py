import numpy as np

X = np.load(
r"C:\Users\Lenovo\OneDrive\Attachments\Agriculture - Hyperspectral (UC-HSI Crop variety dataset\Crop_dataset\X_train.npy"
)

print("Shape :", X.shape)
print("Data Type :", X.dtype)
print("Dimensions :", X.ndim)