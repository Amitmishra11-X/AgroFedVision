import numpy as np

dataset_path = r"C:\Users\Lenovo\OneDrive\Attachments\Agriculture - Hyperspectral (UC-HSI Crop variety dataset)\Crop_dataset"

X = np.load(dataset_path + r"\X_train.npy")

print("Shape:", X.shape)

print("Total NaN values:", np.isnan(X).sum())

print("Total Infinite values:", np.isinf(X).sum())

print("Total Elements:", X.size)