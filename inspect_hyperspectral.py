import numpy as np

dataset_path = r"C:\Users\Lenovo\OneDrive\Attachments\Agriculture - Hyperspectral (UC-HSI Crop variety dataset)\Crop_dataset"

print("=" * 60)
print("Loading X_train.npy")
print("=" * 60)

X = np.load(dataset_path + r"\X_train.npy")

print("Shape      :", X.shape)
print("Data Type  :", X.dtype)
print("Dimensions :", X.ndim)

print("\nFirst Sample Shape:")

if X.ndim > 1:
    print(X[0].shape)
else:
    print(X[0])

print("\n" + "=" * 60)
print("Loading y_train.npy")
print("=" * 60)

y = np.load(dataset_path + r"\y_train.npy")

print("Shape :", y.shape)
print("Unique Labels :", np.unique(y))
print("Number of Classes :", len(np.unique(y)))