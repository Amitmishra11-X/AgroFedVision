import os
import pandas as pd

root = r"D:\Download1\Multi_Crop_Leaves_Disease"

data = []

for folder, _, files in os.walk(root):

    label = os.path.basename(folder)

    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(folder, file)

            data.append([path, label])

df = pd.DataFrame(
    data,
    columns=["image_path", "label"]
)

print(df.head())
print("Total Images:", len(df))

df.to_csv("data/leaf_dataset.csv", index=False)

print("Saved: data/leaf_dataset.csv")