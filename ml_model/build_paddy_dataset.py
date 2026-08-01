import os
import pandas as pd


DATASET_ROOT = r"D:\Download1\archive\paddy-disease-classification"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train_images")

records = []

print("Scanning dataset...")

for disease in sorted(os.listdir(TRAIN_DIR)):

    disease_path = os.path.join(TRAIN_DIR, disease)

    if not os.path.isdir(disease_path):
        continue

    for img in os.listdir(disease_path):

        if img.lower().endswith((".jpg", ".jpeg", ".png")):

            records.append({
                "image_path": os.path.join(disease_path, img),
                "crop_health": disease
            })

df = pd.DataFrame(records)

print(df.head())

print()
print(df["crop_health"].value_counts())

os.makedirs("data", exist_ok=True)

csv_path = "data/paddy_dataset.csv"

df.to_csv(csv_path, index=False)

print()
print(f"Saved {len(df)} images")
print(csv_path)