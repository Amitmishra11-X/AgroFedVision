

import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter

df = pd.read_csv("data/leaf_dataset_coarse.csv")

print("="*60)
print("CHECK 1: Image modes (RGB / L / CMYK / RGBA / P)")
print("="*60)
modes = Counter()
sizes = Counter()
bad_files = []

for i, row in df.iterrows():
    try:
        with Image.open(row["image_path"]) as img:
            modes[img.mode] += 1
            sizes[img.size] += 1
    except Exception as e:
        bad_files.append((row["image_path"], str(e)))

print("Mode distribution:", dict(modes))
print("\nTop 5 raw sizes:", sizes.most_common(5))
print("\nFiles that failed to open:", len(bad_files))
for p, e in bad_files[:10]:
    print(" ", p, "->", e)

print("\n" + "="*60)
print("CHECK 2: Pixel value sanity on a sample of 20 images")
print("="*60)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

sample = df.sample(min(20, len(df)), random_state=0)
for i, row in sample.iterrows():
    try:
        img = load_img(row["image_path"], target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        print(f"label={row['label']:25s} shape={arr.shape} "
              f"min={arr.min():.3f} max={arr.max():.3f} "
              f"mean={arr.mean():.3f} std={arr.std():.3f}")
    except Exception as e:
        print(f"FAILED to load: {row['image_path']} -> {e}")

print("\n" + "="*60)
print("CHECK 3: Are images near-identical / mostly blank?")
print("(low std = mostly flat/blank image, possible loading bug)")
print("="*60)
low_std_count = 0
all_stds = []
sample2 = df.sample(min(100, len(df)), random_state=1)
for i, row in sample2.iterrows():
    try:
        img = load_img(row["image_path"], target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        all_stds.append(arr.std())
        if arr.std() < 0.05:
            low_std_count += 1
    except Exception:
        pass

print(f"Images with std < 0.05 (suspiciously flat): {low_std_count} / {len(sample2)}")
print(f"Overall std distribution: min={min(all_stds):.4f}, "
      f"max={max(all_stds):.4f}, mean={np.mean(all_stds):.4f}")

print("\n" + "="*60)
print("CHECK 4: Label vs index sanity — does train_test_split shuffle")
print("get applied consistently? (checking encoder mapping)")
print("="*60)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoded = encoder.fit_transform(df["label"])
print("Classes:", list(encoder.classes_))
print("First 10 original labels:", df["label"].values[:10].tolist())
print("First 10 encoded labels: ", encoded[:10].tolist())
