

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

df = pd.read_csv("data/leaf_dataset_coarse.csv")

# Grab 4 images each from 3 different classes — easy, distinct case
parts = []
for lbl, g in df.groupby("label"):
    parts.append(g.sample(min(4, len(g)), random_state=2))
sample_df = pd.concat(parts, ignore_index=True)
sample_df = sample_df.groupby("label").head(4)

print("Sample size:", len(sample_df))
print(sample_df["label"].value_counts())

IMG_SIZE = 224
images, labels = [], []
for _, row in sample_df.iterrows():
    img = load_img(row["image_path"], target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    images.append(arr)
    labels.append(row["label"])

X = np.array(images, dtype=np.float32)
print("\nX shape:", X.shape)

# Build ONLY the backbone, no head
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
feat = base_model(inputs, training=False)
pooled = tf.keras.layers.GlobalAveragePooling2D()(feat)
embed_model = tf.keras.Model(inputs, pooled)

embeddings = embed_model.predict(X, verbose=0)
print("\nEmbeddings shape:", embeddings.shape)

print("\n" + "="*60)
print("CHECK A: Are embeddings identical/near-identical across ALL images?")
print("="*60)
# Pairwise distances between all embeddings
from scipy.spatial.distance import pdist, squareform
dists = squareform(pdist(embeddings, metric="euclidean"))
np.fill_diagonal(dists, np.nan)
print("Min pairwise distance (excluding self):", np.nanmin(dists))
print("Max pairwise distance:", np.nanmax(dists))
print("Mean pairwise distance:", np.nanmean(dists))

print("\n" + "="*60)
print("CHECK B: Per-image embedding stats — are they all-zero or NaN?")
print("="*60)
for i, (lbl, emb) in enumerate(zip(labels, embeddings)):
    print(f"{lbl:25s} mean={emb.mean():.4f} std={emb.std():.4f} "
          f"min={emb.min():.4f} max={emb.max():.4f} "
          f"has_nan={np.isnan(emb).any()} all_zero={np.allclose(emb, 0)}")

print("\n" + "="*60)
print("CHECK C: Do embeddings within the SAME class cluster tighter")
print("than embeddings across DIFFERENT classes? (sanity for separability)")
print("="*60)
from collections import defaultdict
by_label = defaultdict(list)
for lbl, emb in zip(labels, embeddings):
    by_label[lbl].append(emb)

within_dists = []
for lbl, embs in by_label.items():
    embs = np.array(embs)
    if len(embs) > 1:
        d = pdist(embs, metric="euclidean")
        within_dists.extend(d)

between_dists = []
label_list = list(by_label.keys())
for i in range(len(label_list)):
    for j in range(i+1, len(label_list)):
        e1 = np.array(by_label[label_list[i]])
        e2 = np.array(by_label[label_list[j]])
        for a in e1:
            for b in e2:
                between_dists.append(np.linalg.norm(a - b))

print(f"Mean WITHIN-class distance:  {np.mean(within_dists):.4f}")
print(f"Mean BETWEEN-class distance: {np.mean(between_dists):.4f}")
if np.mean(between_dists) > np.mean(within_dists) * 1.1:
    print("✅ Between-class > within-class — classes ARE somewhat separable in embedding space")
else:
    print("❌ Between-class ≈ within-class — embeddings do NOT separate these classes well")
