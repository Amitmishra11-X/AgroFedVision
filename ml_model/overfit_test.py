"""
overfit_test.py — Can the model memorize just 40 images?
If NOT, the bug is in data loading, not training hyperparameters.
Run: python overfit_test.py
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array

df = pd.read_csv("data/leaf_dataset_coarse.csv")

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])
num_classes = len(encoder.classes_)

# Take just 8 images per class (or fewer if class is smaller) — small, easy subset
parts = []
for lbl, g in df.groupby("label"):
    parts.append(g.sample(min(8, len(g)), random_state=0))
small_df = pd.concat(parts, ignore_index=True)

print("Tiny subset size:", len(small_df))
print(small_df["label"].value_counts())

IMG_SIZE = 224
images, labels = [], []
for _, row in small_df.iterrows():
    img = load_img(row["image_path"], target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    images.append(img)
    labels.append(row["label"])

X = np.array(images, dtype=np.float32)
y = np.array(labels)
print("\nX shape:", X.shape, "y shape:", y.shape)
print("Unique labels in subset:", np.unique(y))

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=== Attempting to OVERFIT on tiny subset (no val split) ===")
print("Expectation: accuracy should reach >90% within 30 epochs.")
print("If it stays near chance, the bug is in data loading, not hyperparameters.\n")

history = model.fit(X, y, epochs=30, batch_size=8, verbose=1)

final_acc = history.history["accuracy"][-1]
print(f"\nFinal training accuracy on tiny subset: {final_acc:.4f}")
if final_acc > 0.85:
    print("✅ Model CAN learn — pipeline is fine. Problem is dataset size/difficulty at scale.")
else:
    print("❌ Model CANNOT even memorize 40 images — bug is in data loading/array construction.")
