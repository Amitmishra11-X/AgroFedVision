import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/leaf_dataset_coarse.csv")

print("Dataset Shape:", df.shape)

# ==========================================
# ENCODE LABELS
# ==========================================

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])
num_classes = len(encoder.classes_)

print("\nClasses:")
for i, cls in enumerate(encoder.classes_):
    print(i, cls)

# ==========================================
# LOAD IMAGES — NO manual /255.0 !!
# EfficientNet's preprocess_input expects raw [0,255]
# pixel values and does its own correct normalization.
# ==========================================

IMG_SIZE = 224
images = []
labels = []

for _, row in df.iterrows():
    try:
        img = load_img(row["image_path"], target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img)          # stays in [0, 255] range
        images.append(img)
        labels.append(row["label"])
    except Exception:
        print("Failed:", row["image_path"])

X = np.array(images, dtype=np.float32)   # still [0, 255] here
y = np.array(labels)

print("\nLoaded Images:", len(X))
print("Raw pixel range BEFORE preprocess_input:", X.min(), "-", X.max())

# Apply the CORRECT EfficientNet preprocessing
X = preprocess_input(X)
print("Pixel range AFTER preprocess_input:", X.min(), "-", X.max())

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ==========================================
# DATA AUGMENTATION (mild — not too aggressive)
# ==========================================

data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ==========================================
# EFFICIENTNET — FROZEN BACKBONE
# ==========================================

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_aug(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "results/best_leaf_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# ==========================================
# STAGE 1 — TRAIN HEAD ONLY
# ==========================================

print("\n=== STAGE 1: Training classifier head (frozen backbone) ===\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

stage1_acc = max(history.history["val_accuracy"])
print(f"\nStage 1 best val_accuracy: {stage1_acc:.4f}")

# ==========================================
# SAFETY CHECK before fine-tuning
# ==========================================

if stage1_acc < 0.15:
    print("\n⚠️  Stage 1 accuracy is very low (<15%). Stopping before fine-tuning.")
    model.save("results/leaf_disease_model_stage1_only.keras")
else:
    print("\n=== STAGE 2: Fine-tuning last 30 layers ===\n")

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_fine = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=16,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    loss, acc = model.evaluate(X_test, y_test)
    print("\nFinal Accuracy:", acc)

    model.save("results/leaf_disease_model.keras")
    print("\nModel Saved: results/leaf_disease_model.keras")
