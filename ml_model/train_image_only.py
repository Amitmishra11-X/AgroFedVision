import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from image_encoder import build_image_encoder

# ===================================================
# CONFIG
# ===================================================

CSV_FILE = "data/multimodal_dataset.csv"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 10

RESULTS = "results/image_only"
os.makedirs(RESULTS, exist_ok=True)

# ===================================================
# LOAD DATA
# ===================================================

print("\nLoading Dataset...")

df = pd.read_csv(CSV_FILE)

encoder = LabelEncoder()
y = encoder.fit_transform(df["crop_health"])

num_classes = len(encoder.classes_)

print("Classes:", encoder.classes_)

images = []
labels = []

for _, row in df.iterrows():

    try:

        img = load_img(
            row["image_path"],
            target_size=(IMG_SIZE, IMG_SIZE)
        )

        img = img_to_array(img)

        images.append(img)

        labels.append(
            encoder.transform([row["crop_health"]])[0]
        )

    except:

        pass

X = np.array(images, dtype=np.float32)
y = np.array(labels)

print("Images:", len(X))

# ===================================================
# SPLIT
# ===================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ===================================================
# DATASET
# ===================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_ds = (
    train_ds
    .shuffle(len(y_train))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_ds = (
    val_ds
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ===================================================
# MODEL
# ===================================================

print("\nBuilding Image Model...")

encoder_model = build_image_encoder(
    embed_dim=256,
    dropout=0.3,
    trainable_backbone=False
)

inputs = encoder_model.input

x = encoder_model.output

x = tf.keras.layers.Dense(
    256,
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

# ===================================================
# COMPILE
# ===================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-4
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            RESULTS,
            "best_image_model.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True
    )

]

# ===================================================
# STAGE 1
# ===================================================

print("\nStage 1")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS_STAGE1,

    callbacks=callbacks

)

best_acc = max(history.history["val_accuracy"])

print("\nStage 1 Best:", best_acc)

# ===================================================
# STAGE 2
# ===================================================

if best_acc >= 0.35:

    print("\nFine Tuning...")

    for layer in encoder_model.layers:

        if "efficientnetb0" in layer.name:

            layer.trainable = True

            for l in layer.layers[:-20]:

                l.trainable = False

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    history2 = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS_STAGE2,

        callbacks=callbacks

    )

# ===================================================
# EVALUATE
# ===================================================

loss, acc = model.evaluate(val_ds)

print("\nValidation Accuracy:", acc)

# ===================================================
# SAVE
# ===================================================

model.save(

    os.path.join(
        RESULTS,
        "image_only_model.keras"
    )

)

joblib.dump(

    encoder,

    os.path.join(
        RESULTS,
        "label_encoder.pkl"
    )

)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")

plt.legend(["Train","Val"])

plt.subplot(1,2,2)

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Loss")

plt.legend(["Train","Val"])

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS,
        "training_curve.png"
    )

)

print("\nDone.")