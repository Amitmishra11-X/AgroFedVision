"""
train_multimodal.py

AgroFedVision

Image + Sensor + UAV Fusion
"""

import os
import random
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from learnable_gating_fusion_model import (
    build_learnable_gating_model
)

# ==========================================================
# CONFIG
# ==========================================================

DATASET = "data/multimodal_dataset.csv"

RESULTS_DIR = "results"

IMG_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 25

LR = 3e-4

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ==========================================================
# UAV FEATURES
# ==========================================================

UAV_COLS = [

    "NDVI_Mean",

    "NDVI_Std",

    "NDRE_Mean",

    "NDRE_Std"

]

# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    print("\nLoading Dataset...")

    df = pd.read_csv(DATASET)

    print(df.shape)

    # ------------------------------------------------------

    # LABELS

    # ------------------------------------------------------

    encoder = LabelEncoder()

    y = encoder.fit_transform(

        df["crop_health"]

    )

    print("\nClasses")

    print(

        list(

            encoder.classes_

        )

    )

    # ------------------------------------------------------

    # SENSOR FEATURES

    # ------------------------------------------------------

    sensor_cols = [

        c

        for c in df.columns

        if c not in

        [

            "image_path",

            "crop_health",

            "leaf_label"

        ] + UAV_COLS

    ]

    sensor_scaler = MinMaxScaler()

    X_sensor = sensor_scaler.fit_transform(

        df[sensor_cols]

    ).astype(np.float32)

    print(

        "\nSensor Features:",

        X_sensor.shape[1]

    )

    # ------------------------------------------------------

    # UAV FEATURES

    # ------------------------------------------------------

    uav_scaler = MinMaxScaler()

    X_uav = uav_scaler.fit_transform(

        df[UAV_COLS]

    ).astype(np.float32)

    print(

        "UAV Features:",

        X_uav.shape[1]

    )

    # ------------------------------------------------------

    # IMAGES

    # ------------------------------------------------------

    images = []

    valid_idx = []

    for i, path in enumerate(

        df["image_path"]

    ):

        try:

            img = load_img(

                path,

                target_size=(

                    IMG_SIZE,

                    IMG_SIZE

                )

            )

            img = img_to_array(img)

            images.append(img)

            valid_idx.append(i)

        except:

            pass

    X_img = np.array(

        images,

        dtype=np.float32

    )

    X_sensor = X_sensor[valid_idx]

    X_uav = X_uav[valid_idx]

    y = y[valid_idx]

    print(

        "\nImages Loaded:",

        len(X_img)

    )

    return (

        X_sensor,

        X_uav,

        X_img,

        y,

        len(encoder.classes_),

        encoder,

        sensor_scaler,

        uav_scaler,

        sensor_cols,

        UAV_COLS

    )
# ==========================================================
# BUILD TF DATASET
# ==========================================================

def build_tf_dataset(

    X_sensor,

    X_uav,

    X_img,

    y,

    batch_size,

    shuffle=True

):

    ds = tf.data.Dataset.from_tensor_slices(

        (

            {

                "sensor_input": X_sensor,

                "uav_input": X_uav,

                "image_input": X_img

            },

            y

        )

    )

    if shuffle:

        ds = ds.shuffle(

            buffer_size=len(y),

            seed=SEED,
            
            reshuffle_each_iteration=False

        )

    ds = ds.batch(

        batch_size

    )

    ds = ds.prefetch(

        tf.data.AUTOTUNE

    )

    return ds


# ==========================================================
# TRAIN
# ==========================================================

def train():

    print("\n======================================")

    print("Loading Data")

    print("======================================")

    (

        X_sensor,

        X_uav,

        X_img,

        y,

        num_classes,

        encoder,

        sensor_scaler,

        uav_scaler,

        sensor_cols,

        uav_cols

    ) = load_data()

    # ------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # ------------------------------------------------------

    idx = np.arange(

        len(y)

    )

    train_idx, val_idx = train_test_split(

        idx,

        test_size=0.20,

        random_state=SEED,

        stratify=y

    )

    train_ds = build_tf_dataset(

        X_sensor[train_idx],

        X_uav[train_idx],

        X_img[train_idx],

        y[train_idx],

        BATCH_SIZE,

        shuffle=True

    )

    val_ds = build_tf_dataset(

        X_sensor[val_idx],

        X_uav[val_idx],

        X_img[val_idx],

        y[val_idx],

        BATCH_SIZE,

        shuffle=False

    )

    print("\n======================================")

    print("Building AgroFedVision")

    print("======================================")

    model = build_learnable_gating_model(

        num_sensor_features=X_sensor.shape[1],
        num_uav_features=X_uav.shape[1],
        num_classes=num_classes,

        sensor_d_model=64,

        img_embed_dim=256,

        dropout=0.3,

        

    )

    model.summary()

    print(

        "\nTrainable Parameters:",

        np.sum(

            [

                np.prod(v.shape)

                for v in model.trainable_weights

            ]

        )

    )

    # ------------------------------------------------------
    # COMPILE
    # ------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LR

        ),

        loss="sparse_categorical_crossentropy",

        metrics=[

            "accuracy"

        ]

    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_accuracy",

            patience=6,

            restore_best_weights=True,

            verbose=1

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

                RESULTS_DIR,

                "fusion_model_best.keras"

            ),

            monitor="val_accuracy",

            save_best_only=True,

            verbose=1

        )

    ]
    # ======================================================
    # STAGE 1
    # ======================================================

    print("\n======================================")
    print("Stage 1 : Training Fusion Model")
    print("======================================")

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        callbacks=callbacks,

        verbose=1

    )

    stage1_best = max(

        history.history["val_accuracy"]

    )

    print(

        "\nStage 1 Best Accuracy:",

        stage1_best

    )

    # ======================================================
    # STAGE 2 : FINE TUNING
    # ======================================================

    history2 = None

    if stage1_best >= 0.35:

        print("\n======================================")
        print("Fine Tuning EfficientNet")
        print("======================================")

        # ----------------------------------------------
        # Unfreeze EfficientNet Backbone
        # ----------------------------------------------

        for layer in model.layers:

            if "efficientnetb0" in layer.name.lower():

                layer.trainable = True

                for sublayer in layer.layers[:-20]:

                    sublayer.trainable = False

                break

        model.compile(

            optimizer=tf.keras.optimizers.Adam(

                learning_rate=1e-5

            ),

            loss="sparse_categorical_crossentropy",

            metrics=[

                "accuracy"

            ]

        )

        history2 = model.fit(

            train_ds,

            validation_data=val_ds,

            epochs=10,

            callbacks=callbacks,

            verbose=1

        )

    # ======================================================
    # FINAL EVALUATION
    # ======================================================

    print("\n======================================")
    print("Evaluating Model")
    print("======================================")

    loss, accuracy = model.evaluate(

        val_ds,

        verbose=1

    )

    print("\n======================================")
    print("MULTIMODAL RESULTS")
    print("======================================")

    print(f"Validation Loss     : {loss:.4f}")
    print(f"Validation Accuracy : {accuracy:.4f}")
    # ======================================================
    # SAVE MODEL
    # ======================================================

    model.save(
        os.path.join(
            RESULTS_DIR,
            "agrofedvision_fusion_model.keras"
        )
    )

    # ======================================================
    # SAVE SCALERS & ENCODERS
    # ======================================================

    joblib.dump(
        encoder,
        os.path.join(
            RESULTS_DIR,
            "fusion_label_encoder.pkl"
        )
    )

    joblib.dump(
        sensor_scaler,
        os.path.join(
            RESULTS_DIR,
            "fusion_sensor_scaler.pkl"
        )
    )

    joblib.dump(
        uav_scaler,
        os.path.join(
            RESULTS_DIR,
            "fusion_uav_scaler.pkl"
        )
    )

    joblib.dump(
        sensor_cols,
        os.path.join(
            RESULTS_DIR,
            "fusion_sensor_cols.pkl"
        )
    )

    joblib.dump(
        uav_cols,
        os.path.join(
            RESULTS_DIR,
            "fusion_uav_cols.pkl"
        )
    )

    # ======================================================
    # TRAINING HISTORY
    # ======================================================

    history_df = pd.DataFrame(history.history)

    if history2 is not None:

        history2_df = pd.DataFrame(history2.history)

        history_df = pd.concat(
            [history_df, history2_df],
            ignore_index=True
        )

    history_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "fusion_training_history.csv"
        ),
        index=False
    )

    # ======================================================
    # METRICS
    # ======================================================

    metrics = {

        "Validation Accuracy": float(accuracy),

        "Validation Loss": float(loss),

        "Epochs": len(history_df),

        "Sensor Features": len(sensor_cols),

        "UAV Features": len(uav_cols),

        "Classes": list(encoder.classes_)

    }

    with open(
        os.path.join(
            RESULTS_DIR,
            "fusion_metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # ======================================================
    # TRAINING CURVE
    # ======================================================

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)

    plt.plot(
        history_df["accuracy"],
        label="Train"
    )

    plt.plot(
        history_df["val_accuracy"],
        label="Validation"
    )

    plt.title("Accuracy")

    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(
        history_df["loss"],
        label="Train"
    )

    plt.plot(
        history_df["val_loss"],
        label="Validation"
    )

    plt.title("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "fusion_training_curve.png"
        ),
        dpi=300
    )

    plt.close()

    # ======================================================
    # SUMMARY
    # ======================================================

    print("\n======================================")

    print("MULTIMODAL RESULTS")

    print("======================================")

    print(f"Validation Loss     : {loss:.4f}")

    print(f"Validation Accuracy : {accuracy:.4f}")

    print("\n======================================")

    print("Saved Files")

    print("======================================")

    print("agrofedvision_fusion_model.keras")

    print("fusion_label_encoder.pkl")

    print("fusion_sensor_scaler.pkl")

    print("fusion_uav_scaler.pkl")

    print("fusion_sensor_cols.pkl")

    print("fusion_uav_cols.pkl")

    print("fusion_training_history.csv")

    print("fusion_metrics.json")

    print("fusion_training_curve.png")

    print("\n======================================")

    print("Training Completed")

    print("======================================")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    train()