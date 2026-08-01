import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from tensorflow.keras.preprocessing.image import load_img, img_to_array

sys.path.append("ml_model")

from image_encoder import EfficientNetPreprocess

IMG_SIZE = 224

UAV_COLS = [
    "NDVI_Mean",
    "NDVI_Std",
    "NDRE_Mean",
    "NDRE_Std"
]

SKIP_COLS = [
    "image_path",
    "leaf_label",
    "crop_health"
] + UAV_COLS


def load_client_data(client_csv):

    df = pd.read_csv(client_csv)

    sensor_cols = joblib.load(
        "results/fusion_sensor_cols.pkl"
    )

    scaler = joblib.load(
        "results/fusion_sensor_scaler.pkl"
    )

    label_encoder = joblib.load(
        "results/fusion_label_encoder.pkl"
    )

    # -------------------------
    # Sensor
    # -------------------------

    X_sensor = scaler.transform(
        df[sensor_cols]
    ).astype(np.float32)

    # -------------------------
    # UAV
    # -------------------------

    X_uav = df[UAV_COLS].values.astype(
        np.float32
    )

    # -------------------------
    # Labels
    # -------------------------

    y = label_encoder.transform(
        df["crop_health"]
    )

    # -------------------------
    # Images
    # -------------------------

    images = []

    for path in df["image_path"]:

        try:

            img = load_img(
                path,
                target_size=(224, 224)
            )

            img = img_to_array(
                img
            )

            images.append(img)

        except:

            images.append(
                np.zeros(
                    (224, 224, 3),
                    dtype=np.float32
                )
            )

    X_img = np.array(
        images,
        dtype=np.float32
    )

    return (
        X_sensor,
        X_uav,
        X_img,
        y
    )


def train_client(
    client_csv,
    global_model_path,
    epochs=1,
    batch_size=8
):

    print(
        f"\nTraining {client_csv}"
    )

    custom_objects = {
        "EfficientNetPreprocess":
        EfficientNetPreprocess
    }

    model = tf.keras.models.load_model(
        global_model_path,
        custom_objects=custom_objects
    )

    X_sensor, X_uav, X_img, y = load_client_data(
        client_csv
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "sensor_input":
                X_sensor,

                "uav_input":
                X_uav,

                "image_input":
                X_img
            },
            y
        )
    )

    dataset = dataset.shuffle(
        len(y)
    ).batch(
        batch_size
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        dataset,
        epochs=epochs,
        verbose=1
    )

    loss, acc = model.evaluate(
        dataset,
        verbose=0
    )

    print(
        f"Client Accuracy: {acc:.4f}"
    )

    return model.get_weights()


if __name__ == "__main__":

    weights = train_client(
        client_csv=
        "data/clients/farm_a.csv",

        global_model_path=
        "results/agrofedvision_fusion_model.keras",

        epochs=1
    )

    print(
        f"\nReturned {len(weights)} layers"
    )