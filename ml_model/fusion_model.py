"""
fusion_model.py

AgroFedVision V2

Image
Sensor
Expanded UAV

→ Fusion
→ Crop Health Prediction
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from agrofedvision.models.sensor_transformer import build_sensor_transformer
from agrofedvision.models.image_encoder import build_image_encoder

# ==========================================================
# BUILD MODEL
# ==========================================================

def build_fusion_model(

        num_sensor_features,

        num_uav_features,

        num_classes=3,

        sensor_d_model=64,

        img_embed_dim=256,

        dropout=0.30

):

    # ======================================================
    # SENSOR BRANCH
    # ======================================================

    sensor_encoder = build_sensor_transformer(

        num_features=num_sensor_features,

        d_model=sensor_d_model

    )

    sensor_input = sensor_encoder.input

    sensor_embedding = sensor_encoder.output

    print("Sensor Embedding:", sensor_embedding.shape)

    # ======================================================
    # IMAGE BRANCH
    # ======================================================

    image_encoder = build_image_encoder(

        embed_dim=img_embed_dim,

        dropout=dropout,

        trainable_backbone=False

    )

    image_input = image_encoder.input

    image_embedding = image_encoder.output

    print("Image Embedding:", image_embedding.shape)
    # ======================================================
    # UAV BRANCH (Expanded Features)
    # ======================================================

    uav_input = layers.Input(

        shape=(num_uav_features,),

        name="uav_input"

    )

    # ----------------------------------------
    # Block 1
    # ----------------------------------------

    x_uav = layers.Dense(

        128,

        activation="relu",

        name="uav_dense1"

    )(uav_input)

    x_uav = layers.BatchNormalization(

        name="uav_bn1"

    )(x_uav)

    x_uav = layers.Dropout(

        0.30,

        name="uav_drop1"

    )(x_uav)

    # ----------------------------------------
    # Block 2
    # ----------------------------------------

    x_uav = layers.Dense(

        64,

        activation="relu",

        name="uav_dense2"

    )(x_uav)

    x_uav = layers.BatchNormalization(

        name="uav_bn2"

    )(x_uav)

    x_uav = layers.Dropout(

        0.25,

        name="uav_drop2"

    )(x_uav)

    # ----------------------------------------
    # Block 3
    # ----------------------------------------

    x_uav = layers.Dense(

        32,

        activation="relu",

        name="uav_dense3"

    )(x_uav)

    x_uav = layers.BatchNormalization(

        name="uav_bn3"

    )(x_uav)

    uav_embedding = layers.Dropout(

        0.20,

        name="uav_drop3"

    )(x_uav)

    print("UAV Embedding:", uav_embedding.shape)

    # ======================================================
    # FEATURE FUSION
    # ======================================================

    fusion = layers.Concatenate(

        name="feature_fusion"

    )(

        [

            sensor_embedding,

            uav_embedding,

            image_embedding

        ]

    )

    print("Fusion Shape:", fusion.shape)
    # ======================================================
    # CLASSIFICATION HEAD
    # ======================================================

    # ----------------------------------------
    # Dense Block 1
    # ----------------------------------------

    x = layers.Dense(

        256,

        activation="relu",

        name="head_dense1"

    )(fusion)

    x = layers.BatchNormalization(

        name="head_bn1"

    )(x)

    x = layers.Dropout(

        dropout,

        name="head_drop1"

    )(x)

    # ----------------------------------------
    # Dense Block 2
    # ----------------------------------------

    residual = x

    x = layers.Dense(

        256,

        activation="relu",

        name="head_dense2"

    )(x)

    x = layers.BatchNormalization(

        name="head_bn2"

    )(x)

    x = layers.Dropout(

        dropout,

        name="head_drop2"

    )(x)

    # Residual Connection
    x = layers.Add(

        name="head_residual"

    )([x, residual])

    # ----------------------------------------
    # Dense Block 3
    # ----------------------------------------

    x = layers.Dense(

        128,

        activation="relu",

        name="head_dense3"

    )(x)

    x = layers.BatchNormalization(

        name="head_bn3"

    )(x)

    x = layers.Dropout(

        dropout,

        name="head_drop3"

    )(x)

    # ----------------------------------------
    # Dense Block 4
    # ----------------------------------------

    x = layers.Dense(

        64,

        activation="relu",

        name="head_dense4"

    )(x)

    x = layers.BatchNormalization(

        name="head_bn4"

    )(x)

    x = layers.Dropout(

        0.20,

        name="head_drop4"

    )(x)

    # ======================================================
    # OUTPUT
    # ======================================================

    output = layers.Dense(

        num_classes,

        activation="softmax",

        name="crop_health_output"

    )(x)
    # ======================================================
    # BUILD MODEL
    # ======================================================

    model = tf.keras.Model(

        inputs=[

            sensor_input,

            uav_input,

            image_input

        ],

        outputs=output,

        name="AgroFedVision_V2"

    )

    return model


# ==========================================================
# MODEL SUMMARY
# ==========================================================

def get_model_summary(

        num_sensor_features,

        num_uav_features,

        num_classes=3

):

    model = build_fusion_model(

        num_sensor_features=num_sensor_features,

        num_uav_features=num_uav_features,

        num_classes=num_classes

    )

    model.summary()

    trainable = np.sum(

        [

            np.prod(v.shape)

            for v in model.trainable_weights

        ]

    )

    non_trainable = np.sum(

        [

            np.prod(v.shape)

            for v in model.non_trainable_weights

        ]

    )

    print("\n======================================")

    print("MODEL STATISTICS")

    print("======================================")

    print(f"Trainable Parameters     : {trainable:,}")

    print(f"Non-Trainable Parameters : {non_trainable:,}")

    print(f"Total Parameters         : {trainable + non_trainable:,}")

    return model


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    get_model_summary(

        num_sensor_features=19,

        num_uav_features=24,

        num_classes=3

    )