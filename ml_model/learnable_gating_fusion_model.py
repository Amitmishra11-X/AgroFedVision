"""
learnable_gating_fusion_model.py

AgroFedVision V3

Adaptive Learnable Gating Fusion

Image
Sensor
UAV

↓

Learnable Gates

↓

Fusion

↓

Crop Health Prediction
"""

import tensorflow as tf
from tensorflow.keras import layers

from sensor_transformer import build_sensor_transformer
from image_encoder import build_image_encoder


# ==========================================================
# GATING BLOCK
# ==========================================================

def learnable_gate(

        embedding,

        name

):
    """
    Learns a confidence score (0-1)
    for each modality.
    """

    gate = layers.Dense(

        32,

        activation="relu",

        name=f"{name}_gate_dense"

    )(embedding)

    gate = layers.Dense(

        1,

        activation="sigmoid",

        name=f"{name}_gate_score"

    )(gate)

    gated_embedding = layers.Multiply(

        name=f"{name}_gated"

    )([

        embedding,

        gate

    ])

    return gated_embedding, gate


# ==========================================================
# BUILD MODEL
# ==========================================================

def build_learnable_gating_model(

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

    print(

        "Sensor Embedding:",

        sensor_embedding.shape

    )

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

    print(

        "Image Embedding:",

        image_embedding.shape

    )

    # ======================================================
    # UAV INPUT
    # ======================================================

    uav_input = layers.Input(

        shape=(num_uav_features,),

        name="uav_input"

    )
    # ======================================================
    # UAV BRANCH
    # ======================================================

    # Block 1

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

    # Block 2

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

    # Block 3

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

    print(

        "UAV Embedding:",

        uav_embedding.shape

    )

    # ======================================================
    # LEARNABLE GATES
    # ======================================================

    sensor_embedding, sensor_gate = learnable_gate(

        sensor_embedding,

        "sensor"

    )

    image_embedding, image_gate = learnable_gate(

        image_embedding,

        "image"

    )

    uav_embedding, uav_gate = learnable_gate(

        uav_embedding,

        "uav"

    )

    print(

        "Learnable Gating Added."

    )

    print(

        "Sensor Gate :", sensor_gate.shape

    )

    print(

        "Image Gate  :", image_gate.shape

    )

    print(

        "UAV Gate    :", uav_gate.shape

    ) 
    # ======================================================
    # MULTIMODAL FUSION
    # ======================================================

    fusion = layers.Concatenate(

        name="feature_fusion"

    )([

        sensor_embedding,

        uav_embedding,

        image_embedding

    ])

    print(

        "Fusion Shape:",

        fusion.shape

    )

    # ======================================================
    # CLASSIFICATION HEAD
    # (IDENTICAL TO BASELINE)
    # ======================================================

    x = layers.Dense(

        256,

        activation="relu",

        name="fusion_dense1"

    )(fusion)

    x = layers.BatchNormalization(

        name="fusion_bn1"

    )(x)

    x = layers.Dropout(

        dropout,

        name="fusion_drop1"

    )(x)

    # ------------------------------------------------------
    # Residual Block
    # ------------------------------------------------------

    residual = x

    x = layers.Dense(

        256,

        activation="relu",

        name="fusion_dense2"

    )(x)

    x = layers.BatchNormalization(

        name="fusion_bn2"

    )(x)

    x = layers.Dropout(

        dropout,

        name="fusion_drop2"

    )(x)

    x = layers.Add(

        name="fusion_residual"

    )([

        x,

        residual

    ])

    # ------------------------------------------------------
    # Dense Block
    # ------------------------------------------------------

    x = layers.Dense(

        128,

        activation="relu",

        name="fusion_dense3"

    )(x)

    x = layers.BatchNormalization(

        name="fusion_bn3"

    )(x)

    x = layers.Dropout(

        dropout,

        name="fusion_drop3"

    )(x)

    x = layers.Dense(

        64,

        activation="relu",

        name="fusion_dense4"

    )(x)

    x = layers.BatchNormalization(

        name="fusion_bn4"

    )(x)

    x = layers.Dropout(

        0.20,

        name="fusion_drop4"

    )(x)

    # ======================================================
    # OUTPUT
    # ======================================================

    output = layers.Dense(

        num_classes,

        activation="softmax",

        name="crop_health_output"

    )(x)

    print(

        "Output Shape:",

        output.shape

    )
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

        name="AgroFedVision_Learnable_Gating"

    )

    print("\n======================================")

    print("AgroFedVision Learnable Gating Model")

    print("======================================")

    print("Sensor Features :", num_sensor_features)

    print("UAV Features    :", num_uav_features)

    print("Classes         :", num_classes)

    print("Dropout         :", dropout)

    print("======================================")

    return model


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    model = build_learnable_gating_model(

        num_sensor_features=22,

        num_uav_features=4,

        num_classes=3

    )

    model.summary()