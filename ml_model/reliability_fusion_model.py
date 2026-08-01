"""
reliability_fusion_model.py
AgroFedVision — Reliability-Aware Fusion

Weights each modality embedding by its quality score before fusion:
  - Image: per-sample quality from image_quality.csv
  - Sensor: per-sample z-score outlier score (computed at load time)
  - UAV: fixed scalar from uav_quality.json (global score)
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from sensor_transformer import build_sensor_transformer
from image_encoder import build_image_encoder

UAV_RELIABILITY = 0.7203  # from uav_quality.json


def build_reliability_fusion_model(
        num_sensor_features,
        num_classes=3,
        sensor_d_model=64,
        img_embed_dim=256,
        dropout=0.30
):
    # ── Sensor branch ──────────────────────────────────────
    sensor_encoder = build_sensor_transformer(
        num_features=num_sensor_features,
        d_model=sensor_d_model
    )
    sensor_input    = sensor_encoder.input          # (B, num_sensor_features)
    sensor_embedding = sensor_encoder.output        # (B, 64)

    # ── Image branch ───────────────────────────────────────
    image_encoder = build_image_encoder(
        embed_dim=img_embed_dim,
        dropout=dropout,
        trainable_backbone=False
    )
    image_input    = image_encoder.input            # (B, 224, 224, 3)
    image_embedding = image_encoder.output          # (B, 256)

    # ── UAV branch ─────────────────────────────────────────
    uav_input = layers.Input(shape=(4,), name="uav_input")
    x_uav = layers.Dense(128, activation="relu", name="uav_dense1")(uav_input)
    x_uav = layers.BatchNormalization(name="uav_bn1")(x_uav)
    x_uav = layers.Dropout(0.30, name="uav_drop1")(x_uav)
    x_uav = layers.Dense(64, activation="relu", name="uav_dense2")(x_uav)
    x_uav = layers.BatchNormalization(name="uav_bn2")(x_uav)
    x_uav = layers.Dropout(0.25, name="uav_drop2")(x_uav)
    x_uav = layers.Dense(32, activation="relu", name="uav_dense3")(x_uav)
    x_uav = layers.BatchNormalization(name="uav_bn3")(x_uav)
    uav_embedding = layers.Dropout(0.20, name="uav_drop3")(x_uav)  # (B, 32)

    # ── Reliability inputs ─────────────────────────────────
    # Per-sample scalars passed in alongside the data
    image_quality_input  = layers.Input(shape=(1,), name="image_quality_input")
    sensor_quality_input = layers.Input(shape=(1,), name="sensor_quality_input")
    # UAV quality is fixed — no input needed

    # ── Weighted embeddings ────────────────────────────────
    # Multiply each embedding by its reliability weight
    # shape: (B, embed_dim) * (B, 1) → broadcasts correctly
    image_weighted  = layers.Multiply(name="image_weighted")(
        [image_embedding, image_quality_input]
    )
    sensor_weighted = layers.Multiply(name="sensor_weighted")(
        [sensor_embedding, sensor_quality_input]
    )
    # UAV: multiply by fixed scalar via Lambda
    uav_weighted = layers.Lambda(
        lambda x: x * UAV_RELIABILITY,
        name="uav_weighted"
    )(uav_embedding)

    # ── Fusion ─────────────────────────────────────────────
    fusion = layers.Concatenate(name="feature_fusion")(
        [sensor_weighted, uav_weighted, image_weighted]
    )

    # ── Classification head (same as baseline) ─────────────
    x = layers.Dense(256, activation="relu", name="head_dense1")(fusion)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout, name="head_drop1")(x)

    residual = x
    x = layers.Dense(256, activation="relu", name="head_dense2")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(dropout, name="head_drop2")(x)
    x = layers.Add(name="head_residual")([x, residual])

    x = layers.Dense(128, activation="relu", name="head_dense3")(x)
    x = layers.BatchNormalization(name="head_bn3")(x)
    x = layers.Dropout(dropout, name="head_drop3")(x)

    x = layers.Dense(64, activation="relu", name="head_dense4")(x)
    x = layers.BatchNormalization(name="head_bn4")(x)
    x = layers.Dropout(0.20, name="head_drop4")(x)

    output = layers.Dense(num_classes, activation="softmax",
                          name="crop_health_output")(x)

    model = tf.keras.Model(
        inputs=[sensor_input, uav_input, image_input,
                image_quality_input, sensor_quality_input],
        outputs=output,
        name="AgroFedVision_ReliabilityAware"
    )
    return model