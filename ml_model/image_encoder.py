"""
image_encoder.py — EfficientNetB0 leaf image feature encoder.

Input : (batch, 224, 224, 3)
Output: (batch, embed_dim)

Compatible with:
- TensorFlow 2.20+
- Keras 3
- AgroFedVision
"""

import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


# ==========================================================
# Custom preprocessing layer
# ==========================================================

@keras.saving.register_keras_serializable()
class EfficientNetPreprocess(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return preprocess_input(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


# ==========================================================
# Image Encoder
# ==========================================================

def build_image_encoder(
    embed_dim=256,
    dropout=0.30,
    trainable_backbone=False,
):
    """
    EfficientNetB0 feature encoder.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension.

    dropout : float
        Dropout before embedding layer.

    trainable_backbone : bool
        Whether to fine-tune EfficientNet.
    """

    inputs = keras.Input(
        shape=(224, 224, 3),
        name="image_input"
    )

    x = EfficientNetPreprocess(
        name="efficientnet_preprocess"
    )(inputs)

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
    )

    backbone.trainable = trainable_backbone

    x = backbone(x, training=trainable_backbone)

    x = layers.GlobalAveragePooling2D(
        name="global_pool"
    )(x)

    x = layers.BatchNormalization(
        name="encoder_bn"
    )(x)

    x = layers.Dropout(
        dropout,
        name="encoder_dropout"
    )(x)

    embedding = layers.Dense(
        embed_dim,
        activation="relu",
        name="image_embedding",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=embedding,
        name="EfficientNetB0_ImageEncoder",
    )

    return model