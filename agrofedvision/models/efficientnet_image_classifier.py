"""EfficientNetB0 image classifier for Phase 1 centralized training."""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input

from agrofedvision.training.config import Phase1Config


@keras.saving.register_keras_serializable(package="AgroFedVision")
class EfficientNetPreprocess(layers.Layer):
    """Serializable EfficientNet preprocessing layer."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return preprocess_input(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


def build_image_augmentation(seed: int) -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.08, fill_mode="reflect", seed=seed),
            layers.RandomZoom(height_factor=0.12, width_factor=0.12, seed=seed),
            layers.RandomContrast(0.15, seed=seed),
            layers.RandomBrightness(0.12, value_range=(0.0, 255.0), seed=seed),
        ],
        name="image_augmentation",
    )


def build_efficientnet_b0_classifier(config: Phase1Config) -> keras.Model:
    inputs = keras.Input(shape=config.input_shape, name="image")
    x = build_image_augmentation(config.seed)(inputs)
    x = EfficientNetPreprocess(name="efficientnet_preprocess")(x)

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=config.input_shape,
        pooling=None,
        name="efficientnetb0_backbone",
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dropout(config.dropout_rates[0], name="head_dropout_1")(x)
    x = layers.Dense(config.dense_units[0], activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(config.dropout_rates[1], name="head_dropout_2")(x)
    x = layers.Dense(config.dense_units[1], activation="relu", name="dense_128")(x)
    x = layers.BatchNormalization(name="head_bn_3")(x)
    x = layers.Dropout(config.dropout_rates[2], name="head_dropout_3")(x)
    outputs = layers.Dense(
        config.num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="AgroFedVision_EfficientNetB0")


def get_backbone(model: keras.Model) -> keras.Model:
    layer = model.get_layer("efficientnetb0_backbone")
    if not isinstance(layer, keras.Model):
        raise TypeError("Expected efficientnetb0_backbone to be a nested Keras model.")
    return layer


def configure_classifier_training(model: keras.Model) -> None:
    get_backbone(model).trainable = False


def configure_fine_tuning(model: keras.Model, config: Phase1Config) -> None:
    backbone = get_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = layer.name.startswith(config.fine_tune_prefixes)
