"""Grad-CAM and Grad-CAM++ explanations for Keras image models."""

from __future__ import annotations

from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf


class GradCAMExplainer:
    def __init__(self, model: keras.Model, last_conv_layer_name: str | None = None) -> None:
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name or self._find_last_conv_layer()

    def explain(
        self,
        image_batch: np.ndarray,
        class_index: int | None = None,
        use_plus_plus: bool = False,
    ) -> np.ndarray:
        grad_model = keras.Model(
            self.model.inputs,
            [
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output,
            ],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_batch, training=False)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, class_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            raise RuntimeError("Unable to compute gradients for Grad-CAM.")

        if use_plus_plus:
            heatmap = self._gradcam_plus_plus(conv_outputs, grads)
        else:
            weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
            heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap, axis=(1, 2), keepdims=True) + 1e-8)
        return heatmap.numpy()

    def save_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        output_path: Path,
        alpha: float = 0.35,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        base = np.uint8(np.clip(image, 0, 255))
        overlay = cv2.addWeighted(base[..., ::-1], 1.0 - alpha, heatmap_rgb, alpha, 0)
        cv2.imwrite(str(output_path), overlay)
        return output_path

    def _find_last_conv_layer(self) -> str:
        for layer in reversed(self.model.layers):
            output = getattr(layer, "output", None)
            shape = getattr(output, "shape", None)
            if shape is not None and len(shape) == 4:
                return layer.name
        raise ValueError("No 4D convolutional feature output found for Grad-CAM.")

    @staticmethod
    def _gradcam_plus_plus(conv_outputs: tf.Tensor, grads: tf.Tensor) -> tf.Tensor:
        grads_power_2 = tf.square(grads)
        grads_power_3 = grads_power_2 * grads
        denominator = 2.0 * grads_power_2 + tf.reduce_sum(
            conv_outputs * grads_power_3,
            axis=(1, 2),
            keepdims=True,
        )
        alphas = grads_power_2 / (denominator + 1e-8)
        weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(1, 2), keepdims=True)
        return tf.reduce_sum(weights * conv_outputs, axis=-1)

