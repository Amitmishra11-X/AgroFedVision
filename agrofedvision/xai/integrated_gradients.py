"""Integrated Gradients explanations for Keras models."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


class IntegratedGradientsExplainer:
    def __init__(self, model: keras.Model, steps: int = 32) -> None:
        self.model = model
        self.steps = steps

    def explain(
        self,
        image: np.ndarray,
        class_index: int | None = None,
        baseline: np.ndarray | None = None,
    ) -> np.ndarray:
        image = image.astype("float32")
        if baseline is None:
            baseline = np.zeros_like(image, dtype="float32")
        alphas = tf.linspace(0.0, 1.0, self.steps + 1)
        interpolated = baseline[None, ...] + alphas[:, None, None, None] * (
            image[None, ...] - baseline[None, ...]
        )
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = self.model(interpolated, training=False)
            if class_index is None:
                class_index = int(tf.argmax(predictions[-1]))
            target = predictions[:, class_index]
        grads = tape.gradient(target, interpolated)
        if grads is None:
            raise RuntimeError("Unable to compute gradients for Integrated Gradients.")
        avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)
        attribution = (image - baseline) * avg_grads.numpy()
        return attribution

    @staticmethod
    def normalize_attribution(attribution: np.ndarray) -> np.ndarray:
        score = np.sum(np.abs(attribution), axis=-1)
        return score / (np.max(score) + 1e-8)
