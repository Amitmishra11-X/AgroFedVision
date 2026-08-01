"""Runtime helpers for Keras image classifiers."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from agrofedvision.core.contracts import Modality, ModelPrediction, TaskType


def load_image_batch(image_paths: list[str], image_size: tuple[int, int]) -> np.ndarray:
    images = []
    for image_path in image_paths:
        image_bytes = tf.io.read_file(str(Path(image_path)))
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        images.append(tf.cast(image, tf.float32).numpy())
    return np.stack(images, axis=0)


def predict_image_paths(
    model: keras.Model,
    model_id: str,
    image_paths: list[str],
    class_names: list[str],
    image_size: tuple[int, int] = (260, 260),
) -> list[ModelPrediction]:
    batch = load_image_batch(image_paths, image_size)
    probabilities = model.predict(batch, verbose=0)
    predictions: list[ModelPrediction] = []
    for path, scores in zip(image_paths, probabilities, strict=False):
        index = int(np.argmax(scores))
        label = class_names[index] if index < len(class_names) else str(index)
        score_map = {
            class_names[i] if i < len(class_names) else str(i): float(score)
            for i, score in enumerate(scores)
        }
        predictions.append(
            ModelPrediction(
                modality=Modality.IMAGE,
                task=TaskType.DISEASE_CLASSIFICATION,
                model_id=model_id,
                label=label,
                confidence=float(scores[index]),
                scores=score_map,
                findings={"image_path": path},
            )
        )
    return predictions
