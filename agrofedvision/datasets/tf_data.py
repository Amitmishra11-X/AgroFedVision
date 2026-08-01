"""TensorFlow input pipelines for image data."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from agrofedvision.training.config import Phase1Config


AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize(path: tf.Tensor, label: tf.Tensor, image_size: tuple[int, int]):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, method="bilinear")
    image = tf.cast(image, tf.float32)
    image.set_shape([image_size[0], image_size[1], 3])
    return image, tf.cast(label, tf.int32)


def build_image_dataset(
    image_paths: np.ndarray,
    labels: np.ndarray,
    config: Phase1Config,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths.astype(str), labels.astype("int32"))
    )
    dataset = dataset.map(
        lambda path, label: _decode_resize(path, label, config.image_size),
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )
    if config.cache_datasets:
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(image_paths), 4096),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(config.batch_size).prefetch(AUTOTUNE)
