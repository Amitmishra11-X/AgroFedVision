"""
Dataset Loader
AgroFedVision Research Framework
"""

import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

from config import *


AUTOTUNE = tf.data.AUTOTUNE


# =====================================================
# Data Augmentation
# =====================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(ROTATION / 180),

    tf.keras.layers.RandomZoom(ZOOM),

    tf.keras.layers.RandomContrast(0.10),

])


# =====================================================
# Normalization
# =====================================================

normalization = tf.keras.layers.Rescaling(1.0 / 255)


# =====================================================
# Dataset Loader
# =====================================================

def load_dataset(folder, shuffle=True):

    dataset = image_dataset_from_directory(

        folder,

        image_size=(IMAGE_SIZE, IMAGE_SIZE),

        batch_size=BATCH_SIZE,

        shuffle=shuffle,
        
        label_mode="int"

    )

    class_names = dataset.class_names

    dataset = dataset.map(

        lambda x, y: (normalization(x), y),

        num_parallel_calls=AUTOTUNE

    )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset, class_names