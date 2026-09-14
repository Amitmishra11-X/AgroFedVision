"""
============================================================
AgroFedVision Dataset Loader
============================================================
Universal Dataset Loader

Supports

✓ CNN
✓ MobileNet
✓ VGG16
✓ ResNet50
✓ EfficientNet
✓ DenseNet
✓ Xception

Each model can use its own

✓ Image Size
✓ preprocess_input()

============================================================
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
# Dataset Loader
# =====================================================

def load_dataset(

    folder,

    image_size,

    preprocess_fn,

    shuffle=True

):
    print("=" * 60)
    print("Loading Dataset : ", folder)
    print("=" * 60)
    
    dataset = image_dataset_from_directory(

        folder,

        image_size=(image_size, image_size),

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

        label_mode="int"

    )

    class_names = dataset.class_names

    # ----------------------------------------
    # Apply preprocessing
    # ----------------------------------------

    dataset = dataset.map(

        lambda x, y: (

            preprocess_fn(

                tf.cast(x, tf.float32)

            ),

            y

        ),

        num_parallel_calls=AUTOTUNE

    )

    # ----------------------------------------
    # Data Augmentation
    # ----------------------------------------

    if USE_AUGMENTATION and shuffle:

        dataset = dataset.map(

            lambda x, y: (

                data_augmentation(

                    x,

                    training=True

                ),

                y

            ),

            num_parallel_calls=AUTOTUNE

        )

    dataset = dataset.prefetch(

        AUTOTUNE

    )

    return dataset, class_names