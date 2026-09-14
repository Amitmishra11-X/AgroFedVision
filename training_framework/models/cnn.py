from tensorflow.keras import layers
from tensorflow.keras import models

from config import IMAGE_SIZE


def build_cnn(num_classes):

    model = models.Sequential([

        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(96, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),

        layers.Dropout(0.30),

        layers.Dense(num_classes, activation="softmax")

    ])

    return model