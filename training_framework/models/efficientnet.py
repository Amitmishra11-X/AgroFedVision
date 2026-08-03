"""
EfficientNetB0 Builder
"""

from tensorflow.keras.applications import EfficientNetB0

from .common import classification_head

from config import IMAGE_SIZE


def build_efficientnet(num_classes):

    base = EfficientNetB0(

        include_top=False,

        weights="imagenet",

        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)

    )

    base.trainable = False

    return classification_head(

        base,

        num_classes

    )