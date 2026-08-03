"""
MobileNetV2 Builder
"""

from tensorflow.keras.applications import MobileNetV2

from .common import classification_head

from config import IMAGE_SIZE


def build_mobilenet(num_classes):

    base = MobileNetV2(

        include_top=False,

        weights="imagenet",

        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)

    )

    base.trainable = False

    return classification_head(

        base,

        num_classes

    )