"""
ResNet50 Builder
"""

from tensorflow.keras.applications import ResNet50

from .common import classification_head

from config import IMAGE_SIZE


def build_resnet50(num_classes):

    base = ResNet50(

        include_top=False,

        weights="imagenet",

        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)

    )

    base.trainable = False

    model = classification_head(

        base,

        num_classes

    )

    return model