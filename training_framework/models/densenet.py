"""
DenseNet121 Builder
AgroFedVision Research Framework
"""

from tensorflow.keras.applications import DenseNet121

from .common import classification_head

from config import IMAGE_SIZE


def build_densenet(num_classes):

    base = DenseNet121(

        include_top=False,

        weights="imagenet",

        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)

    )

    base.trainable = False

    return classification_head(

        base,

        num_classes

    )