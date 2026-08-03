"""
Xception Builder
AgroFedVision Research Framework
"""

from tensorflow.keras.applications import Xception

from .common import classification_head

from config import IMAGE_SIZE


def build_xception(num_classes):

    base = Xception(

        include_top=False,

        weights="imagenet",

        input_shape=(299, 299, 3)

    )

    base.trainable = False

    return classification_head(

        base,

        num_classes

    )