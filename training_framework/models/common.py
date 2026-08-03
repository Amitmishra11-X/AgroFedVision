"""
Common model utilities
AgroFedVision Research Framework
"""

from tensorflow.keras import layers
from tensorflow.keras import models


def classification_head(
    base_model,
    num_classes,
    dropout=0.30,
    dense_units=512
):
    """
    Shared classification head used by all transfer learning models.
    """

    model = models.Sequential([

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dropout(dropout),

        layers.Dense(
            dense_units,
            activation="relu"
        ),

        layers.Dropout(dropout),

        layers.Dense(
            num_classes,
            activation="softmax"
        )

    ])

    return model