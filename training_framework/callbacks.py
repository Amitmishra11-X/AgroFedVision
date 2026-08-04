"""
============================================================
AgroFedVision Research Callbacks
============================================================
Uses Custom Checkpoint
============================================================
"""

from pathlib import Path

from tensorflow.keras.callbacks import (

    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard

)

from config import *

from custom_checkpoint import SaveTrainingState


def get_callbacks(model_name):

    Path(LOG_DIR).mkdir(

        parents=True,

        exist_ok=True

    )

    Path(HISTORY_DIR).mkdir(

        parents=True,

        exist_ok=True

    )

    callbacks = []

    # =====================================================
    # Custom Checkpoint
    # =====================================================

    callbacks.append(

        SaveTrainingState(

            model_name

        )

    )

    # =====================================================
    # Early Stopping
    # =====================================================

    callbacks.append(

        EarlyStopping(

            monitor="val_loss",

            patience=PATIENCE,

            restore_best_weights=True,

            verbose=1

        )

    )

    # =====================================================
    # Reduce LR
    # =====================================================

    callbacks.append(

        ReduceLROnPlateau(

            monitor="val_loss",

            factor=REDUCE_FACTOR,

            patience=REDUCE_PATIENCE,

            verbose=1

        )

    )

    # =====================================================
    # CSV Logger
    # =====================================================

    callbacks.append(

        CSVLogger(

            str(

                Path(HISTORY_DIR) /

                f"{model_name}.csv"

            ),

            append=True

        )

    )

    # =====================================================
    # TensorBoard
    # =====================================================

    callbacks.append(

        TensorBoard(

            log_dir=str(

                Path(LOG_DIR) /

                model_name

            ),

            histogram_freq=1

        )

    )

    return callbacks