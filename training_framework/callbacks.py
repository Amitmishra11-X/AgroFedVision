"""
Research Training Callbacks
AgroFedVision
"""

from pathlib import Path

from tensorflow.keras.callbacks import (

    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard

)

from config import *


def get_callbacks(model_name):

    Path("checkpoints").mkdir(exist_ok=True)

    Path("logs").mkdir(exist_ok=True)

    Path("outputs/history").mkdir(parents=True, exist_ok=True)

    callbacks = []

    # ----------------------------------------
    # Early Stopping
    # ----------------------------------------

    callbacks.append(

        EarlyStopping(

            monitor="val_loss",

            patience=PATIENCE,

            restore_best_weights=True,

            verbose=1

        )

    )

    # ----------------------------------------
    # Save Best Model
    # ----------------------------------------

    callbacks.append(

        ModelCheckpoint(

            filepath=f"checkpoints/{model_name}_best.keras",

            monitor="val_accuracy",

            save_best_only=True,

            verbose=1

        )

    )

    # ----------------------------------------
    # Learning Rate Scheduler
    # ----------------------------------------

    callbacks.append(

        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=3,

            verbose=1

        )

    )

    # ----------------------------------------
    # CSV Logger
    # ----------------------------------------

    callbacks.append(

        CSVLogger(

            f"outputs/history/{model_name}.csv"

        )

    )

    # ----------------------------------------
    # TensorBoard
    # ----------------------------------------

    callbacks.append(

        TensorBoard(

            log_dir=f"logs/{model_name}",

            histogram_freq=1

        )

    )

    return callbacks