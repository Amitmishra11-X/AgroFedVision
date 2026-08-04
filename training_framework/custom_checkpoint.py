"""
============================================================
AgroFedVision Custom Checkpoint
============================================================
Saves:
    - Latest model weights
    - Best model weights
    - Training state (epoch, best accuracy)
============================================================
"""

import json
from pathlib import Path

import tensorflow as tf

from config import CHECKPOINT_DIR


class SaveTrainingState(tf.keras.callbacks.Callback):

    def __init__(self, model_name):

        super().__init__()

        self.model_name = model_name

        self.folder = Path(CHECKPOINT_DIR)

        self.folder.mkdir(

            parents=True,

            exist_ok=True

        )

        self.latest_weights = self.folder / f"{model_name}.weights.h5"

        self.best_weights = self.folder / f"{model_name}_best.weights.h5"

        self.state_file = self.folder / f"{model_name}_state.json"

        self.best_accuracy = 0.0

    # ----------------------------------------------------
    # Load previous state if available
    # ----------------------------------------------------

    def on_train_begin(self, logs=None):

        if self.state_file.exists():

            try:

                with open(self.state_file, "r") as f:

                    state = json.load(f)

                    self.best_accuracy = state.get(

                        "best_accuracy",

                        0.0

                    )

            except Exception:

                self.best_accuracy = 0.0

    # ----------------------------------------------------
    # Save after every epoch
    # ----------------------------------------------------

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        current_acc = logs.get(

            "val_accuracy",

            0.0

        )

        # Save latest weights

        self.model.save_weights(

            str(self.latest_weights)

        )

        # Save best weights

        if current_acc > self.best_accuracy:

            self.best_accuracy = current_acc

            self.model.save_weights(

                str(self.best_weights)

            )

            print()

            print("=" * 60)

            print("New Best Model Saved")

            print(f"Validation Accuracy : {current_acc:.4f}")

            print("=" * 60)

        # Save training state

        state = {

            "epoch": epoch + 1,

            "best_accuracy": float(self.best_accuracy)

        }

        with open(

            self.state_file,

            "w"

        ) as f:

            json.dump(

                state,

                f,

                indent=4

            )