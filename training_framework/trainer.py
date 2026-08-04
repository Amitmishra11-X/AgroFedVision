"""
============================================================
AgroFedVision Research Trainer
============================================================
Universal Research Trainer

Features
--------
✓ Automatic Compile
✓ Resume Training
✓ Custom Checkpoint
✓ Automatic Evaluation
✓ Automatic Saving
✓ Fine-Tuning
============================================================
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

from callbacks import get_callbacks
from metrics import ResearchMetrics
from config import *


class ResearchTrainer:

    # =====================================================
    # Constructor
    # =====================================================

    def __init__(

        self,

        model,

        train_ds,

        val_ds,

        test_ds,

        class_names,

        model_name

    ):

        self.model = model

        self.train_ds = train_ds

        self.val_ds = val_ds

        self.test_ds = test_ds

        self.class_names = class_names

        self.model_name = model_name

        self.history = None

        self.metric_engine = ResearchMetrics(

            self.class_names

        )

    # =====================================================
    # Resume Training
    # =====================================================

    def resume_training(self):

        if not RESUME_TRAINING:

            return 0

        checkpoint = Path(

            CHECKPOINT_DIR

        ) / f"{self.model_name}.weights.h5"

        state_file = Path(

            CHECKPOINT_DIR

        ) / f"{self.model_name}_state.json"

        initial_epoch = 0

        if checkpoint.exists():

            print()

            print("=" * 60)
            print("Resume Training")
            print("=" * 60)
            print(checkpoint)
            print("=" * 60)

            self.model.load_weights(

                str(checkpoint)

            )

            print("Weights Loaded Successfully")

            if state_file.exists():

                with open(

                    state_file,

                    "r"

                ) as f:

                    state = json.load(f)

                initial_epoch = state.get(

                    "epoch",

                    0

                )

                print()

                print(f"Last Completed Epoch : {initial_epoch}")
                print(f"Resume From Epoch    : {initial_epoch + 1}")

        else:

            print()

            print("=" * 60)
            print("No Previous Checkpoint Found")
            print("Starting Fresh Training")
            print("=" * 60)

        return initial_epoch
        # =====================================================
    # Compile Model
    # =====================================================

    def compile(self):

        if OPTIMIZER.lower() == "adam":

            optimizer = Adam(

                learning_rate=LEARNING_RATE

            )

        elif OPTIMIZER.lower() == "sgd":

            optimizer = SGD(

                learning_rate=LEARNING_RATE,

                momentum=0.9

            )

        else:

            raise ValueError(

                f"Unsupported Optimizer : {OPTIMIZER}"

            )

        self.model.compile(

            optimizer=optimizer,

            loss=LOSS,

            metrics=METRICS

        )

        print()

        print("=" * 60)
        print("Model Compiled")
        print("=" * 60)
        print(f"Model         : {self.model_name}")
        print(f"Optimizer     : {OPTIMIZER}")
        print(f"Learning Rate : {LEARNING_RATE}")
        print(f"Epochs        : {EPOCHS}")
        print("=" * 60)


    # =====================================================
    # Train Model
    # =====================================================

    def train(self):

        # ----------------------------------------
        # Compile
        # ----------------------------------------

        self.compile()

        # ----------------------------------------
        # Resume
        # ----------------------------------------

        initial_epoch = self.resume_training()

        # ----------------------------------------
        # Callbacks
        # ----------------------------------------

        callbacks = get_callbacks(

            self.model_name

        )

        print()

        print("=" * 60)
        print("Training Started")
        print("=" * 60)
        print(f"Model         : {self.model_name}")
        print(f"Start Epoch   : {initial_epoch}")
        print(f"End Epoch     : {EPOCHS}")
        print("=" * 60)

        # ----------------------------------------
        # Train
        # ----------------------------------------

        self.history = self.model.fit(

            self.train_ds,

            validation_data=self.val_ds,

            epochs=EPOCHS,

            initial_epoch=initial_epoch,

            callbacks=callbacks,

            verbose=1

        )

        # ----------------------------------------
        # Save Final Weights
        # ----------------------------------------

        model_folder = Path(MODEL_DIR) / self.model_name

        model_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        final_weights = model_folder / "final.weights.h5"

        self.model.save_weights(

            str(final_weights)

        )

        print()

        print("=" * 60)
        print("Training Completed")
        print("=" * 60)
        print(f"Weights Saved : {final_weights}")
        print("=" * 60)

        return self.history
        # =====================================================
    # Evaluate Model
    # =====================================================

    def evaluate(self):

        print()

        print("=" * 60)
        print("Research Evaluation")
        print("=" * 60)

        if self.test_ds is not None:

            dataset = self.test_ds

        elif self.val_ds is not None:

            dataset = self.val_ds

        else:

            raise ValueError(

                "No validation/test dataset found."

            )

        predictions = self.model.predict(

            dataset,

            verbose=1

        )

        y_prob = predictions

        y_pred = np.argmax(

            predictions,

            axis=1

        )

        y_true = []

        for _, labels in dataset:

            y_true.extend(

                labels.numpy()

            )

        y_true = np.array(

            y_true

        )

        report_folder = Path(

            REPORT_DIR

        ) / self.model_name

        report_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        metrics, cm, report = self.metric_engine.evaluate(

            y_true=y_true,

            y_pred=y_pred,

            y_prob=y_prob,

            output_dir=report_folder

        )

        print()

        print("=" * 60)
        print("Evaluation Results")
        print("=" * 60)

        for key, value in metrics.items():

            if value is None:

                print(f"{key:<20}: N/A")

            else:

                print(f"{key:<20}: {value:.4f}")

        print("=" * 60)

        return metrics


    # =====================================================
    # Save Final Model
    # =====================================================

    def save(self):

        model_folder = Path(

            MODEL_DIR

        ) / self.model_name

        model_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        final_weights = model_folder / "final.weights.h5"

        self.model.save_weights(

            str(final_weights)

        )

        print()

        print("=" * 60)
        print("Final Weights Saved")
        print("=" * 60)
        print(final_weights)
        print("=" * 60)

        return final_weights


    # =====================================================
    # Load Saved Weights
    # =====================================================

    def load_model(self, weight_path):

        self.model.load_weights(

            str(weight_path)

        )

        print()

        print("=" * 60)
        print("Weights Loaded")
        print("=" * 60)
        print(weight_path)
        print("=" * 60)

        return self.model


    # =====================================================
    # Fine Tune
    # =====================================================

    def fine_tune(

        self,

        unfreeze_layers=20,

        learning_rate=1e-5,

        epochs=10

    ):

        print()

        print("=" * 60)
        print("Fine Tuning")
        print("=" * 60)

        base_model = self.model.layers[0]

        base_model.trainable = True

        for layer in base_model.layers[:-unfreeze_layers]:

            layer.trainable = False

        self.model.compile(

            optimizer=Adam(

                learning_rate=learning_rate

            ),

            loss=LOSS,

            metrics=METRICS

        )

        history = self.model.fit(

            self.train_ds,

            validation_data=self.val_ds,

            epochs=epochs,

            verbose=1

        )

        fine_folder = Path(

            MODEL_DIR

        ) / self.model_name

        fine_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        fine_weights = fine_folder / "finetuned.weights.h5"

        self.model.save_weights(

            str(fine_weights)

        )

        print()

        print("=" * 60)
        print("Fine Tuning Completed")
        print("=" * 60)
        print(fine_weights)
        print("=" * 60)

        return history