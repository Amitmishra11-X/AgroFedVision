"""
============================================================
AgroFedVision Research Trainer
============================================================
Universal Research Trainer

Features
--------
✓ Automatic Compile
✓ Automatic Training
✓ Automatic Evaluation
✓ Automatic Saving
✓ Benchmark Ready
✓ TensorBoard
✓ CSV Logger
✓ Future Fine-Tuning
============================================================
"""

import numpy as np
import tensorflow as tf

from pathlib import Path

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

from callbacks import get_callbacks
from metrics import ResearchMetrics
from config import *


class ResearchTrainer:

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

        # Research Metrics Engine
        self.metric_engine = ResearchMetrics(

            self.class_names

        )
# =====================================================
# Compile Model
# =====================================================

    # =====================================================
    # Compile Model
    # =====================================================

    def compile(self):

        """
        Compile the selected model.
        """

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

                f"Unknown Optimizer : {OPTIMIZER}"

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

        print(f"Model       : {self.model_name}")

        print(f"Optimizer   : {OPTIMIZER}")

        print(f"Loss        : {LOSS}")

        print(f"LearningRate: {LEARNING_RATE}")

        print(f"Epochs      : {EPOCHS}")

        print(f"Batch Size  : {BATCH_SIZE}")

        print("=" * 60)
            # =====================================================
    # Train Model
    # =====================================================

    def train(self):

        # ----------------------------------------
        # Compile Model
        # ----------------------------------------

        self.compile()

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

        print(f"Model   : {self.model_name}")

        print(f"Epochs  : {EPOCHS}")

        print(f"Batch   : {BATCH_SIZE}")

        print("=" * 60)

        # ----------------------------------------
        # Train
        # ----------------------------------------

        self.history = self.model.fit(

            self.train_ds,

            validation_data=self.val_ds,

            epochs=EPOCHS,

            callbacks=callbacks,

            verbose=1

        )

        # ----------------------------------------
        # Create Model Folder
        # ----------------------------------------

        model_folder = Path(

            MODEL_DIR

        ) / self.model_name

        model_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        # ----------------------------------------
        # Save Final Model
        # ----------------------------------------

        model_path = model_folder / f"{self.model_name}.keras"

        self.model.save(

            model_path

        )

        print()

        print("=" * 60)

        print("Training Completed")

        print("=" * 60)

        print(f"Model Saved : {model_path}")

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

        # ----------------------------------------
        # Select Dataset
        # ----------------------------------------

        if self.test_ds is not None:

            dataset = self.test_ds

        elif self.val_ds is not None:

            dataset = self.val_ds

        else:

            raise ValueError(

                "No validation or test dataset available."

            )

        # ----------------------------------------
        # Predict
        # ----------------------------------------

        predictions = self.model.predict(

            dataset,

            verbose=1

        )

        y_prob = predictions

        y_pred = np.argmax(

            predictions,

            axis=1

        )

        # ----------------------------------------
        # Ground Truth
        # ----------------------------------------

        y_true = []

        for _, labels in dataset:

            y_true.extend(

                labels.numpy()

            )

        y_true = np.array(

            y_true

        )

        # ----------------------------------------
        # Output Folder
        # ----------------------------------------

        report_folder = Path(

            REPORT_DIR

        ) / self.model_name

        report_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        # ----------------------------------------
        # Research Metrics
        # ----------------------------------------

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

        for k, v in metrics.items():

            if v is None:

                print(f"{k:<20}: N/A")

            else:

                print(f"{k:<20}: {v:.4f}")

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

        model_path = model_folder / f"{self.model_name}.keras"

        self.model.save(

            model_path

        )

        print()

        print("=" * 60)

        print("Model Saved Successfully")

        print("=" * 60)

        print(f"Path : {model_path}")

        print("=" * 60)

        return model_path


    # =====================================================
    # Load Saved Model
    # =====================================================

    @staticmethod

    def load_model(model_path):

        print()

        print("=" * 60)

        print("Loading Saved Model")

        print("=" * 60)

        print(model_path)

        print("=" * 60)

        return tf.keras.models.load_model(

            model_path

        )


    # =====================================================
    # Fine Tune Model (Future Support)
    # =====================================================

    def fine_tune(

        self,

        unfreeze_layers=20,

        learning_rate=1e-5,

        epochs=10

    ):

        print()

        print("=" * 60)

        print("Fine Tuning Started")

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

            callbacks=get_callbacks(

                self.model_name + "_finetune"

            ),

            verbose=1

        )

        fine_tune_folder = Path(

            MODEL_DIR

        ) / self.model_name

        fine_tune_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        fine_tune_path = fine_tune_folder / f"{self.model_name}_finetuned.keras"

        self.model.save(

            fine_tune_path

        )

        print()

        print("=" * 60)

        print("Fine Tuning Completed")

        print("=" * 60)

        print(f"Saved : {fine_tune_path}")

        print("=" * 60)

        return history