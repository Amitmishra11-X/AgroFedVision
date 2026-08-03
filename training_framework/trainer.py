"""
============================================================
AgroFedVision Research Trainer
============================================================

Features
--------
✓ Universal Trainer
✓ Transfer Learning
✓ Automatic Compilation
✓ Callbacks
✓ Evaluation
✓ Future Fine-Tuning Support
✓ TensorBoard
✓ CSV Logger

============================================================
"""

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from metrics import ResearchMetrics

from callbacks import get_callbacks

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

    # =====================================================
    # Compile
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

                f"Unknown Optimizer {OPTIMIZER}"

            )

        self.model.compile(

            optimizer=optimizer,

            loss=LOSS,

            metrics=[

                "accuracy"

            ]

        )

        print("\n")

        print("="*60)

        print("Model Compiled")

        print("="*60)

        print("Optimizer :", OPTIMIZER)

        print("Loss      :", LOSS)

        print("LR        :", LEARNING_RATE)

        print("="*60)


    # =====================================================
    # Train
    # =====================================================

    def train(self):

        callbacks = get_callbacks(

            self.model_name

        )

        print("\n")

        print("="*60)

        print("Training Started")

        print("="*60)

        self.history = self.model.fit(

            self.train_ds,

            validation_data=self.val_ds,

            epochs=EPOCHS,

            callbacks=callbacks,

            verbose=1

        )

        return self.history


    # =====================================================
    # Evaluation
    # =====================================================
def evaluate(self):

    print("\n")

    print("="*60)

    print("Research Evaluation")

    print("="*60)

    evaluator = ResearchMetrics()

    metrics, report, cm = evaluator.evaluate(

        self.model,

        self.test_ds,

        self.class_names

    )

    print()

    for k, v in metrics.items():

        print(f"{k:20}: {v:.4f}")

    return metrics
    # =====================================================
    # Save Final Model
    # =====================================================

    def save(self):

        path = f"outputs/{self.model_name}_final.keras"

        self.model.save(path)

        print("\nSaved Model :", path)