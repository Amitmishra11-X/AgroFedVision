"""
============================================================
AgroFedVision Plot Generator
============================================================

Generates

✓ Training Accuracy
✓ Training Loss
✓ Confusion Matrix
✓ ROC Curve
✓ Precision Recall Curve
✓ Benchmark Comparison

============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (

    ConfusionMatrixDisplay,

    RocCurveDisplay,

    PrecisionRecallDisplay,

    roc_curve,

    precision_recall_curve

)

from config import *


class ResearchPlots:

    def __init__(

        self,

        model_name

    ):

        self.model_name = model_name

        self.output = Path(

            PLOT_DIR

        ) / model_name

        self.output.mkdir(

            parents=True,

            exist_ok=True

        )
            # =====================================================
    # Accuracy Plot
    # =====================================================

    def accuracy(

        self,

        history

    ):

        plt.figure(

            figsize=(8,6)

        )

        plt.plot(

            history.history["accuracy"],

            label="Training"

        )

        plt.plot(

            history.history["val_accuracy"],

            label="Validation"

        )

        plt.title(

            "Training Accuracy"

        )

        plt.xlabel(

            "Epoch"

        )

        plt.ylabel(

            "Accuracy"

        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            self.output /

            "accuracy.png",

            dpi=300

        )

        plt.close()
            # =====================================================
    # Loss Plot
    # =====================================================

    def loss(

        self,

        history

    ):

        plt.figure(

            figsize=(8,6)

        )

        plt.plot(

            history.history["loss"],

            label="Training"

        )

        plt.plot(

            history.history["val_loss"],

            label="Validation"

        )

        plt.title(

            "Training Loss"

        )

        plt.xlabel(

            "Epoch"

        )

        plt.ylabel(

            "Loss"

        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            self.output /

            "loss.png",

            dpi=300

        )

        plt.close()