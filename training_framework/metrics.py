"""
============================================================
AgroFedVision Research Metrics
============================================================
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
)


class ResearchMetrics:

    def __init__(self):

        Path("outputs/reports").mkdir(
            parents=True,
            exist_ok=True
        )

    def evaluate(

        self,

        model,

        test_ds,

        class_names

    ):

        y_true = []

        y_pred = []

        for images, labels in test_ds:

            predictions = model.predict(
                images,
                verbose=0
            )

            y_pred.extend(
                np.argmax(predictions, axis=1)
            )

            # Handle both one-hot and integer labels
            if len(labels.shape) > 1:
                y_true.extend(
                    np.argmax(labels.numpy(), axis=1)
                )
            else:
                y_true.extend(
                    labels.numpy()
                )

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = {

            "Accuracy": accuracy_score(y_true, y_pred),

            "Precision": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "F1": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "Macro_F1": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

            "Micro_F1": f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

            "Cohen_Kappa": cohen_kappa_score(
                y_true,
                y_pred
            ),

            "MCC": matthews_corrcoef(
                y_true,
                y_pred
            )

        }

        report = classification_report(

            y_true,

            y_pred,

            target_names=class_names,

            output_dict=True,

            zero_division=0

        )

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        pd.DataFrame(report).transpose().to_csv(
            "outputs/reports/classification_report.csv"
        )

        pd.DataFrame(cm).to_csv(
            "outputs/reports/confusion_matrix.csv",
            index=False
        )

        with open(
            "outputs/reports/metrics.json",
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

        return metrics, report, cm