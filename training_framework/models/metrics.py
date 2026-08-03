"""
============================================================
AgroFedVision Research Metrics
============================================================
Computes all evaluation metrics required for research papers.
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
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
)


class ResearchMetrics:

    def __init__(self, class_names):

        self.class_names = class_names

    def evaluate(

        self,

        y_true,

        y_pred,

        y_prob=None,

        output_dir=None,

    ):

        metrics = {}

        metrics["Accuracy"] = accuracy_score(y_true, y_pred)

        metrics["Precision"] = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        metrics["Recall"] = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        metrics["F1"] = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        metrics["Macro_F1"] = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        metrics["Micro_F1"] = f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )

        metrics["MCC"] = matthews_corrcoef(
            y_true,
            y_pred,
        )

        metrics["Cohen_Kappa"] = cohen_kappa_score(
            y_true,
            y_pred,
        )

        # ROC AUC (only if probabilities available)

        if y_prob is not None:

            try:

                metrics["ROC_AUC"] = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                )

            except Exception:

                metrics["ROC_AUC"] = None

        else:

            metrics["ROC_AUC"] = None

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        report = classification_report(

            y_true,

            y_pred,

            target_names=self.class_names,

            output_dict=True,

            zero_division=0,

        )

        if output_dir is not None:

            output_dir = Path(output_dir)

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            pd.DataFrame(report).transpose().to_csv(

                output_dir /
                "classification_report.csv",

                index=True,

            )

            pd.DataFrame(cm).to_csv(

                output_dir /
                "confusion_matrix.csv",

                index=False,

            )

            with open(

                output_dir /
                "metrics.json",

                "w",

            ) as f:

                json.dump(

                    metrics,

                    f,

                    indent=4,

                )

        return metrics, cm, report