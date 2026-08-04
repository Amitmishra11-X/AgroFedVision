"""
============================================================
AgroFedVision Benchmark Manager
============================================================
"""

from pathlib import Path
import pandas as pd


class Benchmark:

    def __init__(self):

        self.results = []

    def add(

        self,

        model_name,

        metrics,

        training_time

    ):

        row = {

            "Model": model_name,

            "Accuracy": metrics.get("Accuracy"),

            "Precision": metrics.get("Precision"),

            "Recall": metrics.get("Recall"),

            "F1": metrics.get("F1"),

            "Macro_F1": metrics.get("Macro_F1"),

            "Micro_F1": metrics.get("Micro_F1"),

            "MCC": metrics.get("MCC"),

            "Cohen_Kappa": metrics.get("Cohen_Kappa"),

            "ROC_AUC": metrics.get("ROC_AUC"),

            "Training_Time": training_time

        }

        self.results.append(row)

    def save(

        self,

        dataset_name

    ):

        output = Path("outputs") / dataset_name

        output.mkdir(

            parents=True,

            exist_ok=True

        )

        df = pd.DataFrame(

            self.results

        )

        df.sort_values(

            by="Accuracy",

            ascending=False,

            inplace=True

        )

        df.to_csv(

            output / "benchmark.csv",

            index=False

        )

        print()

        print("=" * 60)

        print("Benchmark Saved")

        print("=" * 60)

        print(output / "benchmark.csv")

        print("=" * 60)

        return df