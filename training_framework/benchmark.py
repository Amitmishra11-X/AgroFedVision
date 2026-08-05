"""
============================================================
AgroFedVision Benchmark Manager
============================================================
"""

from pathlib import Path
import pandas as pd

from config import OUTPUT_DIR


class Benchmark:

    def __init__(self):

        self.results = []

    # =====================================================
    # Add Model Result
    # =====================================================

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

    # =====================================================
    # Save Benchmark
    # =====================================================

    def save(self, dataset_name):

        output = Path(OUTPUT_DIR) / dataset_name

        output.mkdir(

            parents=True,

            exist_ok=True

        )

        benchmark_file = output / "benchmark.csv"

        new_df = pd.DataFrame(

            self.results

        )

        # ----------------------------------------
        # Load Existing Benchmark
        # ----------------------------------------

        if benchmark_file.exists():

            old_df = pd.read_csv(

                benchmark_file

            )

            df = pd.concat(

                [

                    old_df,

                    new_df

                ],

                ignore_index=True

            )

            # Keep only latest result of each model
            df = df.drop_duplicates(

                subset="Model",

                keep="last"

            )

        else:

            df = new_df

        # ----------------------------------------
        # Sort by Accuracy
        # ----------------------------------------

        df = df.sort_values(

            by="Accuracy",

            ascending=False

        )

        df.reset_index(

            drop=True,

            inplace=True

        )

        # ----------------------------------------
        # Save
        # ----------------------------------------

        df.to_csv(

            benchmark_file,

            index=False

        )

        print()

        print("=" * 60)
        print("Benchmark Updated")
        print("=" * 60)
        print(benchmark_file)
        print("=" * 60)

        return df