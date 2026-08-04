"""
============================================================
AgroFedVision
Automatic Benchmark Runner
============================================================
"""

import time
import json
from pathlib import Path

from config import *

from dataset import load_dataset

from models.model_registry import build_model

from trainer import ResearchTrainer

from benchmark import Benchmark

from preprocessing.preprocessing_pipeline import run_preprocessing


def main():

    print("\n")
    print("="*70)
    print("AgroFedVision Automatic Benchmark")
    print("="*70)

    print(f"Dataset : {CURRENT_DATASET}")
    print()

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    print("Running preprocessing...")

    clean_train = run_preprocessing(TRAIN_DIR)

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    print("\nLoading Dataset...")

    train_ds, class_names = load_dataset(clean_train)

    # --------------------------------------------------
    # Validation Dataset
    # --------------------------------------------------

    clean_val = Path(clean_train).parent / "valid"

    if clean_val.exists():

        val_ds, _ = load_dataset(

            str(clean_val),

            shuffle=False

        )

    else:

        val_ds = None


    # --------------------------------------------------
    # Test Dataset
    # --------------------------------------------------

    clean_test = Path(clean_train).parent / "test"

    if clean_test.exists():

        test_ds, _ = load_dataset(

            str(clean_test),

            shuffle=False

        )

    else:

        test_ds = None


    print()

    print("Classes")

    for i, c in enumerate(class_names):

        print(i, "->", c)

    print()

    print("Classes")

    for i, c in enumerate(class_names):

        print(i, "->", c)

    print()

    benchmark = Benchmark()

    best_accuracy = 0

    best_model = None

    # --------------------------------------------------
    # Train Every Model
    # --------------------------------------------------

    for model_name in MODELS:

        print("\n")
        print("="*70)
        print(f"Training : {model_name.upper()}")
        print("="*70)

        start = time.time()

        model = build_model(

            model_name,

            len(class_names)

        )

        trainer = ResearchTrainer(

            model=model,

            train_ds=train_ds,

            val_ds=val_ds,

            test_ds=test_ds,

            model_name=model_name,

            class_names=class_names

        )
        
        history = trainer.train()

        metrics = trainer.evaluate()

        end = time.time()

        training_time = end-start

        benchmark.add(

            model_name,

            metrics,

            training_time

        )

        if metrics["Accuracy"] > best_accuracy:

            best_accuracy = metrics["Accuracy"]

            best_model = {

                "model": model_name,

                "accuracy": best_accuracy

            }

    # --------------------------------------------------
    # Save Benchmark
    # --------------------------------------------------

    df = benchmark.save(

        CURRENT_DATASET

    )

    print()

    print(df)

    # --------------------------------------------------
    # Save Best Model
    # --------------------------------------------------

    Path(BENCHMARK_DIR).mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        Path(BENCHMARK_DIR) / "best_model.json",

        "w"

    ) as f:

        json.dump(

            best_model,

            f,

            indent=4

        )

    print()

    print("="*70)

    print("Benchmark Finished")

    print("="*70)

    print()

    print("Best Model")

    print(best_model)


if __name__ == "__main__":

    main()