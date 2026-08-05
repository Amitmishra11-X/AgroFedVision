"""
============================================================
AgroFedVision
Automatic Benchmark Runner
============================================================
"""

import json
import time
from pathlib import Path

from config import *

from dataset import load_dataset
from models.model_registry import build_model
from trainer import ResearchTrainer
from benchmark import Benchmark

from preprocessing.preprocessing_pipeline import (
    run_preprocessing,
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 70)
    print("AgroFedVision Automatic Benchmark")
    print("=" * 70)

    print(f"Current Dataset : {CURRENT_DATASET}")
    print(f"Models          : {MODELS}")
    print()

    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------

    print("Running preprocessing...\n")

    clean_train = run_preprocessing(
        TRAIN_DIR
    )

    benchmark = Benchmark()

    best_accuracy = -1.0
    best_model = None

    # ------------------------------------------------------
    # Loop Through Models
    # ------------------------------------------------------

    for model_name in MODELS:

        print()
        print("=" * 70)
        print(f"Training : {model_name.upper()}")
        print("=" * 70)

        start = time.time()

        try:

            # ------------------------------------------
            # Build Model
            # ------------------------------------------

            model_info = build_model(

                model_name,

                NUM_CLASSES

            )

            model = model_info["model"]

            image_size = model_info["image_size"]

            preprocess_fn = model_info["preprocess"]

            print()

            print("=" * 60)
            print("Model Built Successfully")
            print("=" * 60)

            print(f"Input Size : {image_size}")

            print()

            # ------------------------------------------
            # Load Train Dataset
            # ------------------------------------------

            train_ds, class_names = load_dataset(

                clean_train,

                image_size=image_size,

                preprocess_fn=preprocess_fn,

                shuffle=True

            )
                        # ------------------------------------------
            # Validation Dataset
            # ------------------------------------------

            val_ds = None

            if VAL_DIR is not None:

                val_ds, _ = load_dataset(

                    VAL_DIR,

                    image_size=image_size,

                    preprocess_fn=preprocess_fn,

                    shuffle=False

                )

            else:

                clean_val = Path(clean_train).parent / "valid"

                if clean_val.exists():

                    val_ds, _ = load_dataset(

                        str(clean_val),

                        image_size=image_size,

                        preprocess_fn=preprocess_fn,

                        shuffle=False

                    )

            # ------------------------------------------
            # Test Dataset
            # ------------------------------------------

            test_ds = None

            if TEST_DIR is not None:

                test_ds, _ = load_dataset(

                    TEST_DIR,

                    image_size=image_size,

                    preprocess_fn=preprocess_fn,

                    shuffle=False

                )

            else:

                clean_test = Path(clean_train).parent / "test"

                if clean_test.exists():

                    test_ds, _ = load_dataset(

                        str(clean_test),

                        image_size=image_size,

                        preprocess_fn=preprocess_fn,

                        shuffle=False

                    )

            # ------------------------------------------
            # Display Dataset Information
            # ------------------------------------------

            print()

            print("=" * 60)
            print("Dataset Loaded")
            print("=" * 60)

            print(f"Training Path   : {clean_train}")

            if val_ds is not None:
                print("Validation Set  : Available")
            else:
                print("Validation Set  : Not Available")

            if test_ds is not None:
                print("Test Set        : Available")
            else:
                print("Test Set        : Not Available")

            print()

            print("=" * 60)
            print("Classes")
            print("=" * 60)

            for idx, cls in enumerate(class_names):

                print(f"{idx:2d} -> {cls}")

            print("=" * 60)

            # ------------------------------------------
            # Create Trainer
            # ------------------------------------------

            trainer = ResearchTrainer(

                model=model,

                train_ds=train_ds,

                val_ds=val_ds,

                test_ds=test_ds,

                class_names=class_names,

                model_name=model_name

            )
                        # ------------------------------------------
            # Train Model
            # ------------------------------------------

            history = trainer.train()

            # ------------------------------------------
            # Evaluate Model
            # ------------------------------------------

            metrics = trainer.evaluate()

            # ------------------------------------------
            # Save Final Model
            # ------------------------------------------

            trainer.save()

            # ------------------------------------------
            # Save Training History
            # ------------------------------------------

            history_folder = Path(HISTORY_DIR)

            history_folder.mkdir(

                parents=True,

                exist_ok=True

            )

            history_file = history_folder / f"{model_name}.json"

            with open(

                history_file,

                "w"

            ) as f:

                json.dump(

                    history.history,

                    f,

                    indent=4

                )

            # ------------------------------------------
            # Training Time
            # ------------------------------------------

            training_time = time.time() - start

            print()

            print("=" * 60)
            print("Training Summary")
            print("=" * 60)

            print(f"Model          : {model_name}")

            print(f"Accuracy       : {metrics['Accuracy']:.4f}")

            print(f"Training Time  : {training_time/60:.2f} Minutes")

            print("=" * 60)

            # ------------------------------------------
            # Benchmark
            # ------------------------------------------

            benchmark.add(

                model_name=model_name,

                metrics=metrics,

                training_time=training_time

            )

            # Save benchmark after every model

            benchmark.save(

                CURRENT_DATASET

            )

            # ------------------------------------------
            # Best Model
            # ------------------------------------------

            if metrics["Accuracy"] > best_accuracy:

                best_accuracy = metrics["Accuracy"]

                best_model = {

                    "model": model_name,

                    "accuracy": float(best_accuracy)

                }

                best_model_dir = Path(

                    BENCHMARK_DIR

                )

                best_model_dir.mkdir(

                    parents=True,

                    exist_ok=True

                )

                trainer.model.save(

                    best_model_dir /

                    f"best_{model_name}.keras"

                )

                print()

                print("=" * 60)
                print("New Best Model")
                print("=" * 60)

                print(best_model)

        # ------------------------------------------
        # Continue if One Model Fails
        # ------------------------------------------

        except Exception as e:

            print()

            print("=" * 70)
            print(f"{model_name.upper()} FAILED")
            print("=" * 70)

            print(e)

            continue
            # ======================================================
    # Save Final Benchmark
    # ======================================================

    print()

    print("=" * 70)
    print("Saving Benchmark Results")
    print("=" * 70)

    benchmark_df = benchmark.save(

        CURRENT_DATASET

    )

    # ======================================================
    # Save Best Model Information
    # ======================================================

    benchmark_dir = Path(

        BENCHMARK_DIR

    )

    benchmark_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    best_json = benchmark_dir / "best_model.json"

    with open(

        best_json,

        "w"

    ) as f:

        json.dump(

            best_model,

            f,

            indent=4

        )

    # ======================================================
    # Display Leaderboard
    # ======================================================

    print()

    print("=" * 70)
    print("FINAL BENCHMARK LEADERBOARD")
    print("=" * 70)

    print(benchmark_df)

    print()

    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    if best_model is not None:

        print(f"Model    : {best_model['model']}")
        print(f"Accuracy : {best_model['accuracy']:.4f}")

    else:

        print("No successful model training.")

    print()

    print("=" * 70)
    print("Benchmark Completed Successfully")
    print("=" * 70)


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    main()