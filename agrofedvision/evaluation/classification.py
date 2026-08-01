"""Classification metrics, reports, and plots."""

from __future__ import annotations

import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from agrofedvision.utils.json_io import write_json
from agrofedvision.utils.runtime import fold_name


def save_history_from_csv(csv_paths: list[Path], output_path: Path) -> None:
    frames = []
    for csv_path in csv_paths:
        if csv_path.exists() and csv_path.stat().st_size > 0:
            frame = pd.read_csv(csv_path)
            frame["source_log"] = csv_path.name
            frames.append(frame)
    if not frames:
        write_json(output_path, {})
        return
    history = pd.concat(frames, ignore_index=True)
    write_json(output_path, history.to_dict(orient="list"))


def plot_training_history(history_path: Path, plot_dir: Path, fold_index: int) -> None:
    if not history_path.exists():
        return
    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)

    fold = fold_name(fold_index)
    for metric, ylabel in (("accuracy", "Accuracy"), ("loss", "Loss")):
        plt.figure(figsize=(8, 5))
        plt.plot(history.get(metric, []), label=f"train_{metric}")
        val_metric = f"val_{metric}"
        if val_metric in history:
            plt.plot(history[val_metric], label=val_metric)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{fold} {ylabel}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{fold}_{metric}.png", dpi=160)
        plt.close()


def evaluate_classifier(
    model: keras.Model,
    dataset,
    y_true: np.ndarray,
    class_names: list[str],
    paths: dict[str, Path],
    fold_index: int,
) -> dict[str, float]:
    fold = fold_name(fold_index)
    probabilities = model.predict(dataset, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    (paths["metrics"] / f"{fold}_classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    write_json(paths["metrics"] / f"{fold}_classification_report.json", report_dict)
    write_json(paths["metrics"] / f"{fold}_metrics.json", metrics)
    np.savetxt(
        paths["metrics"] / f"{fold}_confusion_matrix.csv",
        matrix,
        fmt="%d",
        delimiter=",",
    )

    plt.figure(figsize=(9, 8))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"{fold} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(paths["plots"] / f"{fold}_confusion_matrix.png", dpi=180)
    plt.close()
    return metrics
