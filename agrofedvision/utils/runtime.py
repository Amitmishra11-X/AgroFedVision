"""Runtime, reproducibility, and output-directory helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


RESULT_SUBDIRS = (
    "models",
    "checkpoints",
    "backup",
    "history",
    "logs",
    "metrics",
    "plots",
    "summary",
)


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def configure_mixed_precision(enabled: bool) -> None:
    policy = "mixed_float16" if enabled else "float32"
    tf.keras.mixed_precision.set_global_policy(policy)


def ensure_results_tree(results_dir: Path) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {"root": results_dir}
    for subdir in RESULT_SUBDIRS:
        path = results_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = path
    return paths


def fold_name(fold_index: int) -> str:
    return f"fold_{fold_index + 1}"
