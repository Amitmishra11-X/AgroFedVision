"""Generic image-classification manifest support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agrofedvision.training.config import Phase1Config
from agrofedvision.utils.json_io import write_json


@dataclass(frozen=True)
class ImageClassificationManifest:
    frame: pd.DataFrame
    class_names: list[str]

    @property
    def image_paths(self) -> np.ndarray:
        return self.frame["image_path"].to_numpy(dtype=str)

    @property
    def labels(self) -> np.ndarray:
        return self.frame["label"].to_numpy(dtype=np.int32)


def _from_kaggle_style(df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": df.apply(
                lambda row: str(image_root / str(row["label"]) / str(row["image_id"])),
                axis=1,
            ),
            "label_name": df["label"].astype(str),
        }
    )


def _from_path_label_style(df: pd.DataFrame) -> pd.DataFrame:
    label_column = "crop_health" if "crop_health" in df.columns else "label"
    return pd.DataFrame(
        {
            "image_path": df["image_path"].astype(str),
            "label_name": df[label_column].astype(str),
        }
    )


def load_image_classification_manifest(config: Phase1Config) -> ImageClassificationManifest:
    csv_path = config.resolved_train_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {csv_path}. Provide --data-root, "
            "--train-csv, or AGROFED_TRAIN_CSV."
        )

    raw = pd.read_csv(csv_path)
    if {"image_id", "label"}.issubset(raw.columns):
        frame = _from_kaggle_style(raw, config.resolved_image_root())
    elif "image_path" in raw.columns and ({"crop_health"} & set(raw.columns) or "label" in raw.columns):
        frame = _from_path_label_style(raw)
    else:
        raise ValueError(
            "Image manifest must contain image_id,label or image_path,label columns."
        )

    class_names = sorted(frame["label_name"].unique().tolist())
    if len(class_names) != config.num_classes:
        raise ValueError(
            f"Expected {config.num_classes} classes, found {len(class_names)}: {class_names}"
        )

    class_to_index = {name: index for index, name in enumerate(class_names)}
    frame["label"] = frame["label_name"].map(class_to_index).astype("int32")
    frame = frame[["image_path", "label", "label_name"]].reset_index(drop=True)
    return ImageClassificationManifest(frame=frame, class_names=class_names)


def save_manifest_summary(
    manifest: ImageClassificationManifest,
    results_dir: Path,
) -> None:
    counts = manifest.frame["label_name"].value_counts().sort_index().to_dict()
    write_json(
        results_dir / "summary" / "image_manifest_summary.json",
        {
            "num_examples": int(len(manifest.frame)),
            "num_classes": int(len(manifest.class_names)),
            "class_names": manifest.class_names,
            "class_counts": {str(key): int(value) for key, value in counts.items()},
        },
    )
