"""Configuration objects for Phase 1 centralized training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StageConfig:
    name: str
    epochs: int
    learning_rate: float


@dataclass(frozen=True)
class Phase1Config:
    project_name: str = "AgroFedVision"
    seed: int = 42
    image_size: tuple[int, int] = (260, 260)
    input_shape: tuple[int, int, int] = (260, 260, 3)
    num_classes: int = 10
    batch_size: int = 32
    n_splits: int = 5
    cache_datasets: bool = True
    use_mixed_precision: bool = True

    data_root: Path = Path(
        os.environ.get("AGROFED_DATA_ROOT", "/content/agrofedvision-dataset")
    )
    train_csv: Path | None = (
        Path(os.environ["AGROFED_TRAIN_CSV"])
        if os.environ.get("AGROFED_TRAIN_CSV")
        else None
    )
    image_root: Path | None = (
        Path(os.environ["AGROFED_IMAGE_ROOT"])
        if os.environ.get("AGROFED_IMAGE_ROOT")
        else None
    )
    results_dir: Path = Path(os.environ.get("AGROFED_RESULTS_DIR", "results"))

    dense_units: tuple[int, int] = (256, 128)
    dropout_rates: tuple[float, float, float] = (0.40, 0.35, 0.30)
    fine_tune_prefixes: tuple[str, ...] = ("block6", "block7", "top_conv")

    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.3
    min_lr: float = 1e-7

    stage1: StageConfig = StageConfig("stage1", epochs=25, learning_rate=3e-4)
    stage2: StageConfig = StageConfig("stage2", epochs=25, learning_rate=1e-5)
    supported_modalities: tuple[str, ...] = field(
        default=("image", "sensor", "uav", "weather", "multispectral")
    )

    @property
    def stages(self) -> tuple[StageConfig, StageConfig]:
        return (self.stage1, self.stage2)

    def resolved_train_csv(self) -> Path:
        if self.train_csv is not None:
            return self.train_csv
        return self.data_root / "train.csv"

    def resolved_image_root(self) -> Path:
        if self.image_root is not None:
            return self.image_root
        return self.data_root / "train_images"


DEFAULT_PHASE1_CONFIG = Phase1Config()
