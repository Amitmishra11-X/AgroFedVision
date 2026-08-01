"""Checkpoint helpers for Flower clients and servers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import keras

from agrofedvision.utils.json_io import read_json, write_json


@dataclass
class FederatedCheckpointManager:
    root_dir: Path
    identity: str

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.root_dir / f"{self.identity}.keras"

    @property
    def state_path(self) -> Path:
        return self.root_dir / f"{self.identity}_state.json"

    def save_model(self, model: keras.Model, state: dict[str, object]) -> None:
        model.save(self.model_path)
        write_json(self.state_path, state)

    def load_model_if_available(self) -> keras.Model | None:
        if self.model_path.exists():
            return keras.models.load_model(self.model_path)
        return None

    def load_state(self) -> dict[str, object]:
        return read_json(self.state_path, {"round": 0, "status": "new"})
