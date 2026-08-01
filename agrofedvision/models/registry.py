"""Model registry for modality-aware prediction pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import keras

from agrofedvision.core.contracts import Modality, ModelPrediction, TaskType, UserInputBundle


class Predictor(Protocol):
    def predict(self, inputs: UserInputBundle) -> list[ModelPrediction]:
        ...


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    modality: Modality
    task: TaskType
    version: str
    artifact_path: Path | None = None
    class_names: list[str] = field(default_factory=list)
    enabled: bool = True
    weight: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._predictors: dict[str, Predictor] = {}

    def register(self, spec: ModelSpec, predictor: Predictor | None = None) -> None:
        if spec.model_id in self._specs:
            raise ValueError(f"Model already registered: {spec.model_id}")
        self._specs[spec.model_id] = spec
        if predictor is not None:
            self._predictors[spec.model_id] = predictor

    def unregister(self, model_id: str) -> None:
        self._specs.pop(model_id, None)
        self._predictors.pop(model_id, None)

    def list_models(
        self,
        modality: Modality | None = None,
        task: TaskType | None = None,
        enabled_only: bool = True,
    ) -> list[ModelSpec]:
        specs = list(self._specs.values())
        if enabled_only:
            specs = [spec for spec in specs if spec.enabled]
        if modality is not None:
            specs = [spec for spec in specs if spec.modality == modality]
        if task is not None:
            specs = [spec for spec in specs if spec.task == task]
        return sorted(specs, key=lambda spec: (spec.modality.value, spec.task.value, spec.model_id))

    def predictor_for(self, model_id: str) -> Predictor:
        if model_id in self._predictors:
            return self._predictors[model_id]
        spec = self._specs.get(model_id)
        if spec is None:
            raise KeyError(f"Unknown model: {model_id}")
        if spec.artifact_path is None:
            raise ValueError(f"Model {model_id} has no artifact path or predictor.")
        model = keras.models.load_model(spec.artifact_path)
        predictor = KerasImageClassifierPredictor(spec, model)
        self._predictors[model_id] = predictor
        return predictor

    def select_for_inputs(self, inputs: UserInputBundle) -> list[ModelSpec]:
        modalities = inputs.available_modalities()
        selected = [
            spec for spec in self.list_models(enabled_only=True) if spec.modality in modalities
        ]
        return sorted(selected, key=lambda spec: spec.weight, reverse=True)


class KerasImageClassifierPredictor:
    def __init__(self, spec: ModelSpec, model: keras.Model) -> None:
        self.spec = spec
        self.model = model

    def predict(self, inputs: UserInputBundle) -> list[ModelPrediction]:
        from agrofedvision.prediction.image_runtime import predict_image_paths

        if not inputs.image_paths:
            return []
        return predict_image_paths(
            model=self.model,
            model_id=self.spec.model_id,
            image_paths=inputs.image_paths,
            class_names=self.spec.class_names,
        )


def default_registry() -> ModelRegistry:
    from agrofedvision.prediction.heuristics import (
        SensorHeuristicPredictor,
        UAVVegetationPredictor,
    )

    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            model_id="sensor_rule_engine_v1",
            modality=Modality.SENSOR,
            task=TaskType.SENSOR_INTELLIGENCE,
            version="1.0.0",
            metadata={"kind": "deterministic_rule_model"},
        ),
        SensorHeuristicPredictor(),
    )
    registry.register(
        ModelSpec(
            model_id="uav_vegetation_engine_v1",
            modality=Modality.UAV,
            task=TaskType.UAV_HEALTH_ANALYSIS,
            version="1.0.0",
            metadata={"kind": "rgb_vegetation_index_model"},
        ),
        UAVVegetationPredictor(),
    )
    return registry
