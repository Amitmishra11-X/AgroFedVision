"""Prediction orchestrator that activates pipelines from available inputs."""

from __future__ import annotations

from agrofedvision.core.contracts import AgriculturalReport, ModelPrediction, UserInputBundle
from agrofedvision.fusion.engine import FusionEngine
from agrofedvision.models.registry import ModelRegistry, default_registry


class PredictionOrchestrator:
    def __init__(
        self,
        registry: ModelRegistry | None = None,
        fusion_engine: FusionEngine | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.fusion_engine = fusion_engine or FusionEngine()

    def predict(self, inputs: UserInputBundle) -> AgriculturalReport:
        predictions: list[ModelPrediction] = []
        for spec in self.registry.select_for_inputs(inputs):
            predictor = self.registry.predictor_for(spec.model_id)
            predictions.extend(predictor.predict(inputs))
        return self.fusion_engine.fuse(predictions)
