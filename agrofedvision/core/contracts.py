"""Shared prediction and reporting contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Modality(StrEnum):
    IMAGE = "image"
    SENSOR = "sensor"
    UAV = "uav"
    MULTISPECTRAL = "multispectral"
    WEATHER = "weather"
    SATELLITE = "satellite"


class TaskType(StrEnum):
    DISEASE_CLASSIFICATION = "disease_classification"
    SENSOR_INTELLIGENCE = "sensor_intelligence"
    UAV_HEALTH_ANALYSIS = "uav_health_analysis"
    YIELD_PREDICTION = "yield_prediction"
    CROP_RECOMMENDATION = "crop_recommendation"
    FERTILIZER_RECOMMENDATION = "fertilizer_recommendation"
    IRRIGATION_RECOMMENDATION = "irrigation_recommendation"
    HEALTH_SCORING = "health_scoring"


@dataclass(frozen=True)
class UserInputBundle:
    image_paths: list[str] = field(default_factory=list)
    uav_image_paths: list[str] = field(default_factory=list)
    multispectral_paths: list[str] = field(default_factory=list)
    sensor_data: dict[str, float] = field(default_factory=dict)
    weather_data: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def available_modalities(self) -> set[Modality]:
        modalities: set[Modality] = set()
        if self.image_paths:
            modalities.add(Modality.IMAGE)
        if self.uav_image_paths:
            modalities.add(Modality.UAV)
        if self.multispectral_paths:
            modalities.add(Modality.MULTISPECTRAL)
        if self.sensor_data:
            modalities.add(Modality.SENSOR)
        if self.weather_data:
            modalities.add(Modality.WEATHER)
        return modalities


@dataclass
class ModelPrediction:
    modality: Modality
    task: TaskType
    model_id: str
    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    findings: dict[str, Any] = field(default_factory=dict)
    explanation_refs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["modality"] = self.modality.value
        payload["task"] = self.task.value
        return payload


@dataclass
class Recommendation:
    category: str
    priority: str
    title: str
    action: str
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class AgriculturalReport:
    status: str
    health_score: float
    risk_level: str
    confidence: float
    primary_diagnosis: str
    predictions: list[ModelPrediction]
    recommendations: list[Recommendation]
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "health_score": self.health_score,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "primary_diagnosis": self.primary_diagnosis,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "recommendations": [
                recommendation.to_dict() for recommendation in self.recommendations
            ],
            "summary": self.summary,
            "evidence": self.evidence,
        }
