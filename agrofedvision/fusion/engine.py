"""Decision fusion engine for multimodal agricultural reports."""

from __future__ import annotations

from collections import Counter

from agrofedvision.core.contracts import AgriculturalReport, ModelPrediction
from agrofedvision.recommendations.engine import RecommendationEngine


class FusionEngine:
    def __init__(self, recommendation_engine: RecommendationEngine | None = None) -> None:
        self.recommendation_engine = recommendation_engine or RecommendationEngine()

    def fuse(self, predictions: list[ModelPrediction]) -> AgriculturalReport:
        if not predictions:
            recommendations = self.recommendation_engine.generate([], 50.0)
            return AgriculturalReport(
                status="insufficient_data",
                health_score=50.0,
                risk_level="unknown",
                confidence=0.0,
                primary_diagnosis="No supported input was available.",
                predictions=[],
                recommendations=recommendations,
                summary="Upload at least one supported agricultural data type to generate a report.",
            )

        stress_values = [_prediction_stress(prediction) for prediction in predictions]
        weighted_stress = sum(
            stress * max(0.05, prediction.confidence)
            for stress, prediction in zip(stress_values, predictions, strict=False)
        ) / sum(max(0.05, prediction.confidence) for prediction in predictions)
        health_score = float(round(max(0.0, min(100.0, 100.0 * (1.0 - weighted_stress))), 2))
        confidence = float(round(sum(p.confidence for p in predictions) / len(predictions), 4))
        primary = _primary_diagnosis(predictions)
        risk_level = _risk_level(health_score)
        status = "healthy" if health_score >= 75 and risk_level == "low" else "attention_required"
        recommendations = self.recommendation_engine.generate(predictions, health_score)
        summary = (
            f"Primary diagnosis: {primary}. Health score is {health_score:.1f}/100 "
            f"with {risk_level} risk based on {len(predictions)} model output(s)."
        )
        return AgriculturalReport(
            status=status,
            health_score=health_score,
            risk_level=risk_level,
            confidence=confidence,
            primary_diagnosis=primary,
            predictions=predictions,
            recommendations=recommendations,
            summary=summary,
            evidence={
                "modalities": sorted({prediction.modality.value for prediction in predictions}),
                "tasks": sorted({prediction.task.value for prediction in predictions}),
            },
        )


def _prediction_stress(prediction: ModelPrediction) -> float:
    if "stress_risk" in prediction.scores:
        return float(prediction.scores["stress_risk"])
    if "canopy_stress" in prediction.scores:
        return float(prediction.scores["canopy_stress"])
    label = prediction.label.lower()
    if label in {"healthy", "normal", "sensor_conditions_stable", "uav_canopy_stable"}:
        return 1.0 - prediction.confidence
    return prediction.confidence


def _primary_diagnosis(predictions: list[ModelPrediction]) -> str:
    weighted = Counter()
    for prediction in predictions:
        weighted[prediction.label] += prediction.confidence
    return weighted.most_common(1)[0][0]


def _risk_level(health_score: float) -> str:
    if health_score >= 75:
        return "low"
    if health_score >= 55:
        return "medium"
    return "high"
