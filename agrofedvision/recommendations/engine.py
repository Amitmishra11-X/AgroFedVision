"""Agronomic recommendation engine."""

from __future__ import annotations

from agrofedvision.core.contracts import ModelPrediction, Recommendation


class RecommendationEngine:
    def generate(self, predictions: list[ModelPrediction], health_score: float) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        labels = " ".join(prediction.label.lower() for prediction in predictions)
        findings = _merged_findings(predictions)

        if "disease" in labels or any(
            label not in {"healthy", "sensor_conditions_stable", "uav_canopy_stable"}
            for label in labels.split()
        ):
            recommendations.append(
                Recommendation(
                    category="disease_management",
                    priority="high" if health_score < 60 else "medium",
                    title="Scout affected plants and isolate severe symptoms",
                    action="Inspect representative plants, remove heavily infected leaves, and apply crop-approved treatment only after field confirmation.",
                    rationale="The model outputs indicate disease or crop stress signals that need field validation.",
                )
            )

        if findings.get("moisture_status") == "low":
            recommendations.append(
                Recommendation(
                    category="irrigation",
                    priority="high",
                    title="Increase irrigation scheduling",
                    action="Apply irrigation in shorter, controlled cycles and recheck soil moisture after infiltration.",
                    rationale="Sensor data indicates low soil moisture.",
                )
            )
        elif findings.get("moisture_status") == "high":
            recommendations.append(
                Recommendation(
                    category="irrigation",
                    priority="medium",
                    title="Reduce irrigation and improve drainage",
                    action="Delay irrigation and inspect drainage in low-lying field sections.",
                    rationale="High moisture can increase disease pressure and root stress.",
                )
            )

        nutrient_actions = []
        for nutrient in ("nitrogen", "phosphorus", "potassium"):
            if findings.get(f"{nutrient}_status") == "low":
                nutrient_actions.append(nutrient)
        if nutrient_actions:
            recommendations.append(
                Recommendation(
                    category="fertilizer",
                    priority="medium",
                    title="Correct nutrient deficiency",
                    action=f"Plan a soil-test-guided amendment for {', '.join(nutrient_actions)}.",
                    rationale="Sensor values suggest one or more nutrient levels are below expected agronomic ranges.",
                )
            )

        if any(prediction.scores.get("canopy_stress", 0.0) > 0.45 for prediction in predictions):
            recommendations.append(
                Recommendation(
                    category="yield_risk",
                    priority="medium",
                    title="Map and monitor stressed field zones",
                    action="Create a field-zone inspection route and compare UAV stress areas with sensor and leaf observations.",
                    rationale="UAV vegetation analysis indicates spatial canopy stress.",
                )
            )

        if not recommendations:
            recommendations.append(
                Recommendation(
                    category="monitoring",
                    priority="low",
                    title="Continue routine monitoring",
                    action="Keep collecting leaf images, sensor readings, and UAV snapshots on a consistent schedule.",
                    rationale="Available model outputs do not indicate urgent stress.",
                )
            )
        return recommendations


def _merged_findings(predictions: list[ModelPrediction]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for prediction in predictions:
        merged.update(prediction.findings)
    return merged
