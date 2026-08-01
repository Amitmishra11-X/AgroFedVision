"""Deterministic baseline predictors for non-image modalities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from agrofedvision.core.contracts import Modality, ModelPrediction, TaskType, UserInputBundle


class SensorHeuristicPredictor:
    def predict(self, inputs: UserInputBundle) -> list[ModelPrediction]:
        if not inputs.sensor_data:
            return []

        data = {key.lower(): float(value) for key, value in inputs.sensor_data.items()}
        findings: dict[str, str | float] = {}
        risk = 0.0

        moisture = _first_present(data, ("soil_moisture", "moisture", "vwc"))
        if moisture is not None:
            if moisture < 25:
                findings["moisture_status"] = "low"
                risk += 0.25
            elif moisture > 75:
                findings["moisture_status"] = "high"
                risk += 0.18
            else:
                findings["moisture_status"] = "normal"

        nitrogen = _first_present(data, ("nitrogen", "n"))
        phosphorus = _first_present(data, ("phosphorus", "p"))
        potassium = _first_present(data, ("potassium", "k"))
        for name, value, threshold in (
            ("nitrogen", nitrogen, 35),
            ("phosphorus", phosphorus, 20),
            ("potassium", potassium, 25),
        ):
            if value is not None and value < threshold:
                findings[f"{name}_status"] = "low"
                risk += 0.15

        ph = _first_present(data, ("ph", "soil_ph"))
        if ph is not None:
            if ph < 5.5:
                findings["ph_status"] = "acidic"
                risk += 0.12
            elif ph > 8.0:
                findings["ph_status"] = "alkaline"
                risk += 0.12
            else:
                findings["ph_status"] = "normal"

        confidence = float(min(0.95, 0.55 + len(findings) * 0.06))
        risk = float(min(1.0, risk))
        label = "sensor_stress_detected" if risk >= 0.3 else "sensor_conditions_stable"
        return [
            ModelPrediction(
                modality=Modality.SENSOR,
                task=TaskType.SENSOR_INTELLIGENCE,
                model_id="sensor_rule_engine_v1",
                label=label,
                confidence=confidence,
                scores={"stress_risk": risk, "stability": 1.0 - risk},
                findings=findings,
            )
        ]


class UAVVegetationPredictor:
    def predict(self, inputs: UserInputBundle) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        for image_path in inputs.uav_image_paths:
            image = cv2.imread(str(Path(image_path)), cv2.IMREAD_COLOR)
            if image is None:
                predictions.append(
                    ModelPrediction(
                        modality=Modality.UAV,
                        task=TaskType.UAV_HEALTH_ANALYSIS,
                        model_id="uav_vegetation_engine_v1",
                        label="uav_image_unreadable",
                        confidence=0.0,
                        findings={"image_path": image_path},
                    )
                )
                continue
            b, g, r = cv2.split(image.astype(np.float32) + 1.0)
            exg = 2.0 * g - r - b
            gli = (2.0 * g - r - b) / (2.0 * g + r + b)
            green_fraction = float(np.mean(exg > 20.0))
            gli_mean = float(np.clip(np.nanmean(gli), -1.0, 1.0))
            stress = float(np.clip(0.75 - green_fraction + max(0.0, 0.15 - gli_mean), 0.0, 1.0))
            label = "uav_crop_stress" if stress >= 0.45 else "uav_canopy_stable"
            predictions.append(
                ModelPrediction(
                    modality=Modality.UAV,
                    task=TaskType.UAV_HEALTH_ANALYSIS,
                    model_id="uav_vegetation_engine_v1",
                    label=label,
                    confidence=float(max(0.55, min(0.95, 1.0 - abs(0.5 - stress)))),
                    scores={"canopy_stress": stress, "green_fraction": green_fraction},
                    findings={
                        "image_path": image_path,
                        "green_fraction": green_fraction,
                        "gli_mean": gli_mean,
                    },
                )
            )
        return predictions


def _first_present(data: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in data:
            return data[key]
    return None
