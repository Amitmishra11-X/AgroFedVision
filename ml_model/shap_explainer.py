"""
shap_explainer.py
AgroFedVision - SHAP explanation for the unified multimodal fusion model.

Explains the structured inputs:
    sensor features + UAV features

The image is held fixed while SHAP measures how structured features
change the predicted class probability. This is intentionally separate
from Grad-CAM, which explains spatial image evidence.
"""

import json
import numpy as np
import shap


class AgroFedSHAP:
    def __init__(
        self,
        model,
        sensor_cols,
        class_names,
        background_sensor,
        background_uav,
        background_image,
    ):
        self.model = model
        self.sensor_cols = list(sensor_cols)
        self.class_names = [str(x) for x in class_names]

        self.background_sensor = np.asarray(
            background_sensor, dtype=np.float32
        )
        self.background_uav = np.asarray(
            background_uav, dtype=np.float32
        )
        self.background_image = np.asarray(
            background_image, dtype=np.float32
        )

        self.uav_cols = [
            "NDVI_Mean",
            "NDVI_Std",
            "NDRE_Mean",
            "NDRE_Std",
        ]

        self.feature_names = self.sensor_cols + self.uav_cols

    def _predict_structured(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_sensor = len(self.sensor_cols)

        sensor = X[:, :n_sensor]
        uav = X[:, n_sensor:]

        image = np.repeat(
            self.background_image,
            repeats=len(X),
            axis=0,
        )

        # KernelExplainer can ask for thousands of masked samples at once.
        # Never materialize all of those 224x224x3 images simultaneously.
        batch_size = 16
        outputs = []
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            xb = X[start:end]
            sensor_b = xb[:, :n_sensor]
            uav_b = xb[:, n_sensor:]
            image_b = np.repeat(self.background_image, repeats=len(xb), axis=0)
            pred = self.model.predict(
                {
                    "sensor_input": sensor_b,
                    "uav_input": uav_b,
                    "image_input": image_b,
                },
                verbose=0,
                batch_size=batch_size,
            )
            outputs.append(np.asarray(pred, dtype=np.float64))
        return np.concatenate(outputs, axis=0)

    @staticmethod
    def _extract_class_shap(shap_values, predicted_index, n_features, n_classes):
        """
        Normalize SHAP's different output formats across SHAP versions.
        Returns one vector of length n_features for the predicted class.
        """
        arr = np.asarray(shap_values)

        # Older SHAP: list[class] -> (samples, features)
        if isinstance(shap_values, list):
            arr = np.asarray(shap_values[predicted_index])
            if arr.ndim == 2:
                return arr[0].reshape(-1)

        # Common modern SHAP: (samples, features, outputs)
        if arr.ndim == 3:
            if arr.shape[0] == 1 and arr.shape[1] == n_features:
                return arr[0, :, predicted_index].reshape(-1)

            # Alternative: (outputs, samples, features)
            if (
                arr.shape[0] == n_classes
                and arr.shape[1] == 1
                and arr.shape[2] == n_features
            ):
                return arr[predicted_index, 0, :].reshape(-1)

            # Alternative single-output-like layout
            if arr.shape[0] == 1 and arr.shape[2] == n_features:
                return arr[0, predicted_index, :].reshape(-1)

        # Single-output / already selected class
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0].reshape(-1)

        if arr.ndim == 1:
            return arr.reshape(-1)

        raise ValueError(
            f"Unsupported SHAP output shape: {arr.shape}"
        )

    def explain(
        self,
        sensor,
        uav,
        image,
        top_k=10,
        nsamples=100,
    ):
        sensor = np.asarray(sensor, dtype=np.float32).reshape(1, -1)
        uav = np.asarray(uav, dtype=np.float32).reshape(1, -1)
        image = np.asarray(image, dtype=np.float32)

        structured_input = np.concatenate(
            [sensor, uav],
            axis=1,
        )

        background = np.concatenate(
            [
                self.background_sensor,
                self.background_uav,
            ],
            axis=1,
        )

        if len(background) == 0:
            raise ValueError("SHAP background is empty.")

        prediction = self._predict_structured(structured_input)[0]
        predicted_index = int(np.argmax(prediction))
        explained_class = self.class_names[predicted_index]
        predicted_probability = float(prediction[predicted_index])

        explainer = shap.KernelExplainer(
            self._predict_structured,
            background,
        )

        shap_values = explainer.shap_values(
            structured_input,
            nsamples=nsamples,
        )

        class_shap = self._extract_class_shap(
            shap_values=shap_values,
            predicted_index=predicted_index,
            n_features=len(self.feature_names),
            n_classes=len(self.class_names),
        )

        if len(class_shap) != len(self.feature_names):
            raise ValueError(
                "SHAP feature count mismatch: "
                f"{len(class_shap)} values for "
                f"{len(self.feature_names)} features."
            )

        n_sensor = len(self.sensor_cols)

        sensor_abs = float(
            np.sum(np.abs(class_shap[:n_sensor]))
        )
        uav_abs = float(
            np.sum(np.abs(class_shap[n_sensor:]))
        )
        total_abs = sensor_abs + uav_abs

        if total_abs > 0:
            sensor_percentage = sensor_abs / total_abs * 100.0
            uav_percentage = uav_abs / total_abs * 100.0
        else:
            sensor_percentage = 0.0
            uav_percentage = 0.0

        features = []

        for i, feature in enumerate(self.feature_names):
            shap_value = float(class_shap[i])

            if shap_value > 0:
                direction = "positive_for_predicted_class"
            elif shap_value < 0:
                direction = "negative_for_predicted_class"
            else:
                direction = "neutral"

            features.append(
                {
                    "feature": feature,
                    "modality": (
                        "sensor"
                        if i < n_sensor
                        else "uav"
                    ),
                    "value": float(structured_input[0, i]),
                    "shap_value": shap_value,
                    "absolute_shap": abs(shap_value),
                    "direction": direction,
                }
            )

        features.sort(
            key=lambda item: item["absolute_shap"],
            reverse=True,
        )

        top_features = features[:top_k]

        return {
            "method": "SHAP KernelExplainer",
            "scope": (
                "Structured fusion inputs only "
                "(sensor + UAV); image held fixed."
            ),
            "explained_class": explained_class,
            "predicted_probability": predicted_probability,
            "modality_contribution": {
                "sensor": float(sensor_percentage),
                "uav": float(uav_percentage),
            },
            "top_features": top_features,
            "all_features": features,
        }


def save_shap_result(result, output_path):
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False,
        )
