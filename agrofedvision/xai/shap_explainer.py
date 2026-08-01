"""SHAP explanation wrapper for Keras image models."""

from __future__ import annotations

import keras
import numpy as np


class ShapImageExplainer:
    def __init__(self, model: keras.Model, background: np.ndarray) -> None:
        import shap

        self.model = model
        self.background = background.astype("float32")
        self.explainer = shap.GradientExplainer(model, self.background)

    def explain(self, images: np.ndarray, ranked_outputs: int | None = None):
        if ranked_outputs is None:
            return self.explainer.shap_values(images.astype("float32"))
        values, indexes = self.explainer.shap_values(
            images.astype("float32"),
            ranked_outputs=ranked_outputs,
        )
        return {"values": values, "indexes": indexes}
