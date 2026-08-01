"""
Plant Statistics Module
-----------------------
Aggregates predictions from multiple leaf images into
plant-level health statistics.
"""

from collections import Counter
import numpy as np


class PlantStatistics:

    def __init__(self):

        self.total_images = 0

        self.best_image = None
        self.best_confidence = -1.0

        self.class_counter = Counter()

        self.per_image_results = []

        self.all_probabilities = []

        # Severity mapping
        self.severity_map = {
            "Healthy": 0,
            "Moderate_Risk": 1,
            "High_Risk": 2
        }

    # -------------------------------------------------
    # Add One Image Result
    # -------------------------------------------------

    def add_prediction(
        self,
        image_path,
        predicted_class,
        confidence,
        probabilities
    ):

        self.total_images += 1

        self.class_counter[predicted_class] += 1

        self.per_image_results.append({

            "image": image_path,

            "prediction": predicted_class,

            "confidence": confidence

        })

        self.all_probabilities.append(probabilities)

        if confidence > self.best_confidence:

            self.best_confidence = confidence

            self.best_image = image_path

    # -------------------------------------------------
    # Average Probabilities
    # -------------------------------------------------

    def average_probabilities(self):

        return np.mean(

            np.stack(self.all_probabilities),

            axis=0

        )

    # -------------------------------------------------
    # Majority Class
    # -------------------------------------------------

    def majority_class(self):

        if self.total_images == 0:
            return None

        return self.class_counter.most_common(1)[0][0]

    # -------------------------------------------------
    # Severity %
    # -------------------------------------------------

    def severity_percentage(self):

        score = 0

        for disease, count in self.class_counter.items():

            score += (

                self.severity_map.get(disease, 0)

                * count

            )

        maximum = self.total_images * 2

        if maximum == 0:
            return 0.0

        return (score / maximum) * 100

    # -------------------------------------------------
    # Statistics Dictionary
    # -------------------------------------------------

        # -------------------------------------------------
    # Plant Summary
    # -------------------------------------------------

    def summary(self):

        healthy = self.class_counter.get("Healthy", 0)
        moderate = self.class_counter.get("Moderate_Risk", 0)
        high = self.class_counter.get("High_Risk", 0)

        # ---------------------------------------------
        # Disease Percentage
        # ---------------------------------------------

        healthy_percent = (healthy / self.total_images) * 100 \
            if self.total_images else 0

        moderate_percent = (moderate / self.total_images) * 100 \
            if self.total_images else 0

        high_percent = (high / self.total_images) * 100 \
            if self.total_images else 0

        # ---------------------------------------------
        # Plant Health Index (0–100)
        # ---------------------------------------------

        health_index = (

            healthy * 100 +

            moderate * 60 +

            high * 20

        ) / self.total_images if self.total_images else 0

        # ---------------------------------------------
        # Reliability Score
        # ---------------------------------------------

        if len(self.all_probabilities):

            reliability = float(

                np.mean(

                    [

                        np.max(prob)

                        for prob in self.all_probabilities

                    ]

                )

            )

        else:

            reliability = 0

        # ---------------------------------------------
        # Mixed Disease Detection
        # ---------------------------------------------

        disease_types = 0

        if healthy:
            disease_types += 1

        if moderate:
            disease_types += 1

        if high:
            disease_types += 1

        mixed_disease = disease_types > 1

        # ---------------------------------------------
        # Disease Distribution
        # ---------------------------------------------

        distribution = {

            "Healthy": healthy,

            "Moderate_Risk": moderate,

            "High_Risk": high

        }

        # ---------------------------------------------
        # Final Summary
        # ---------------------------------------------

        return {

            "total_images": self.total_images,

            "healthy": healthy,

            "moderate": moderate,

            "high": high,

            "healthy_percent": healthy_percent,

            "moderate_percent": moderate_percent,

            "high_percent": high_percent,

            "majority_class": self.majority_class(),

            "severity_percentage": self.severity_percentage(),

            "health_index": health_index,

            "reliability_score": reliability,

            "mixed_disease": mixed_disease,

            "distribution": distribution,

            "best_image": self.best_image,

            "best_confidence": self.best_confidence,

            "per_image_results": self.per_image_results

        }