"""
==============================================================
AgroFedVision Decision Fusion Engine (Version 3)
==============================================================

Supports:

1. Image Only
2. Sensor Only
3. UAV Only
4. Image + Sensor
5. Image + UAV
6. Sensor + UAV
7. Image + Sensor + UAV

Outputs:

• Prediction Mode
• Individual Modality Scores
• Overall Health Score
• Overall Confidence
• Disease Severity
• Stress Level
• Environmental Condition
• Overall Status
• Risk Level
• Explainable Fusion Details
• Unified Recommendations

==============================================================
"""

from collections import OrderedDict
# ==========================================================
# IMAGE SCORING
# ==========================================================
def calculate_image_score(image_result):
    """
    Plant-level intelligent image scoring.

    Uses:
        • Plant Health Index
        • Disease Distribution
        • Reliability
        • Mixed Disease
        • Majority Class

    Falls back to legacy scoring if plant statistics
    are unavailable.
    """

    if image_result is None:
        return None

    # =====================================================
    # NEW PLANT-LEVEL SCORING
    # =====================================================

    if "plant_statistics" in image_result:

        ps = image_result["plant_statistics"]

    total = max(ps.get("total_images", 1), 1)

    healthy = ps.get("healthy", 0)
    moderate = ps.get("moderate", 0)
    high = ps.get("high", 0)

    reliability = ps.get(
        "reliability_score",
        1.0
    )

    mixed = ps.get(
        "mixed_disease",
        False
    )

    healthy_ratio = healthy / total
    moderate_ratio = moderate / total
    high_ratio = high / total

    # -------------------------------------
    # Weighted Health Score
    # -------------------------------------

    score = (

        healthy_ratio * 100 +

        moderate_ratio * 60 +

        high_ratio * 20

    )

    # -------------------------------------
    # Reliability Bonus
    # -------------------------------------

    score += reliability * 3

    # -------------------------------------
    # Small Mixed Disease Penalty
    # -------------------------------------

    if mixed:
        score -= 2

    score = max(0, min(score, 100))

    return round(score, 2)
    # =====================================================
    # LEGACY VERSION
    # =====================================================

    prediction = image_result["prediction"]

    confidence = image_result.get(
        "confidence",
        1.0
    )

    if prediction.lower() == "healthy":

        score = 80 + confidence * 20

    else:

        score = max(
            20,
            (1 - confidence) * 60
        )

    return round(score, 2)

# ==========================================================
# SENSOR SCORING
# ==========================================================

def calculate_sensor_score(sensor_result):
    """
    Convert sensor analysis into health score.
    """

    if sensor_result is None:
        return None

    analysis = sensor_result["analysis"]

    base = analysis["sensor_health_score"]

    confidence = sensor_result.get("confidence", 0.5)

    score = base * (0.5 + confidence / 2)

    score = max(0, min(score, 100))

    return round(score, 2)
# ==========================================================
# UAV SCORING
# ==========================================================

def calculate_uav_score(uav_result):
    """
    Intelligent UAV field scoring using
    NDVI + NDRE + field variability.
    """

    if uav_result is None:
        return None

    ndvi = uav_result.get("NDVI_Mean", 0.50)

    ndvi_std = uav_result.get("NDVI_Std", 0)

    ndre = uav_result.get("NDRE_Mean", ndvi)

    score = ndvi * 70

    score += ndre * 20

    score += 10

    if ndvi_std > 0.20:

        score -= 10

    elif ndvi_std > 0.10:

        score -= 5

    score = max(0, min(score, 100))

    return round(score, 2)
# ==========================================================
# HEALTH STATUS
# ==========================================================

def get_status(score):

    if score >= 85:
        return "Healthy", "Low"

    elif score >= 55:
        return "Moderate", "Medium"

    elif score >= 35:
        return "Moderate-High", "High"

    else:
        return "High Risk", "High"
# ==========================================================
# FUSION CONFIDENCE
# ==========================================================

def calculate_overall_confidence(
    image_result=None,
    sensor_result=None,
    uav_result=None
):
    """
    Calculate confidence of the fused decision.
    """

    confidence = 0
    total_weight = 0

    if image_result is not None:

        confidence += image_result.get("confidence", 1.0) * 0.60
        total_weight += 0.60

    if sensor_result is not None:

        confidence += sensor_result.get("confidence", 0.5) * 0.25
        total_weight += 0.25

    if uav_result is not None:

        # UAV currently has no confidence model
        confidence += 1.0 * 0.15
        total_weight += 0.15

    if total_weight == 0:
        return 0

    confidence = confidence / total_weight

    return round(confidence * 100, 2)
# ==========================================================
# DISEASE SEVERITY
# ==========================================================

def get_disease_severity(image_result):

    if image_result is None:
        return "Unknown"

    prediction = image_result["prediction"].lower()

    confidence = image_result.get("confidence", 1)

    if prediction == "healthy":
        return "None"

    if confidence >= 0.90:
        return "Severe"

    elif confidence >= 0.70:
        return "Moderate"

    return "Mild"
# ==========================================================
# STRESS LEVEL
# ==========================================================

def get_stress_level(sensor_score, uav_score):

    values = []

    if sensor_score is not None:
        values.append(sensor_score)

    if uav_score is not None:
        values.append(uav_score)

    if len(values) == 0:
        return "Unknown"

    avg = sum(values) / len(values)

    if avg >= 80:
        return "Low"

    elif avg >= 60:
        return "Moderate"

    return "High"
# ==========================================================
# ENVIRONMENTAL CONDITION
# ==========================================================

def get_environment_condition(sensor_score):

    if sensor_score is None:
        return "Unknown"

    if sensor_score >= 80:
        return "Optimal"

    elif sensor_score >= 60:
        return "Acceptable"

    return "Poor"
# ==========================================================
# MAIN DECISION FUSION
# ==========================================================

def fuse_predictions(
    image_result=None,
    sensor_result=None,
    uav_result=None
):
    """
    Perform multimodal decision fusion.

    Supports:
        Image
        Sensor
        UAV

    in every possible combination.
    """

    # ------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------

    recommendations = []

    scores = {}

    weights = {}

    mode = []

    # ======================================================
    # IMAGE MODULE
    # ======================================================

    if image_result is not None:

        mode.append("Image")

        image_score = calculate_image_score(image_result)

        scores["image"] = image_score

        weights["image"] = 0.60

        recommendation = image_result.get("recommendation")

        if recommendation is not None:

            actions = recommendation.get("actions", [])

            recommendations.extend(actions)

    # ======================================================
    # SENSOR MODULE
    # ======================================================

    if sensor_result is not None:

        mode.append("Sensor")

        sensor_score = calculate_sensor_score(sensor_result)

        scores["sensor"] = sensor_score

        weights["sensor"] = 0.25

        analysis = sensor_result.get("analysis", {})

        fertilizer = analysis.get("fertilizer", [])

        recommendations.extend(fertilizer)

        irrigation = analysis.get("irrigation")

        if irrigation:
            recommendations.append(irrigation)

    # ======================================================
    # UAV MODULE
    # ======================================================

    if uav_result is not None:

        mode.append("UAV")

        uav_score = calculate_uav_score(uav_result)

        scores["uav"] = uav_score

        weights["uav"] = 0.15

        ndvi = uav_result.get("NDVI_Mean")

        if ndvi is not None:

            ndvi_std = uav_result.get("NDVI_Std", 0)

            if ndvi < 0.35:

                recommendations.append(
            "Field shows severe vegetation stress. Immediate inspection recommended."
        )

            elif ndvi < 0.55:

                recommendations.append(
            "Moderate vegetation vigor detected. Monitor crop growth."
        )

            else:

             recommendations.append(
            "Overall field vegetation is healthy."
        )

            if ndvi_std > 0.20:

             recommendations.append(
            "High NDVI variability detected. Crop health is uneven across the field."
        )
    # ======================================================
    # DETERMINE PREDICTION MODE
    # ======================================================

    prediction_mode = " + ".join(mode)

    # ======================================================
    # NORMALIZE WEIGHTS
    # ======================================================

    total_weight = sum(weights.values())

    if total_weight > 0:

        for key in weights:

            weights[key] = round(
                weights[key] / total_weight,
                4
            )

    # ======================================================
    # CALCULATE OVERALL SCORE
    # ======================================================

    overall_score = 0

    for key in scores:

        overall_score += scores[key] * weights[key]

    overall_score = round(overall_score, 2)

    # ======================================================
    # OVERALL STATUS
    # ======================================================

    overall_status, risk_level = get_status(
        overall_score
    )# ======================================================
# PLANT-LEVEL ESCALATION
# ======================================================

    if image_result is not None:

        ps = image_result.get("plant_statistics", {})

        total = max(ps.get("total_images", 1), 1)

        high_ratio = ps.get("high", 0) / total

    # Escalate only if most leaves are High Risk
        if high_ratio >= 0.50:

            overall_status = "High Risk"

            risk_level = "High"

    # Flag mixed disease when minority of leaves are High Risk
        elif high_ratio >= 0.25 and overall_status == "Moderate":

            overall_status = "Moderate (Mixed Disease)"
    

    # ======================================================
    # EXTRA ANALYSIS
    # ======================================================

    overall_confidence = calculate_overall_confidence(
        image_result,
        sensor_result,
        uav_result
    )

    disease_severity = get_disease_severity(
        image_result
    )

    stress_level = get_stress_level(
        scores.get("sensor"),
        scores.get("uav")
    )

    environmental_condition = get_environment_condition(
        scores.get("sensor")
    )

    # ======================================================
    # REMOVE DUPLICATE RECOMMENDATIONS
    # ======================================================

    recommendations = list(
        OrderedDict.fromkeys(recommendations)
    )

    # ======================================================
    # BUILD EXPLAINABLE FUSION DETAILS
    # ======================================================

    fusion_details = {

       "image": {

    "available": image_result is not None,

    "score": scores.get("image"),

    "weight": weights.get("image"),

    "confidence":
        None
        if image_result is None
        else image_result.get("confidence"),

    "health_index":
        None
        if image_result is None
        else image_result.get(
            "plant_statistics",
            {}
        ).get("health_index"),

    "majority_class":
        None
        if image_result is None
        else image_result.get(
            "plant_statistics",
            {}
        ).get("majority_class"),

    "mixed_disease":
        None
        if image_result is None
        else image_result.get(
            "plant_statistics",
            {}
        ).get("mixed_disease"),

    "reliability":
        None
        if image_result is None
        else image_result.get(
            "plant_statistics",
            {}
        ).get("reliability_score"),

    "distribution":
        None
        if image_result is None
        else image_result.get(
            "plant_statistics",
            {}
        ).get("distribution")

},
        "sensor": {

            "available": sensor_result is not None,

            "score": scores.get("sensor"),

            "weight": weights.get("sensor"),

            "confidence":
                None
                if sensor_result is None
                else sensor_result.get("confidence")

        },

        "uav": {

    "available": uav_result is not None,

    "score": scores.get("uav"),

    "weight": weights.get("uav"),

    "ndvi":
        None
        if uav_result is None
        else uav_result.get("NDVI_Mean"),

    "ndvi_std":
        None
        if uav_result is None
        else uav_result.get("NDVI_Std"),

    "ndre":
        None
        if uav_result is None
        else uav_result.get("NDRE_Mean"),

    "captures":
        None
        if uav_result is None
        else uav_result.get(
            "uav_statistics",
            {}
        ).get("total_images"),

    "best_capture":
        None
        if uav_result is None
        else uav_result.get(
            "uav_statistics",
            {}
        ).get("best_image")

}
    }

    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    return {

        "prediction_mode": prediction_mode,

        "overall_health_score": overall_score,

        "overall_confidence": overall_confidence,

        "overall_status": overall_status,

        "risk_level": risk_level,

        "stress_level": stress_level,

        "disease_severity": disease_severity,

        "environmental_condition": environmental_condition,

        "image_score": scores.get("image"),

        "sensor_score": scores.get("sensor"),

        "uav_score": scores.get("uav"),

        "weights": weights,

        "fusion_details": fusion_details,

        "recommendations": recommendations

    }