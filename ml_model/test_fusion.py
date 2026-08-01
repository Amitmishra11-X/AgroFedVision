"""
==========================================================
AgroFedVision Decision Fusion Test
==========================================================

Tests all supported prediction modes.

1. Image Only
2. Sensor Only
3. UAV Only
4. Image + Sensor
5. Image + UAV
6. Sensor + UAV
7. Image + Sensor + UAV
"""

from pprint import pprint

from decision_fusion import fuse_predictions


# ==========================================================
# SAMPLE DATA
# ==========================================================

image_result = {
    "prediction": "Healthy",
    "confidence": 1.0,
    "recommendation": {
        "actions": [
            "Continue regular monitoring."
        ]
    }
}


sensor_result = {

    "confidence": 0.50,

    "analysis": {

        "sensor_health_score": 55,

        "fertilizer": [

            "Apply nitrogen-rich fertilizer",

            "Apply potash fertilizer"

        ],

        "irrigation": "Irrigation recommended"

    }

}


uav_result = {

    "NDVI_Mean": 0.42

}


# ==========================================================
# REPORT FUNCTION
# ==========================================================

def print_report(title, result):

    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)

    print(f"Prediction Mode          : {result['prediction_mode']}")
    print(f"Overall Status           : {result['overall_status']}")
    print(f"Risk Level               : {result['risk_level']}")
    print(f"Overall Health Score     : {result['overall_health_score']}/100")
    print(f"Overall Confidence       : {result['overall_confidence']}%")
    print(f"Disease Severity         : {result['disease_severity']}")
    print(f"Stress Level             : {result['stress_level']}")
    print(f"Environmental Condition  : {result['environmental_condition']}")

    print("\nIndividual Scores")

    if result["image_score"] is not None:
        print(f"Image Score              : {result['image_score']}")

    if result["sensor_score"] is not None:
        print(f"Sensor Score             : {result['sensor_score']}")

    if result["uav_score"] is not None:
        print(f"UAV Score                : {result['uav_score']}")

    print("\nFusion Weights")

    for k, v in result["weights"].items():
        print(f"{k.capitalize():<10}: {v*100:.1f}%")

    print("\nFusion Details")

    pprint(result["fusion_details"])

    print("\nRecommendations")

    for rec in result["recommendations"]:
        print(f"• {rec}")


# ==========================================================
# TEST 1
# IMAGE ONLY
# ==========================================================

result = fuse_predictions(
    image_result=image_result
)

print_report("TEST 1 : IMAGE ONLY", result)


# ==========================================================
# TEST 2
# SENSOR ONLY
# ==========================================================

result = fuse_predictions(
    sensor_result=sensor_result
)

print_report("TEST 2 : SENSOR ONLY", result)


# ==========================================================
# TEST 3
# UAV ONLY
# ==========================================================

result = fuse_predictions(
    uav_result=uav_result
)

print_report("TEST 3 : UAV ONLY", result)


# ==========================================================
# TEST 4
# IMAGE + SENSOR
# ==========================================================

result = fuse_predictions(
    image_result=image_result,
    sensor_result=sensor_result
)

print_report("TEST 4 : IMAGE + SENSOR", result)


# ==========================================================
# TEST 5
# IMAGE + UAV
# ==========================================================

result = fuse_predictions(
    image_result=image_result,
    uav_result=uav_result
)

print_report("TEST 5 : IMAGE + UAV", result)


# ==========================================================
# TEST 6
# SENSOR + UAV
# ==========================================================

result = fuse_predictions(
    sensor_result=sensor_result,
    uav_result=uav_result
)

print_report("TEST 6 : SENSOR + UAV", result)


# ==========================================================
# TEST 7
# IMAGE + SENSOR + UAV
# ==========================================================

result = fuse_predictions(
    image_result=image_result,
    sensor_result=sensor_result,
    uav_result=uav_result
)

print_report("TEST 7 : IMAGE + SENSOR + UAV", result)