"""
sensor_recommendations.py

Rule-based interpretation of environmental sensor readings.

This module converts raw sensor values into
farmer-friendly recommendations.
"""


def analyze_sensor_readings(sensor):

    report = {}

    # -------------------------------
    # Nitrogen
    # -------------------------------
    if sensor["N"] < 40:
        report["nitrogen_status"] = "Low"
    elif sensor["N"] < 80:
        report["nitrogen_status"] = "Optimal"
    else:
        report["nitrogen_status"] = "High"

    # -------------------------------
    # Phosphorus
    # -------------------------------
    if sensor["P"] < 25:
        report["phosphorus_status"] = "Low"
    elif sensor["P"] < 60:
        report["phosphorus_status"] = "Optimal"
    else:
        report["phosphorus_status"] = "High"

    # -------------------------------
    # Potassium
    # -------------------------------
    if sensor["K"] < 30:
        report["potassium_status"] = "Low"
    elif sensor["K"] < 70:
        report["potassium_status"] = "Optimal"
    else:
        report["potassium_status"] = "High"

    # -------------------------------
    # Moisture
    # -------------------------------
    moisture = sensor["soil_moisture"]

    if moisture < 20:
        report["moisture_status"] = "Very Low"
    elif moisture < 40:
        report["moisture_status"] = "Low"
    elif moisture < 70:
        report["moisture_status"] = "Optimal"
    else:
        report["moisture_status"] = "High"

    # -------------------------------
    # pH
    # -------------------------------
    ph = sensor["ph"]

    if ph < 5.5:
        report["soil_ph"] = "Acidic"
    elif ph <= 7.5:
        report["soil_ph"] = "Neutral"
    else:
        report["soil_ph"] = "Alkaline"

    # -------------------------------
    # Temperature
    # -------------------------------
    temp = sensor["temperature"]

    if temp < 18:
        report["temperature_status"] = "Low"
    elif temp <= 32:
        report["temperature_status"] = "Optimal"
    else:
        report["temperature_status"] = "High"

    # -------------------------------
    # Humidity
    # -------------------------------
    humidity = sensor["humidity"]

    if humidity < 35:
        report["humidity_status"] = "Low"
    elif humidity <= 80:
        report["humidity_status"] = "Optimal"
    else:
        report["humidity_status"] = "High"

    # --------------------------------
    # Fertilizer recommendation
    # --------------------------------

    fertilizer = []

    if report["nitrogen_status"] == "Low":
        fertilizer.append("Apply nitrogen-rich fertilizer")

    if report["phosphorus_status"] == "Low":
        fertilizer.append("Apply phosphorus fertilizer")

    if report["potassium_status"] == "Low":
        fertilizer.append("Apply potash fertilizer")

    report["fertilizer"] = fertilizer

    # --------------------------------
    # Irrigation recommendation
    # --------------------------------

    if moisture < 20:
        irrigation = "Immediate irrigation required"

    elif moisture < 40:
        irrigation = "Irrigation recommended"

    else:
        irrigation = "No irrigation required"

    report["irrigation"] = irrigation

    # --------------------------------
    # Stress Score
    # --------------------------------

    score = 100

    if moisture < 40:
        score -= 20

    if report["nitrogen_status"] == "Low":
        score -= 15

    if report["phosphorus_status"] == "Low":
        score -= 10

    if report["potassium_status"] == "Low":
        score -= 10

    if temp > 35:
        score -= 10

    report["sensor_health_score"] = max(score, 0)

    return report