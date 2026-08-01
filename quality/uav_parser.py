"""
uav_parser.py
----------------------------------
Parses AgroFedVision UAV reports.

Author: AgroFedVision
"""

import re
from pathlib import Path


# -------------------------------------------------
# Helper
# -------------------------------------------------

def extract_percentage(text, keyword):
    """
    Extract percentage value.

    Example:
    Healthy : 23.80%
    """

    pattern = rf"{keyword}\s*:?\s*([0-9.]+)%"

    m = re.search(pattern, text, re.IGNORECASE)

    if m:
        return float(m.group(1))

    return None


# -------------------------------------------------
# Adaptive Nutrient Report
# -------------------------------------------------

def parse_adaptive_report(report_path):

    report_path = Path(report_path)

    if not report_path.exists():
        raise FileNotFoundError(report_path)

    text = report_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    data = {

        "healthy": extract_percentage(text, "Healthy"),

        "nitrogen": extract_percentage(text, "Nitrogen"),

        "phosphorus": extract_percentage(text, "Phosphorus"),

        "potassium": extract_percentage(text, "Potassium"),

        "recommendations": []
    }

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("-"):

            data["recommendations"].append(
                line.replace("-", "").strip()
            )

    return data


# -------------------------------------------------
# Prescription Report
# -------------------------------------------------

def parse_prescription_report(report_path):

    report_path = Path(report_path)

    if not report_path.exists():
        raise FileNotFoundError(report_path)

    text = report_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    data = {

        "healthy": extract_percentage(text, "Healthy Area"),

        "nitrogen": extract_percentage(text, "Nitrogen Deficiency"),

        "phosphorus": extract_percentage(text, "Phosphorus Deficiency"),

        "potassium": extract_percentage(text, "Potassium Deficiency"),

        "recommendations": []
    }

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("-"):

            data["recommendations"].append(
                line.replace("-", "").strip()
            )

    return data


# -------------------------------------------------
# Test
# -------------------------------------------------

if __name__ == "__main__":

    adaptive = parse_adaptive_report(
        "results/adaptive_nutrient_report.txt"
    )

    prescription = parse_prescription_report(
        "results/fertilizer_recommendation_report.txt"
    )

    print("\nAdaptive Report")
    print(adaptive)

    print("\nPrescription Report")
    print(prescription)