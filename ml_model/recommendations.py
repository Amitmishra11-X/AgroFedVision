"""
recommendations.py

AgroFedVision -- Recommendation Engine

This is a curated knowledge base, NOT a machine learning model.
Maps (crop, predicted_class) -> actionable advice text. Honest about
what it is: a lookup table triggered by the model's output, not an
AI-generated recommendation. Expand per-crop as you refine advice
with real agronomic input.
"""

RECOMMENDATIONS = {
    "guava": {
        "Healthy": {
            "summary": "No signs of disease detected.",
            "actions": [
                "Continue current care routine.",
                "Monitor weekly for early signs of stress or discoloration.",
            ],
        },
        "Moderate_Risk": {
            "summary": "Early-stage stress or mild infection detected.",
            "actions": [
                "Inspect affected leaves closely and remove any severely damaged ones.",
                "Improve airflow around the plant to reduce humidity buildup.",
                "Re-check in 5-7 days to see if symptoms progress.",
            ],
        },
        "High_Risk": {
            "summary": "Significant disease pressure detected.",
            "actions": [
                "Isolate or prioritize this plant for treatment.",
                "Apply an appropriate fungicide/pesticide based on visual symptoms.",
                "Remove and dispose of heavily infected leaves to limit spread.",
                "Consult a local agricultural extension officer if symptoms are unfamiliar.",
            ],
        },
    },
    "maize": {
        "Healthy": {
            "summary": "No signs of infection detected.",
            "actions": [
                "Continue current care routine.",
                "Monitor for pest activity, especially during humid periods.",
            ],
        },
        "High_Risk": {
            "summary": "Infection detected in the crop.",
            "actions": [
                "Inspect the field for the extent of spread.",
                "Apply appropriate treatment based on visible symptoms.",
                "Consider consulting a local expert -- this model currently "
                "distinguishes Healthy/Infected only, not specific disease type.",
            ],
        },
        # Note: Maize currently has no Moderate_Risk data (see maize_dataset.csv
        # construction) -- this gap is disclosed, not hidden.
    },
}


def get_recommendation(crop, predicted_class):
    if crop not in RECOMMENDATIONS:
        return {"summary": "No recommendation data for this crop yet.", "actions": []}
    if predicted_class not in RECOMMENDATIONS[crop]:
        return {"summary": f"No recommendation data for class '{predicted_class}'.", "actions": []}
    return RECOMMENDATIONS[crop][predicted_class]