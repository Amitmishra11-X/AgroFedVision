"""
Crop-specific interpretation layer.

Converts raw model predictions into
a unified representation for
decision fusion.
"""

# ==========================================================
# Paddy
# ==========================================================

PADDY_MAP = {

    "Healthy": {

        "health_state": "Healthy",

        "risk": "Low",

        "severity": "None"

    },

    "Blast": {

        "health_state": "High_Risk",

        "risk": "High",

        "severity": "Severe"

    },

    "Brown_Spot": {

        "health_state": "Moderate_Risk",

        "risk": "Medium",

        "severity": "Moderate"

    },

    "Bacterial_Leaf_Blight": {

        "health_state": "High_Risk",

        "risk": "High",

        "severity": "Severe"

    },

    "Tungro": {

        "health_state": "High_Risk",

        "risk": "High",

        "severity": "Severe"

    }

}

# ==========================================================
# Maize
# ==========================================================

MAIZE_MAP = {

    "Healthy": {

        "health_state": "Healthy",

        "risk": "Low",

        "severity": "None"

    },

    "High_Risk": {

        "health_state": "High_Risk",

        "risk": "High",

        "severity": "High"

    }

}

# ==========================================================
# Guava
# ==========================================================

GUAVA_MAP = {

    "Healthy": {

        "health_state": "Healthy",

        "risk": "Low",

        "severity": "None"

    },

    "Moderate_Risk": {

        "health_state": "Moderate_Risk",

        "risk": "Medium",

        "severity": "Moderate"

    },

    "High_Risk": {

        "health_state": "High_Risk",

        "risk": "High",

        "severity": "High"

    }

}

# ==========================================================
# Interpreter
# ==========================================================

def interpret_prediction(crop, prediction):

    crop = crop.lower()

    if crop == "paddy":

        table = PADDY_MAP

    elif crop == "maize":

        table = MAIZE_MAP

    elif crop == "guava":

        table = GUAVA_MAP

    else:

        return {

            "disease": prediction,

            "health_state": prediction,

            "risk": "Unknown",

            "severity": "Unknown"

        }

    info = table.get(

        prediction,

        {

            "health_state": prediction,

            "risk": "Unknown",

            "severity": "Unknown"

        }

    )

    return {

        "crop": crop,

        "disease": prediction,

        **info

    }