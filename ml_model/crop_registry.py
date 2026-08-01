"""
crop_registry.py

AgroFedVision -- Crop Registry

Single source of truth for which crops the system supports, where
their trained models live, and what their class labels mean. Adding
a new crop later means adding one entry here, not rewiring code.
"""

CROP_REGISTRY = {
    "guava": {
        "display_name": "Guava",
        "model_path": "results/phase1_guava/fold_1_model.keras",  # update to your best fold
        "encoder_path": "results/phase1_guava/label_encoder.pkl",
        "img_size": 224,
        "status": "trained",
    },
    
"paddy": {
    "display_name": "Paddy",
    "model_path": "results/paddy/paddy_best_model.keras",
    "encoder_path": "results/paddy/label_encoder.pkl",
    "img_size": 224,
    "status": "trained",
},
"maize": {
    "display_name": "Maize (Corn)",
    "model_path": "results/maize/maize_best_model.keras",
    "encoder_path": "results/maize/label_encoder.pkl",
    "img_size": 224,
    "status": "trained",
},
    # Add future crops here, e.g.:
    # "rice": {
    #     "display_name": "Rice",
    #     "model_path": "results/phase1_rice/fold_1_model.keras",
    #     "encoder_path": "results/phase1_rice/label_encoder.pkl",
    #     "img_size": 224,
    #     "status": "coming_soon",   # use this until a model is trained
    # },
}


def list_available_crops():
    """Returns crop keys that have a trained model ready to use."""
    return [k for k, v in CROP_REGISTRY.items() if v["status"] == "trained"]


def get_crop_config(crop_name):
    if crop_name not in CROP_REGISTRY:
        raise ValueError(
            f"Unknown crop '{crop_name}'. Available: {list(CROP_REGISTRY.keys())}"
        )
    config = CROP_REGISTRY[crop_name]
    if config["status"] != "trained":
        raise ValueError(
            f"Crop '{crop_name}' has no trained model yet (status: {config['status']})"
        )
    return config