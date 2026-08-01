import os
import sys
import pandas as pd
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath("."))

# Import YOUR fusion model builder
from ml_model.fusion_model import build_fusion_model


def train_client(model_path, client_csv):

    print(f"Training {client_csv}")

    # Rebuild architecture instead of load_model()
    model = build_fusion_model(
        num_sensor_features=23,
        num_uav_features=4,
        num_classes=3
    )

    try:
        model.load_weights(model_path)
        print("Weights loaded successfully")
    except Exception as e:
        print("Weight loading failed:", e)

    df = pd.read_csv(client_csv)

    print("Samples:", len(df))
    print("Columns:", len(df.columns))

    # For now just return weights
    return model.get_weights()