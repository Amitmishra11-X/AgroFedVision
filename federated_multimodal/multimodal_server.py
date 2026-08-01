import glob
import tensorflow as tf

from multimodal_client import train_client
from aggregate import aggregate_weights
from ml_model.fusion_model import build_fusion_model

MODEL_PATH = "results/agrofedvision_fusion_model.keras"

client_files = glob.glob("data/clients/*.csv")

client_weights = []

for client in client_files:

    weights = train_client(
        MODEL_PATH,
        client
    )

    client_weights.append(weights)

# Rebuild global model
global_model = build_fusion_model(
    num_sensor_features=23,
    num_uav_features=4,
    num_classes=3
)

new_weights = aggregate_weights(
    client_weights
)

global_model.set_weights(
    new_weights
)

global_model.save_weights(
    "results/global_agrofedvision.weights.h5"
)

print("\nGlobal Weights Saved")