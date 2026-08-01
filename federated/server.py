import pandas as pd
import numpy as np
from fedprox import train_client

clients = [
    "data/clients/farm_a.csv",
    "data/clients/farm_b.csv",
    "data/clients/farm_c.csv",
    "data/clients/farm_d.csv"
]

feature_importances = []

for client in clients:

    model = train_client(client)

    feature_importances.append(
        model.feature_importances_
    )

    print("Trained:", client)

global_importance = np.mean(
    feature_importances,
    axis=0
)

print("\nGlobal Feature Importance")

for i, value in enumerate(global_importance):
    print(f"Feature {i}: {value:.4f}")