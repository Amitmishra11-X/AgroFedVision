import os
import shutil

from client_train import train_client
from fedprox_utils import fedprox_aggregate
from differential_privacy import apply_dp

import tensorflow as tf
import sys

sys.path.append("ml_model")

from image_encoder import EfficientNetPreprocess


# =====================================================
# CONFIG
# =====================================================

ROUNDS = 3

CLIENTS = [
    "data/clients/farm_a.csv",
    "data/clients/farm_b.csv",
    "data/clients/farm_c.csv",
    "data/clients/farm_d.csv"
]

GLOBAL_DIR = "federated_multimodal/global_model"

os.makedirs(
    GLOBAL_DIR,
    exist_ok=True
)

INITIAL_MODEL = (
    "results/agrofedvision_fusion_model.keras"
)

# =====================================================
# INITIAL GLOBAL MODEL
# =====================================================

current_global = os.path.join(
    GLOBAL_DIR,
    "global_round_0.keras"
)

if not os.path.exists(current_global):

    shutil.copy(
        INITIAL_MODEL,
        current_global
    )

    print(
        "\nCreated Initial Global Model"
    )

# =====================================================
# FEDERATED ROUNDS
# =====================================================

for round_num in range(1, ROUNDS + 1):

    print("\n" + "=" * 60)
    print(
        f"FEDPROX ROUND {round_num}"
    )
    print("=" * 60)

    client_weights = []

    # -----------------------------------------
    # Local Training
    # -----------------------------------------

    for client in CLIENTS:

        print(
            f"\nTraining Client: {client}"
        )

        weights = train_client(
            client_csv=client,
            global_model_path=current_global,
            epochs=1,
            batch_size=8
        )
        weights = apply_dp(
    weights,
    sigma=0.001,
    clip_norm=1.0
)

        client_weights.append(
            weights
        )

    # -----------------------------------------
    # Aggregate
    # -----------------------------------------

    print(
        "\nAggregating Client Weights..."
    )

    global_weights = fedprox_aggregate(
        client_weights
    )

    # -----------------------------------------
    # Load Global Model
    # -----------------------------------------

    model = tf.keras.models.load_model(
        current_global,
        custom_objects={
            "EfficientNetPreprocess":
            EfficientNetPreprocess
        }
    )

    model.set_weights(
        global_weights
    )

    new_global = os.path.join(
        GLOBAL_DIR,
        f"global_round_{round_num}.keras"
    )

    model.save(
        new_global
    )

    current_global = new_global

    print(
        f"Saved: {new_global}"
    )

# =====================================================
# FINAL MODEL
# =====================================================

final_model = os.path.join(
    GLOBAL_DIR,
    f"global_round_{ROUNDS}.keras"
)

print("\n" + "=" * 60)
print("FEDPROX TRAINING COMPLETE")
print("=" * 60)

print(
    f"\nFinal Global Model:\n{final_model}"
)