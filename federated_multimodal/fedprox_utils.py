"""
fedprox_utils.py

FedProx helper functions for AgroFedVision.

FedAvg:
    minimize local loss

FedProx:
    minimize local loss +
    (mu/2) * ||w_local - w_global||²

This prevents client models from drifting too far
from the global model on non-IID data.
"""

import tensorflow as tf
import numpy as np


# --------------------------------------------------
# FedProx Penalty
# --------------------------------------------------

def fedprox_penalty(
    local_weights,
    global_weights
):
    """
    Computes:

    ||w_local - w_global||²
    """

    penalty = 0.0

    for lw, gw in zip(
        local_weights,
        global_weights
    ):

        penalty += tf.reduce_sum(
            tf.square(lw - gw)
        )

    return penalty


# --------------------------------------------------
# FedProx Loss
# --------------------------------------------------

def fedprox_loss(
    y_true,
    y_pred,
    model,
    global_weights,
    mu=0.01
):
    """
    Total Loss

    CrossEntropy
      +
    FedProx Penalty
    """

    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true,
        y_pred
    )

    ce_loss = tf.reduce_mean(
        ce_loss
    )

    prox = fedprox_penalty(
        model.trainable_weights,
        global_weights
    )

    total_loss = ce_loss + (
        mu / 2.0
    ) * prox

    return total_loss


# --------------------------------------------------
# Save Model Weights
# --------------------------------------------------

def get_weights(model):
    """
    Extract weights as numpy arrays.
    """

    return [
        w.numpy()
        for w in model.weights
    ]


# --------------------------------------------------
# Set Model Weights
# --------------------------------------------------

def set_weights(
    model,
    weights
):
    """
    Load weights into model.
    """

    model.set_weights(
        weights
    )


# --------------------------------------------------
# FedAvg Aggregation
# --------------------------------------------------

def fedavg(
    client_weights
):
    """
    Standard Federated Averaging.
    """

    avg_weights = []

    for weights in zip(*client_weights):

        avg_weights.append(
            np.mean(
                weights,
                axis=0
            )
        )

    return avg_weights


# --------------------------------------------------
# FedProx Aggregation
# --------------------------------------------------

def fedprox_aggregate(
    client_weights,
    client_sizes=None
):
    """
    Weighted averaging.

    Larger farms contribute more.
    """

    if client_sizes is None:

        client_sizes = [
            1
        ] * len(client_weights)

    total = sum(
        client_sizes
    )

    agg_weights = []

    for layer_weights in zip(
        *client_weights
    ):

        weighted_sum = np.zeros_like(
            layer_weights[0]
        )

        for w, size in zip(
            layer_weights,
            client_sizes
        ):

            weighted_sum += (
                w * size
            )

        agg_weights.append(
            weighted_sum / total
        )

    return agg_weights


# --------------------------------------------------
# Evaluate Drift
# --------------------------------------------------

def model_distance(
    local_weights,
    global_weights
):
    """
    How far a client moved away
    from the global model.
    """

    dist = 0.0

    for lw, gw in zip(
        local_weights,
        global_weights
    ):

        dist += np.sum(
            (lw - gw) ** 2
        )

    return float(
        np.sqrt(dist)
    )


# --------------------------------------------------
# Client Statistics
# --------------------------------------------------

def print_client_stats(
    client_name,
    local_weights,
    global_weights
):
    """
    Debug helper.
    """

    drift = model_distance(
        local_weights,
        global_weights
    )

    print(
        f"{client_name} Drift: "
        f"{drift:.4f}"
    )
