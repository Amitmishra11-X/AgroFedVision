import numpy as np

def aggregate_weights(client_weights):

    avg_weights = []

    for weights in zip(*client_weights):

        avg_weights.append(
            np.mean(weights, axis=0)
        )

    return avg_weights