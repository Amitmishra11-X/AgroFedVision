import numpy as np


def add_dp_noise(
    weights,
    sigma=0.001
):
    """
    Add Gaussian noise to model weights.

    Differential Privacy:
        w = w + N(0, sigma²)
    """

    noisy_weights = []

    for w in weights:

        noise = np.random.normal(
            loc=0.0,
            scale=sigma,
            size=w.shape
        )

        noisy_weights.append(
            w + noise
        )

    return noisy_weights



    """
    Gradient/weight clipping.
    """

    clipped = []

    for w in weights:

        norm = np.linalg.norm(w)

        if norm > clip_norm:

            w = w * (
                clip_norm / norm
            )

        clipped.append(w)

    return clipped


def apply_dp(weights, sigma=0.00005):
    noisy_weights = []

    for w in weights:
        noise = np.random.normal(
            0,
            sigma,
            w.shape
        )

        noisy_weights.append(
            w + noise
        )

    return noisy_weights