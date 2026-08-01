"""
image_batch_predictor.py

Supports:
- Single image prediction
- Multiple image prediction
- Probability averaging
"""

import os
import numpy as np
from tensorflow.keras.preprocessing import image

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


def load_images_from_folder(folder):
    """
    Returns all image paths inside folder.
    """

    image_paths = []

    for file in sorted(os.listdir(folder)):
        if file.lower().endswith(SUPPORTED_EXTENSIONS):
            image_paths.append(os.path.join(folder, file))

    return image_paths


def predict_single_image(model, img_path, img_size):
    """
    Predict one image.

    Returns
    -------
    probs : ndarray
    """

    img = image.load_img(
        img_path,
        target_size=(img_size, img_size)
    )

    arr = image.img_to_array(img)

    arr = arr.astype("float32") / 255.0

    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]

    return probs


def predict_folder(
    model,
    folder,
    img_size,
):
    """
    Predict every image in folder.

    Returns
    -------
    average probabilities

    per-image predictions

    best image
    """

    image_paths = load_images_from_folder(folder)

    if len(image_paths) == 0:
        raise ValueError(
            f"No images found inside {folder}"
        )

    all_probs = []

    per_image = []

    best_conf = -1

    best_image = None

    for img_path in image_paths:

        probs = predict_single_image(
            model,
            img_path,
            img_size,
        )

        conf = float(np.max(probs))

        all_probs.append(probs)

        per_image.append({

            "image": img_path,

            "confidence": conf,

            "probabilities": probs,

        })

        if conf > best_conf:

            best_conf = conf

            best_image = img_path

    avg_probs = np.mean(
        np.stack(all_probs),
        axis=0,
    )

    return {

        "average_probabilities": avg_probs,

        "best_image": best_image,

        "best_confidence": best_conf,

        "per_image": per_image,

        "num_images": len(image_paths),

    }