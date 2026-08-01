"""
gradcam.py

AgroFedVision - Explainable AI using Grad-CAM

Works with EfficientNet-based crop disease models.

Author: Amit Mishra
"""

import os
import cv2
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model):
    """
    Recursively find the last Conv2D layer in the model,
    including nested models (e.g. EfficientNet backbone).
    """

    def search(layers):
        for layer in reversed(layers):

            # Nested model
            if hasattr(layer, "layers"):
                found = search(layer.layers)
                if found is not None:
                    return found

            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        return None

    layer_name = search(model.layers)

    if layer_name is None:
        raise ValueError("No Conv2D layer found in model.")

    return layer_name


def preprocess_image(image_path, img_size):
    """
    Load image for prediction.
    """

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32)

    return tf.expand_dims(img, axis=0)


def make_gradcam_heatmap(image, model, pred_index=None):
    """
    Grad-CAM for AgroFedVision models where EfficientNet is wrapped
    inside the Functional layer 'ImageEncoder'.
    """

    # Backbone
    encoder = model.get_layer("ImageEncoder")

    # Last convolution layer
    last_conv = encoder.get_layer("top_conv")

    # Build a model that outputs:
    #   1. top_conv activations
    #   2. final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            last_conv.output,
            model.output
        ],
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def overlay_heatmap(
        image_path,
        heatmap,
        output_path,
        alpha=0.4):
    """
    Overlay heatmap on original image.
    """

    image = cv2.imread(image_path)

    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    cv2.imwrite(output_path, overlay)


def generate_gradcam(
        model,
        image_path,
        img_size=224,
        save_dir="results/gradcam"):
    """
    Main function.

    Returns:
        output image path
    """

    os.makedirs(save_dir, exist_ok=True)

    image = preprocess_image(
        image_path,
        img_size
    )



    heatmap = make_gradcam_heatmap(
        image,
        model,
        
    )

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        save_dir,
        filename + "_gradcam.jpg"
    )

    overlay_heatmap(
        image_path,
        heatmap,
        output_path
    )

    return output_path