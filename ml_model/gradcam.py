"""
gradcam.py
AgroFedVision - Grad-CAM for the image branch.

Compatible with the ACTUAL saved AgroFedVision model:

    AgroFedVision_Learnable_Gating
        |
        +-- EfficientNet image layers
        |       |
        |       +-- top_conv  <-- Grad-CAM target
        |
        +-- Sensor branch
        |
        +-- UAV branch
        |
        +-- Learnable modality gating
        |
        +-- Feature fusion
        |
        +-- Fusion classification head
        |
        +-- crop_health_output

Grad-CAM explains:
    Which regions of the leaf image influenced the
    predicted crop-health class?

SHAP explains:
    Sensor + UAV structured features.

IMPORTANT:
    The saved model is a FLAT Keras Functional graph.
    There is NO nested EfficientNetB0_ImageEncoder layer
    in the loaded model.

Therefore Grad-CAM directly uses:
    model.get_layer("top_conv")
"""

import os

import cv2
import numpy as np
import tensorflow as tf


# ==========================================================
# FIND GRAD-CAM TARGET
# ==========================================================

def _find_gradcam_layer(model):
    """
    Find the convolutional layer used for Grad-CAM.

    The actual saved AgroFedVision model contains:

        top_conv
        top_bn
        top_activation

    directly at the top level.
    """

    # ------------------------------------------------------
    # Preferred target
    # ------------------------------------------------------

    try:
        layer = model.get_layer(
            "top_conv"
        )

        print(
            "[Grad-CAM] Target layer:",
            layer.name
        )

        return layer

    except Exception:
        pass

    # ------------------------------------------------------
    # Fallback: search top-level Conv2D layers
    # ------------------------------------------------------

    for layer in reversed(model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            print(
                "[Grad-CAM] Fallback target layer:",
                layer.name
            )

            return layer

    raise ValueError(
        "Could not find a convolutional layer "
        "for Grad-CAM in AgroFedVision."
    )


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(
    image_path,
    img_size=224,
):
    """
    Load image for AgroFedVision.

    The saved model itself contains:
        efficientnet_preprocess

    Therefore we DO NOT apply EfficientNet preprocessing
    here. We only resize and convert to float32.
    """

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        image,
        (
            img_size,
            img_size,
        ),
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    image = tf.expand_dims(
        image,
        axis=0,
    )

    return image


# ==========================================================
# PREPARE STRUCTURED INPUTS
# ==========================================================

def _prepare_sensor(
    model,
    sensor,
):
    """
    Prepare sensor input.
    """

    sensor_dim = int(
        model.inputs[0].shape[-1]
    )

    if sensor is None:

        return np.zeros(
            (
                1,
                sensor_dim,
            ),
            dtype=np.float32,
        )

    sensor = np.asarray(
        sensor,
        dtype=np.float32,
    )

    sensor = sensor.reshape(
        1,
        -1,
    )

    if sensor.shape[1] != sensor_dim:

        raise ValueError(
            "Sensor feature dimension mismatch. "
            f"Expected {sensor_dim}, "
            f"received {sensor.shape[1]}."
        )

    return sensor


def _prepare_uav(
    model,
    uav,
):
    """
    Prepare UAV input.
    """

    uav_dim = int(
        model.inputs[1].shape[-1]
    )

    if uav is None:

        return np.zeros(
            (
                1,
                uav_dim,
            ),
            dtype=np.float32,
        )

    uav = np.asarray(
        uav,
        dtype=np.float32,
    )

    uav = uav.reshape(
        1,
        -1,
    )

    if uav.shape[1] != uav_dim:

        raise ValueError(
            "UAV feature dimension mismatch. "
            f"Expected {uav_dim}, "
            f"received {uav.shape[1]}."
        )

    return uav


# ==========================================================
# GRAD-CAM HEATMAP
# ==========================================================

def make_gradcam_heatmap(
    image,
    model,
    sensor=None,
    uav=None,
    pred_index=None,
):
    """
    Generate Grad-CAM heatmap.

    The model receives:

        sensor_input
        uav_input
        image_input

    Sensor and UAV remain fixed.

    Gradients are calculated with respect to
    the image branch convolutional feature map.
    """

    # ------------------------------------------------------
    # Verify model inputs
    # ------------------------------------------------------

    if len(model.inputs) != 3:

        raise ValueError(
            "Grad-CAM expects the 3-input AgroFedVision model."
        )

    # ------------------------------------------------------
    # Actual target layer
    # ------------------------------------------------------

    last_conv = _find_gradcam_layer(
        model
    )

    # ------------------------------------------------------
    # Prepare structured inputs
    # ------------------------------------------------------

    sensor = _prepare_sensor(
        model,
        sensor,
    )

    uav = _prepare_uav(
        model,
        uav,
    )

    # ------------------------------------------------------
    # Make sure image is float32
    # ------------------------------------------------------

    image = tf.cast(
        image,
        tf.float32,
    )

    # ======================================================
    # GRAD-CAM MODEL
    # ======================================================

    # IMPORTANT:
    #
    # The actual saved model is a flat Functional graph.
    #
    # Therefore top_conv.output is directly connected to
    # model.output.
    #
    # No nested image encoder is required.

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            last_conv.output,
            model.output,
        ],
        name="AgroFedVision_GradCAM",
    )

    # ======================================================
    # FORWARD + GRADIENT
    # ======================================================

    sensor = tf.convert_to_tensor(sensor, dtype=tf.float32)
    uav = tf.convert_to_tensor(uav, dtype=tf.float32)
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
        [sensor, uav, image],
        training=False
    )
        # --------------------------------------------------
        # Determine predicted class
        # --------------------------------------------------

        if pred_index is None:

            pred_index = int(
                tf.argmax(
                    predictions[0]
                ).numpy()
            )

        # --------------------------------------------------
        # Select predicted class probability
        # --------------------------------------------------

        class_channel = predictions[
            :,
            pred_index
        ]

    # ======================================================
    # GRADIENT WITH RESPECT TO CONVOLUTIONAL ACTIVATIONS
    # ======================================================

    grads = tape.gradient(
        class_channel,
        conv_outputs,
    )

    if grads is None:

        raise ValueError(
            "Gradients are None. "
            "The top_conv layer is not connected "
            "to the selected prediction."
        )

    # ======================================================
    # GLOBAL AVERAGE POOLING OF GRADIENTS
    # ======================================================

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(
            0,
            1,
            2,
        ),
    )

    # ------------------------------------------------------
    # Remove batch dimension
    # ------------------------------------------------------

    conv_outputs = conv_outputs[0]

    # ======================================================
    # WEIGHT FEATURE MAPS
    # ======================================================

    heatmap = tf.reduce_sum(
        conv_outputs
        * pooled_grads,
        axis=-1,
    )

    # ======================================================
    # RELU
    # ======================================================

    heatmap = tf.maximum(
        heatmap,
        0,
    )

    # ======================================================
    # NORMALIZATION
    # ======================================================

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value
        + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # Safety normalization.
    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    return (
        heatmap,
        pred_index,
    )


# ==========================================================
# OVERLAY HEATMAP
# ==========================================================

def overlay_heatmap(
    image_path,
    heatmap,
    output_path,
    alpha=0.40,
):
    """
    Overlay Grad-CAM heatmap over original image.
    """

    # ------------------------------------------------------
    # Read original image
    # ------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # ------------------------------------------------------
    # Resize heatmap
    # ------------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (
            image.shape[1],
            image.shape[0],
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # ------------------------------------------------------
    # Convert to 8-bit
    # ------------------------------------------------------

    heatmap_uint8 = np.uint8(
        255
        * np.clip(
            heatmap,
            0,
            1,
        )
    )

    # ------------------------------------------------------
    # Apply color map
    # ------------------------------------------------------

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    # ------------------------------------------------------
    # Blend with original
    # ------------------------------------------------------

    overlay = cv2.addWeighted(
        image,
        1.0 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    success = cv2.imwrite(
        output_path,
        overlay,
    )

    if not success:

        raise IOError(
            f"Could not save Grad-CAM image: "
            f"{output_path}"
        )

    return output_path


# ==========================================================
# PUBLIC GRAD-CAM FUNCTION
# ==========================================================

def generate_gradcam(
    model,
    image_path,
    sensor=None,
    uav=None,
    img_size=224,
    save_dir="results/gradcam",
):
    """
    Generate and save a Grad-CAM overlay.

    Returns:
        Dictionary containing:

            path
            class_index
            target_layer
    """

    # ------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Load image
    # ------------------------------------------------------

    image = preprocess_image(
        image_path,
        img_size,
    )

    # ------------------------------------------------------
    # Generate heatmap
    # ------------------------------------------------------

    heatmap, pred_index = make_gradcam_heatmap(
        image=image,
        model=model,
        sensor=sensor,
        uav=uav,
    )

    # ------------------------------------------------------
    # Output filename
    # ------------------------------------------------------

    filename = os.path.splitext(
        os.path.basename(
            image_path
        )
    )[0]

    output_path = os.path.join(
        save_dir,
        filename
        + "_fusion_gradcam.jpg",
    )

    # ------------------------------------------------------
    # Create overlay
    # ------------------------------------------------------

    overlay_heatmap(
        image_path=image_path,
        heatmap=heatmap,
        output_path=output_path,
    )

    # ------------------------------------------------------
    # Get class name if possible
    # ------------------------------------------------------

    class_name = None

    try:

        predictions = model(
            [
                _prepare_sensor(
                    model,
                    sensor,
                ),
                _prepare_uav(
                    model,
                    uav,
                ),
                image,
            ],
            training=False,
        )

        class_index = int(
            tf.argmax(
                predictions[0]
            ).numpy()
        )

        if class_index == pred_index:

            class_name = str(
                class_index
            )

    except Exception:

        class_name = str(
            pred_index
        )

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------

    print(
        f"[Grad-CAM] Target layer : top_conv"
    )

    print(
        f"[Grad-CAM] Class index   : {pred_index}"
    )

    print(
        f"[Grad-CAM] Saved         : {output_path}"
    )

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

    return {
        "path": output_path,
        "class_index": int(pred_index),
        "target_layer": "top_conv",
        "class_name": class_name,
    }