import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.preprocessing.image import load_img, img_to_array

sys.path.append(os.path.abspath("."))

from ml_model.image_encoder import EfficientNetPreprocess

MODEL_PATH = "results/agrofedvision_fusion_model.keras"

IMG_PATH = r"D:\Download1\Multi_Crop_Leaves_Disease\Multi_Crop_Leaves_Disease\Brinjal\Healthy_brinjal\Brinjal flower.jpg"

IMG_SIZE = 224

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "EfficientNetPreprocess": EfficientNetPreprocess
    }
)

print("Model loaded.")

# ---------------------------------------------------
# Load image
# ---------------------------------------------------

img = load_img(
    IMG_PATH,
    target_size=(IMG_SIZE, IMG_SIZE)
)

img_array = img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# ---------------------------------------------------
# Find last convolution layer
# ---------------------------------------------------

last_conv_layer = None

for layer in reversed(model.layers):
    try:
        output_shape = layer.output.shape

        if len(output_shape) == 4:
            last_conv_layer = layer.name
            break
    except:
        pass

print("Using Conv Layer:", last_conv_layer)

# ---------------------------------------------------
# Build GradCAM model
# ---------------------------------------------------

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer(last_conv_layer).output,
        model.output
    ]
)

# ---------------------------------------------------
# Dummy inputs
# ---------------------------------------------------

sensor_dummy = np.zeros((1, 22), dtype=np.float32)

uav_dummy = np.zeros((1, 4), dtype=np.float32)

# ---------------------------------------------------
# Compute GradCAM
# ---------------------------------------------------

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        {
            "sensor_input": sensor_dummy,
            "uav_input": uav_dummy,
            "image_input": img_batch
        }
    )

    pred_index = tf.argmax(predictions[0])

    class_channel = predictions[:, pred_index]

grads = tape.gradient(
    class_channel,
    conv_outputs
)

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

heatmap = heatmap.numpy()

heatmap = np.maximum(
    heatmap,
    0
)

heatmap = heatmap / (np.max(heatmap) + 1e-8)
# ---------------------------------------------------
# Create overlay
# ---------------------------------------------------

img_original = cv2.imread(IMG_PATH)

heatmap = cv2.resize(
    heatmap,
    (
        img_original.shape[1],
        img_original.shape[0]
    )
)

heatmap_uint8 = np.uint8(
    255 * heatmap
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    img_original,
    0.6,
    heatmap_color,
    0.4,
    0
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

cv2.imwrite(
    "results/gradcam_heatmap.jpg",
    heatmap_color
)

cv2.imwrite(
    "results/gradcam_overlay.jpg",
    overlay
)

print("\nSaved:")
print("results/gradcam_heatmap.jpg")
print("results/gradcam_overlay.jpg")

# ---------------------------------------------------
# Show
# ---------------------------------------------------

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
plt.title("GradCAM")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()