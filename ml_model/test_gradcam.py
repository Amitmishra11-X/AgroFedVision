import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Register custom EfficientNetPreprocess
import image_encoder

from gradcam import generate_gradcam


MODEL_PATH = "results/agrofedvision_fusion_model.keras"

IMAGE_PATH = (
    r"D:\Download1\archive (5)\CNN256\CNN256\test"
    r"\black_mold\black_mold0007sa_jpg.rf.961dbef3120fa96177bbd1ae7f786898.jpg"
)


print("\n======================================")
print(" AgroFedVision Grad-CAM Test")
print("======================================\n")


print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print(
    "Model:",
    model.name
)

print(
    "Inputs:",
    [
        tuple(x.shape)
        for x in model.inputs
    ]
)


print("\nChecking target layer...")

top_conv = model.get_layer(
    "top_conv"
)

print(
    "Found:",
    top_conv.name
)

print(
    "Output shape:",
    top_conv.output.shape
)


# ------------------------------------------------------
# Dummy structured inputs for technical Grad-CAM test
# ------------------------------------------------------

sensor = np.zeros(
    (
        1,
        22,
    ),
    dtype=np.float32,
)

uav = np.zeros(
    (
        1,
        4,
    ),
    dtype=np.float32,
)


print("\nGenerating Grad-CAM...\n")

result = generate_gradcam(
    model=model,
    image_path=IMAGE_PATH,
    sensor=sensor,
    uav=uav,
)


print("\n======================================")
print(" SUCCESS")
print("======================================")

print(
    "Grad-CAM path:",
    result["path"]
)

print(
    "Class index:",
    result["class_index"]
)

print(
    "Target layer:",
    result["target_layer"]
)