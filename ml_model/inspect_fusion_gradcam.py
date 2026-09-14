import os
import sys

ML_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

if ML_MODEL_DIR not in sys.path:
    sys.path.insert(0, ML_MODEL_DIR)

import tensorflow as tf
import image_encoder


MODEL_PATH = "results/agrofedvision_fusion_model.keras"


def walk(layer, level=0):
    indent = "  " * level

    print(
        f"{indent}- {layer.name} "
        f"| {type(layer).__name__}"
    )

    for child in getattr(layer, "layers", []):
        walk(child, level + 1)


print("\n==============================================")
print(" AgroFedVision Grad-CAM Model Inspection")
print("==============================================\n")

print("Loading:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found.")
    sys.exit(1)


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("\nMODEL:")
print("Name:", model.name)

print("\nINPUTS:")
for i, inp in enumerate(model.inputs):
    print(
        i,
        "|",
        inp.name,
        "|",
        inp.shape
    )

print("\nTOP LEVEL LAYERS:")
print("----------------------------------------------")

for i, layer in enumerate(model.layers):
    print(
        i,
        "|",
        layer.name,
        "|",
        type(layer).__name__
    )


print("\nNESTED MODEL STRUCTURE:")
print("----------------------------------------------")

walk(model)


print("\n==============================================")
print("POSSIBLE IMAGE / EFFICIENTNET LAYERS")
print("==============================================")

for layer in model.layers:

    name = layer.name.lower()

    if (
        "image" in name
        or "efficient" in name
        or "conv" in name
    ):
        print(
            layer.name,
            "|",
            type(layer).__name__
        )

    for child in getattr(layer, "layers", []):

        child_name = child.name.lower()

        if (
            "image" in child_name
            or "efficient" in child_name
            or "conv" in child_name
        ):
            print(
                "  ",
                child.name,
                "|",
                type(child).__name__
            )


print("\n==============================================")
print("DONE")
print("==============================================")