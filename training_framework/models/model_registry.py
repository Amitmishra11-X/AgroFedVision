"""
============================================================
AgroFedVision Research Framework
Universal Model Registry
============================================================
"""

from .cnn import build_cnn
from .mobilenet import build_mobilenet
from .vgg16 import build_vgg16
from .resnet50 import build_resnet50
from .efficientnet import build_efficientnet
from .densenet import build_densenet
from .xception import build_xception

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess


# ============================================================
# Model Registry
# ============================================================

MODEL_REGISTRY = {

    "cnn": {

        "builder": build_cnn,
        "paper": "Custom CNN",
        "family": "CNN",
        "year": 2025,
        "pretrained": False,
        "input_size": 224,
        "preprocess": lambda x: x / 255.0

    },

    "mobilenet": {

        "builder": build_mobilenet,
        "paper": "MobileNetV2",
        "family": "MobileNet",
        "year": 2018,
        "pretrained": True,
        "input_size": 224,
        "preprocess": mobilenet_preprocess

    },

    "vgg16": {

        "builder": build_vgg16,
        "paper": "VGG16",
        "family": "VGG",
        "year": 2014,
        "pretrained": True,
        "input_size": 224,
        "preprocess": vgg16_preprocess

    },

    "resnet50": {

        "builder": build_resnet50,
        "paper": "Deep Residual Learning",
        "family": "ResNet",
        "year": 2015,
        "pretrained": True,
        "input_size": 224,
        "preprocess": resnet50_preprocess

    },

    "efficientnet": {

        "builder": build_efficientnet,
        "paper": "EfficientNetB0",
        "family": "EfficientNet",
        "year": 2019,
        "pretrained": True,
        "input_size": 224,
        "preprocess": efficientnet_preprocess

    },

    "densenet": {

        "builder": build_densenet,
        "paper": "DenseNet121",
        "family": "DenseNet",
        "year": 2017,
        "pretrained": True,
        "input_size": 224,
        "preprocess": densenet_preprocess

    },

    "xception": {

        "builder": build_xception,
        "paper": "Xception",
        "family": "CNN",
        "year": 2017,
        "pretrained": True,
        "input_size": 299,
        "preprocess": xception_preprocess

    }

}


# ============================================================
# Show Available Models
# ============================================================

def show_models():

    print()
    print("=" * 60)
    print("Available Models")
    print("=" * 60)

    for name, info in MODEL_REGISTRY.items():

        print(f"{name:15}{info['paper']:25}{info['year']}")

    print("=" * 60)


# ============================================================
# Build Model
# ============================================================

def build_model(model_name, num_classes):

    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:

        show_models()

        raise ValueError(f"Unknown Model : {model_name}")

    info = MODEL_REGISTRY[model_name]

    print()
    print("=" * 60)
    print("Building Model")
    print("=" * 60)
    print(f"Architecture : {model_name}")
    print(f"Paper        : {info['paper']}")
    print(f"Family       : {info['family']}")
    print(f"Published    : {info['year']}")
    print(f"ImageNet     : {info['pretrained']}")
    print(f"Input Size   : {info['input_size']}")
    print("=" * 60)

    model = info["builder"](num_classes)

    return {

        "model": model,

        "image_size": info["input_size"],

        "preprocess": info["preprocess"],

        "metadata": info

    }


# ============================================================
# Get Metadata
# ============================================================

def get_model_info(model_name):

    return MODEL_REGISTRY[model_name.lower()]