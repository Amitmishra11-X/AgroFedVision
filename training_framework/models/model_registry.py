"""
============================================================
AgroFedVision Research Framework
Universal Model Registry
============================================================

This file is responsible for

1. Registering every architecture
2. Building models
3. Printing model metadata
4. Returning architecture information

Future Models:
---------------
CNN
MobileNetV2
VGG16
ResNet50
ResNet34
EfficientNetB0
DenseNet121
Xception
ConvNeXt
ViT
Swin Transformer

============================================================
"""

from .cnn import build_cnn
from .mobilenet import build_mobilenet
from .vgg16 import build_vgg16
from .resnet50 import build_resnet50
from .efficientnet import build_efficientnet
from .densenet import build_densenet
from .xception import build_xception


# ============================================================
# Model Registry
# ============================================================

MODEL_REGISTRY = {

    "cnn":{

        "builder":build_cnn,

        "paper":"Custom CNN",

        "family":"CNN",

        "year":2025,

        "pretrained":False,

        "input_size":224

    },

    "mobilenet":{

        "builder":build_mobilenet,

        "paper":"MobileNetV2",

        "family":"MobileNet",

        "year":2018,

        "pretrained":True,

        "input_size":224

    },

    "vgg16":{

        "builder":build_vgg16,

        "paper":"VGG16",

        "family":"VGG",

        "year":2014,

        "pretrained":True,

        "input_size":224

    },

    "resnet50":{

        "builder":build_resnet50,

        "paper":"Deep Residual Learning",

        "family":"ResNet",

        "year":2015,

        "pretrained":True,

        "input_size":224

    },

    "xception":{

        "builder":build_xception,

        "paper":"Xception",

        "family":"CNN",

        "year":2017,

        "pretrained":True,

        "input_size":299

    },

    "efficientnet":{

        "builder":build_efficientnet,

        "paper":"EfficientNetB0",

        "family":"EfficientNet",

        "year":2019,

        "pretrained":True,

        "input_size":224

    },

    "densenet":{

        "builder":build_densenet,

        "paper":"DenseNet121",

        "family":"DenseNet",

        "year":2017,

        "pretrained":True,

        "input_size":224

    }

}


# ============================================================
# Show Available Models
# ============================================================

def show_models():

    print("\n")

    print("="*60)

    print("Available Models")

    print("="*60)

    for name in MODEL_REGISTRY:

        info = MODEL_REGISTRY[name]

        print(

            f"{name:15}"

            f"{info['paper']:25}"

            f"{info['year']}"

        )

    print("="*60)


# ============================================================
# Build Model
# ============================================================

def build_model(

    model_name,

    num_classes

):

    model_name=model_name.lower()

    if model_name not in MODEL_REGISTRY:

        show_models()

        raise ValueError(

            f"\nUnknown Model : {model_name}"

        )

    info=MODEL_REGISTRY[model_name]

    print("\n")

    print("="*60)

    print("Building Model")

    print("="*60)

    print(f"Architecture : {model_name}")

    print(f"Paper        : {info['paper']}")

    print(f"Family       : {info['family']}")

    print(f"Published    : {info['year']}")

    print(f"ImageNet     : {info['pretrained']}")

    print(f"Input Size   : {info['input_size']}")

    print("="*60)

    model=info["builder"](num_classes)

    return model


# ============================================================
# Get Metadata
# ============================================================

def get_model_info(

    model_name

):

    model_name=model_name.lower()

    return MODEL_REGISTRY[model_name]