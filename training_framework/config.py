"""
============================================================
AgroFedVision Crop Registry
============================================================
One place to manage every dataset path.
Change only this file when moving to another PC.
============================================================
"""

from pathlib import Path

# ==========================================================
# GUAVA
# ==========================================================

GUAVA = {

    "name": "guava",

    "train": Path(r"D:\Download1\archive (5)\CNN256\CNN256\train"),

    "valid": Path(r"D:\Download1\archive (5)\CNN256\CNN256\valid"),

    "test": Path(r"D:\Download1\archive (5)\CNN256\CNN256\test"),

    "num_classes": 9,

    "image_size": 224

}


# ==========================================================
# MAIZE / CORN
# ==========================================================

MAIZE = {

    "name": "maize",

    "train": Path(r"D:\Download1\archive (1)\Corn Disease detection"),

    "valid": None,

    "test": None,

    "class_names": [

        "Healthy corn",

        "Infected"

    ],

    "num_classes": 2,

    "image_size": 224

}


# ==========================================================
# PADDY
# ==========================================================

PADDY = {

    "name": "paddy",

    "train": Path(r"D:\Download1\archive\paddy-disease-classification\train_images"),

    "valid": None,

    "test": Path(r"D:\Download1\archive\paddy-disease-classification\test_images"),

    "num_classes": 10,

    "image_size": 224

}


# ==========================================================
# SENSOR DATASETS
# ==========================================================

SENSOR = {

    "new_dataset":

        Path(r"D:\Download1\archive (3)\sensor_Crop_Dataset (1).csv"),

    "old_dataset":

        Path(r"D:\Download1\Crop_Classification_dataset.xlsx")

}


# ==========================================================
# UAV DATASET
# ==========================================================

UAV = {

    "folder":

        Path(
            r"D:\Download1\Agriculture_Multispectral_Aerial"
            r"\Agriculture_Multispectral_Aerial"
            r"\Agri\Agri\Maize"
            r"\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"
        ),

    "feature_csv":

        Path(
            r"C:\Users\Lenovo\Desktop\AgroFedVision\uav_features.csv"
        )

}


# ==========================================================
# MASTER REGISTRY
# ==========================================================

CROP_REGISTRY = {

    "guava": GUAVA,

    "maize": MAIZE,

    "paddy": PADDY

}


# ==========================================================
# FUNCTIONS
# ==========================================================

def get_crop(name):

    name = name.lower()

    if name not in CROP_REGISTRY:

        raise ValueError(f"Unknown crop : {name}")

    return CROP_REGISTRY[name]


def get_sensor():

    return SENSOR


def get_uav():

    return UAV


def show_registry():

    print("\n")

    print("=" * 60)

    print("AgroFedVision Crop Registry")

    print("=" * 60)

    for crop, info in CROP_REGISTRY.items():

        print(f"\n{crop.upper()}")

        print("Train :", info["train"])

        print("Valid :", info["valid"])

        print("Test  :", info["test"])

    print("\nSensor Dataset :", SENSOR["new_dataset"])

    print("Old Sensor     :", SENSOR["old_dataset"])

    print("UAV Folder     :", UAV["folder"])