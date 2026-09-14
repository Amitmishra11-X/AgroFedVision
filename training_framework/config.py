"""
============================================================
AgroFedVision Research Framework
Configuration File
============================================================

Author : Amit Mishra
Project : AgroFedVision

Only modify this file when changing

1. Dataset
2. Training Parameters
3. Paths
4. Models
5. Output Directories

============================================================
"""

import os
import tensorflow as tf

# ==========================================================
# PROJECT
# ==========================================================

PROJECT_NAME = "AgroFedVision"

VERSION = "2.0"

SEED = 42

# ==========================================================
# CURRENT DATASET
# ==========================================================

CURRENT_DATASET = "guava"

# Available:
#
# guava
# maize
# paddy

# ==========================================================
# DATASET REGISTRY
# ==========================================================

DATASETS = {

    "guava": {

        "train": r"D:\Download1\archive (5)\CNN256\CNN256\train",

        "valid": r"D:\Download1\archive (5)\CNN256\CNN256\valid",

        "test": r"D:\Download1\archive (5)\CNN256\CNN256\test",

        "num_classes": 9,

        "image_size": 224,

        "models": [

            "vgg16"  ,



        ]

    },

    "maize": {

        "train": r"D:\Download1\archive (1)\Corn Disease detection",

        "valid": None,

        "test": None,

        "num_classes": 2,

        "image_size": 224,

        "models": [

            "cnn",

            "mobilenet",

            "vgg16",

            "resnet50",

            "xception",

            "efficientnet",

            "densenet"

        ]

    },

    "paddy": {

        "train": r"D:\Download1\archive\paddy-disease-classification\train_images",

        "valid": None,

        "test": None,
        
        "num_classes": 10,

        "image_size": 224,

        "models": [

            "cnn",

            "mobilenet",

            "vgg16",

            "resnet50",

            "xception",

            "efficientnet",

            "densenet"

        ]

    }

}

# ==========================================================
# SENSOR DATASETS
# ==========================================================

SENSOR_DATASET = {

    "new_dataset":

        r"D:\Download1\archive (3)\sensor_Crop_Dataset (1).csv",

    "old_dataset":

        r"D:\Download1\Crop_Classification_dataset.xlsx"

}

# ==========================================================
# UAV DATASET
# ==========================================================

UAV_DATASET = {

    "folder":

        r"D:\Download1\Agriculture_Multispectral_Aerial"
        r"\Agriculture_Multispectral_Aerial"
        r"\Agri\Agri\Maize"
        r"\maize_season4_RededgeMultispectral_20200125_10m_flight1\000",

    "feature_csv":

        r"C:\Users\Lenovo\Desktop\AgroFedVision\uav_features.csv"

}

# ==========================================================
# ACTIVE DATASET
# ==========================================================

CURRENT = DATASETS[CURRENT_DATASET]

TRAIN_DIR = CURRENT["train"]

VAL_DIR = CURRENT["valid"]

TEST_DIR = CURRENT["test"]

NUM_CLASSES = CURRENT["num_classes"]

IMAGE_SIZE = CURRENT["image_size"]

MODELS = CURRENT["models"]

# ==========================================================
# TRAINING
# ==========================================================

EPOCHS = 50

RESUME_TRAINING = True

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

OPTIMIZER = "adam"

LOSS = "sparse_categorical_crossentropy"

METRICS = [

    "accuracy"

]

# ==========================================================
# DATASET
# ==========================================================

SHUFFLE = True

BUFFER_SIZE = 1000

VALIDATION_SPLIT = 0.10

TEST_SPLIT = 0.10

AUTO_CREATE_VALIDATION = True

AUTO_SPLIT_DATASET = True
# ==========================================================
# PREPROCESSING
# ==========================================================

USE_DUPLICATE_REMOVER = True

USE_DATASET_SPLITTER = True

# ==========================================================
# DUPLICATE CHECK
# ==========================================================

CHECK_DUPLICATES = True

REMOVE_DUPLICATES = True

OVERWRITE_CLEAN_DATASET = False

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

USE_AUGMENTATION = True

ROTATION = 20

WIDTH_SHIFT = 0.20

HEIGHT_SHIFT = 0.20

ZOOM = 0.20

SHEAR = 0.20

HORIZONTAL_FLIP = True

VERTICAL_FLIP = False

# ==========================================================
# CALLBACKS
# ==========================================================

EARLY_STOPPING = True

PATIENCE = 10

SAVE_BEST_ONLY = True

REDUCE_LR = True

REDUCE_FACTOR = 0.2

REDUCE_PATIENCE = 5

# ==========================================================
# OUTPUTS
# ==========================================================

OUTPUT_DIR = "outputs"

MODEL_DIR = os.path.join(

    OUTPUT_DIR,

    CURRENT_DATASET,

    "models"

)

REPORT_DIR = os.path.join(

    OUTPUT_DIR,

    CURRENT_DATASET,

    "reports"

)

PLOT_DIR = os.path.join(

    OUTPUT_DIR,

    CURRENT_DATASET,

    "plots"

)

CHECKPOINT_DIR = os.path.join(

    OUTPUT_DIR,

    CURRENT_DATASET,

    "checkpoints"

)

BENCHMARK_DIR = os.path.join(

    OUTPUT_DIR,

    CURRENT_DATASET,

    "benchmark"

)

LOG_DIR = os.path.join(

    OUTPUT_DIR,

    CURRENT_DATASET,

    "logs"


)
HISTORY_DIR = os.path.join(

    OUTPUT_DIR,

    CURRENT_DATASET,

    "history"

)
# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

for folder in [

    OUTPUT_DIR,

    MODEL_DIR,

    REPORT_DIR,

    PLOT_DIR,

    CHECKPOINT_DIR,

    BENCHMARK_DIR,

    LOG_DIR,

    HISTORY_DIR

]:

    os.makedirs(folder, exist_ok=True)

# ==========================================================
# GPU
# ==========================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(

                gpu,

                True

            )

    except Exception:

        pass

# ==========================================================
# SHOW CONFIGURATION
# ==========================================================

def show_config():

    print("\n")

    print("=" * 60)

    print("AgroFedVision Configuration")

    print("=" * 60)

    print(f"Dataset        : {CURRENT_DATASET}")

    print(f"Train Path     : {TRAIN_DIR}")

    print(f"Validation     : {VAL_DIR}")

    print(f"Test Path      : {TEST_DIR}")

    print(f"Classes        : {NUM_CLASSES}")

    print(f"Image Size     : {IMAGE_SIZE}")

    print(f"Epochs         : {EPOCHS}")

    print(f"Batch Size     : {BATCH_SIZE}")

    print(f"Learning Rate  : {LEARNING_RATE}")

    print(f"Models         : {', '.join(MODELS)}")

    print("=" * 60)