import os

# ======================================================
# Dataset
# ======================================================

DATASET_PATH = r"D:\Download1\archive\paddy-disease-classification"

TRAIN_CSV = os.path.join(
    DATASET_PATH,
    "train.csv"
)

TRAIN_IMAGES = os.path.join(
    DATASET_PATH,
    "train_images"
)

# ======================================================
# Image
# ======================================================

IMAGE_SIZE = (224,224)

CHANNELS = 3

# ======================================================
# Training
# ======================================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 3e-4

VALIDATION_SPLIT = 0.20

SEED = 42

# ======================================================
# Output
# ======================================================

OUTPUT_DIR = "outputs"

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

LOG_DIR = os.path.join(
    OUTPUT_DIR,
    "logs"
)

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

for folder in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    FIGURE_DIR,
    REPORT_DIR,
]:
    os.makedirs(folder,exist_ok=True)