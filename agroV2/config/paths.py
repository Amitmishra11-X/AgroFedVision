"""
Centralized project paths.

Every module imports paths from here.
Never hardcode file paths anywhere else.
"""

from pathlib import Path

# -------------------------------------------------------
# Project Root
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -------------------------------------------------------
# Data
# -------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

LEAF_DATA_DIR = DATA_DIR / "leaf"

SENSOR_DATA_DIR = DATA_DIR / "sensor"

UAV_DATA_DIR = DATA_DIR / "uav"

# -------------------------------------------------------
# Models
# -------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

PRETRAINED_DIR = MODELS_DIR / "pretrained"

# -------------------------------------------------------
# Outputs
# -------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "outputs"

RESULTS_DIR = OUTPUT_DIR / "results"

LOG_DIR = OUTPUT_DIR / "logs"

PLOTS_DIR = OUTPUT_DIR / "plots"

REPORT_DIR = OUTPUT_DIR / "reports"

# -------------------------------------------------------
# Config
# -------------------------------------------------------

CONFIG_DIR = PROJECT_ROOT / "configs"

# -------------------------------------------------------
# Create folders automatically
# -------------------------------------------------------

ALL_DIRS = [

    DATA_DIR,

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

    LEAF_DATA_DIR,

    SENSOR_DATA_DIR,

    UAV_DATA_DIR,

    MODELS_DIR,

    CHECKPOINT_DIR,

    PRETRAINED_DIR,

    OUTPUT_DIR,

    RESULTS_DIR,

    LOG_DIR,

    PLOTS_DIR,

    REPORT_DIR,

]

for directory in ALL_DIRS:
    directory.mkdir(parents=True, exist_ok=True)