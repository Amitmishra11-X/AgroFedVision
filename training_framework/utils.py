from pathlib import Path
import random
import numpy as np
import tensorflow as tf


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)


def create_folders():

    folders = [

        "outputs",

        "checkpoints",

        "logs",

        "benchmark",

        "outputs/plots",

        "outputs/reports",

        "outputs/history"

    ]

    for folder in folders:

        Path(folder).mkdir(parents=True, exist_ok=True)