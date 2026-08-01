import os
import sys
import tensorflow as tf

sys.path.append("ml_model")

from image_encoder import EfficientNetPreprocess

from client_train import train_client


CLIENTS = [
    "data/clients/farm_a.csv",
    "data/clients/farm_b.csv",
    "data/clients/farm_c.csv",
    "data/clients/farm_d.csv"
]

GLOBAL_MODEL = "results/agrofz_model.h5"