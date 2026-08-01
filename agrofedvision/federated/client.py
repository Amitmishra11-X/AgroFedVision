"""Flower client implementation for Keras models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import keras
import numpy as np

from agrofedvision.federated.checkpointing import FederatedCheckpointManager


try:
    import flwr as fl
except ImportError:  # pragma: no cover - optional runtime dependency
    fl = None


DatasetFactory = Callable[[], tuple[object, object]]
ModelFactory = Callable[[], keras.Model]


class KerasFlowerClient(fl.client.NumPyClient if fl is not None else object):
    def __init__(
        self,
        client_id: str,
        model_factory: ModelFactory,
        dataset_factory: DatasetFactory,
        checkpoint_dir: Path,
        local_epochs: int = 1,
        batch_size: int = 32,
    ) -> None:
        if fl is None:
            raise RuntimeError("Flower is not installed. Install flwr to use federated clients.")
        self.client_id = client_id
        self.model_factory = model_factory
        self.dataset_factory = dataset_factory
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.checkpoints = FederatedCheckpointManager(checkpoint_dir, f"client_{client_id}")
        self.model = self.checkpoints.load_model_if_available() or self.model_factory()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        train_ds, _ = self.dataset_factory()
        current_round = int(config.get("server_round", 0))
        epochs = int(config.get("local_epochs", self.local_epochs))
        history = self.model.fit(train_ds, epochs=epochs, verbose=0)
        self.checkpoints.save_model(
            self.model,
            {
                "round": current_round,
                "status": "fit_complete",
                "history": {key: [float(v) for v in values] for key, values in history.history.items()},
            },
        )
        return self.model.get_weights(), _dataset_size(train_ds), {"client_id": self.client_id}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        _, val_ds = self.dataset_factory()
        loss, accuracy = self.model.evaluate(val_ds, verbose=0)
        current_round = int(config.get("server_round", 0))
        self.checkpoints.save_model(
            self.model,
            {
                "round": current_round,
                "status": "evaluate_complete",
                "loss": float(loss),
                "accuracy": float(accuracy),
            },
        )
        return float(loss), _dataset_size(val_ds), {"accuracy": float(accuracy)}


def _dataset_size(dataset) -> int:
    cardinality = getattr(dataset, "cardinality", lambda: None)()
    try:
        value = int(cardinality.numpy())
        if value > 0:
            return value
    except Exception:
        pass
    total = 0
    for batch in dataset:
        features = batch[0]
        total += int(np.shape(features)[0])
    return total


def start_client(server_address: str, client: KerasFlowerClient) -> None:
    if fl is None:
        raise RuntimeError("Flower is not installed. Install flwr to start a client.")
    fl.client.start_numpy_client(server_address=server_address, client=client)
