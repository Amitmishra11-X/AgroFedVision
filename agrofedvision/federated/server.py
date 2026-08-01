"""Flower server strategies with checkpoint recovery."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import keras

from agrofedvision.federated.checkpointing import FederatedCheckpointManager


try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
except ImportError:  # pragma: no cover - optional runtime dependency
    fl = None
    ndarrays_to_parameters = None
    parameters_to_ndarrays = None


ModelFactory = Callable[[], keras.Model]


class CheckpointingFedAvg(fl.server.strategy.FedAvg if fl is not None else object):
    def __init__(
        self,
        model_factory: ModelFactory,
        checkpoint_dir: Path,
        **kwargs,
    ) -> None:
        if fl is None:
            raise RuntimeError("Flower is not installed. Install flwr to use federated server.")
        self.model_factory = model_factory
        self.checkpoints = FederatedCheckpointManager(checkpoint_dir, "server")
        model = self.checkpoints.load_model_if_available() or self.model_factory()
        kwargs.setdefault("initial_parameters", ndarrays_to_parameters(model.get_weights()))
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        parameters, metrics = aggregated
        if parameters is not None:
            model = self.model_factory()
            model.set_weights(parameters_to_ndarrays(parameters))
            self.checkpoints.save_model(
                model,
                {
                    "round": int(server_round),
                    "status": "aggregate_fit_complete",
                    "metrics": dict(metrics or {}),
                    "failures": len(failures),
                },
            )
        return aggregated


class CheckpointingFedProx(fl.server.strategy.FedProx if fl is not None else object):
    def __init__(
        self,
        model_factory: ModelFactory,
        checkpoint_dir: Path,
        proximal_mu: float = 0.01,
        **kwargs,
    ) -> None:
        if fl is None:
            raise RuntimeError("Flower is not installed. Install flwr to use federated server.")
        self.model_factory = model_factory
        self.checkpoints = FederatedCheckpointManager(checkpoint_dir, "server")
        model = self.checkpoints.load_model_if_available() or self.model_factory()
        kwargs.setdefault("initial_parameters", ndarrays_to_parameters(model.get_weights()))
        kwargs.setdefault("proximal_mu", proximal_mu)
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        parameters, metrics = aggregated
        if parameters is not None:
            model = self.model_factory()
            model.set_weights(parameters_to_ndarrays(parameters))
            self.checkpoints.save_model(
                model,
                {
                    "round": int(server_round),
                    "status": "aggregate_fit_complete",
                    "strategy": "fedprox",
                    "metrics": dict(metrics or {}),
                    "failures": len(failures),
                },
            )
        return aggregated


def build_strategy(
    strategy_name: str,
    model_factory: ModelFactory,
    checkpoint_dir: Path,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    proximal_mu: float = 0.01,
):
    if fl is None:
        raise RuntimeError("Flower is not installed. Install flwr to use federated server.")

    normalized = strategy_name.lower()
    common_kwargs = {
        "model_factory": model_factory,
        "checkpoint_dir": checkpoint_dir,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
    }
    if normalized == "fedavg":
        return CheckpointingFedAvg(**common_kwargs)
    if normalized == "fedprox":
        return CheckpointingFedProx(proximal_mu=proximal_mu, **common_kwargs)
    raise ValueError(f"Unsupported federated strategy: {strategy_name}")


def start_server(
    server_address: str,
    strategy_name: str,
    model_factory: ModelFactory,
    checkpoint_dir: Path,
    num_rounds: int,
) -> None:
    if fl is None:
        raise RuntimeError("Flower is not installed. Install flwr to start a server.")
    strategy = build_strategy(strategy_name, model_factory, checkpoint_dir)
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
