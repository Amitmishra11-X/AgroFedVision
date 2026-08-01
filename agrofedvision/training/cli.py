"""Command-line interface for Phase 1 training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from agrofedvision.training.config import DEFAULT_PHASE1_CONFIG, Phase1Config
from agrofedvision.training.runner import run_phase1_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AgroFedVision Phase 1 EfficientNetB0 image classifier."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Phase1Config:
    updates = {}
    for key in ("data_root", "train_csv", "image_root", "results_dir", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    if args.no_cache:
        updates["cache_datasets"] = False
    if args.no_mixed_precision:
        updates["use_mixed_precision"] = False
    return replace(DEFAULT_PHASE1_CONFIG, **updates)


def main() -> None:
    run_phase1_training(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
