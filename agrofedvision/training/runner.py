"""Phase 1 centralized training runner."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from agrofedvision.datasets.image_manifest import (
    load_image_classification_manifest,
    save_manifest_summary,
)
from agrofedvision.datasets.tf_data import build_image_dataset
from agrofedvision.evaluation.classification import (
    evaluate_classifier,
    plot_training_history,
    save_history_from_csv,
)
from agrofedvision.models.efficientnet_image_classifier import (
    build_efficientnet_b0_classifier,
    configure_classifier_training,
    configure_fine_tuning,
)
from agrofedvision.training.callbacks import build_training_callbacks, stage_paths
from agrofedvision.training.config import Phase1Config, StageConfig
from agrofedvision.training.resume import ResumeState
from agrofedvision.utils.json_io import write_json
from agrofedvision.utils.runtime import (
    configure_gpu_memory_growth,
    configure_mixed_precision,
    ensure_results_tree,
    fold_name,
    set_reproducibility,
)


def compile_classifier(
    model: keras.Model,
    learning_rate: float,
) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def _load_model(path: Path) -> keras.Model | None:
    if path.exists():
        return keras.models.load_model(path)
    return None


def _prepare_stage_model(
    config: Phase1Config,
    paths: dict[str, Path],
    fold_index: int,
    stage: StageConfig,
    initial_epoch: int,
) -> keras.Model:
    spaths = stage_paths(paths, fold_index, stage.name)
    model = _load_model(spaths["latest_model"]) if initial_epoch > 0 else None

    if model is None and stage.name == "stage2":
        stage1_paths = stage_paths(paths, fold_index, "stage1")
        model = _load_model(stage1_paths["best_model"]) or _load_model(
            stage1_paths["latest_model"]
        )

    if model is None:
        model = build_efficientnet_b0_classifier(config)

    if stage.name == "stage1":
        configure_classifier_training(model)
    elif stage.name == "stage2" and initial_epoch == 0:
        configure_fine_tuning(model, config)

    if initial_epoch == 0 or not getattr(model, "optimizer", None):
        compile_classifier(model, stage.learning_rate)
    return model


def _train_stage(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: Phase1Config,
    stage: StageConfig,
    fold_index: int,
    paths: dict[str, Path],
    resume_state: ResumeState,
    initial_epoch: int,
) -> None:
    if initial_epoch >= stage.epochs:
        return

    resume_state.set_active(fold_index, stage.name, initial_epoch)
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=stage.epochs,
        callbacks=build_training_callbacks(paths, config, stage, fold_index, resume_state),
        verbose=1,
    )

    latest_model = stage_paths(paths, fold_index, stage.name)["latest_model"]
    if not latest_model.exists():
        model.save(latest_model)
    resume_state.mark_stage_complete(fold_index, stage.name)


def run_phase1_training(config: Phase1Config) -> None:
    set_reproducibility(config.seed)
    configure_gpu_memory_growth()
    configure_mixed_precision(config.use_mixed_precision)

    paths = ensure_results_tree(config.results_dir)
    resume_state = ResumeState(config.results_dir / "resume_state.json")

    manifest = load_image_classification_manifest(config)
    save_manifest_summary(manifest, config.results_dir)
    write_json(paths["summary"] / "phase1_config.json", _serializable_config(config))

    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    fold_metrics: dict[str, dict[str, float]] = {}

    for fold_index, (train_idx, val_idx) in enumerate(
        splitter.split(manifest.image_paths, manifest.labels)
    ):
        if resume_state.is_fold_complete(fold_index):
            continue

        fold = fold_name(fold_index)
        write_json(
            paths["summary"] / f"{fold}_split.json",
            {
                "train_size": int(len(train_idx)),
                "validation_size": int(len(val_idx)),
                "class_names": manifest.class_names,
            },
        )

        train_ds = build_image_dataset(
            manifest.image_paths[train_idx],
            manifest.labels[train_idx],
            config,
            training=True,
        )
        val_ds = build_image_dataset(
            manifest.image_paths[val_idx],
            manifest.labels[val_idx],
            config,
            training=False,
        )

        model: keras.Model | None = None
        for stage in config.stages:
            if resume_state.is_stage_complete(fold_index, stage.name):
                continue
            initial_epoch = resume_state.epoch_for(fold_index, stage.name)
            model = _prepare_stage_model(
                config,
                paths,
                fold_index,
                stage,
                initial_epoch,
            )
            _train_stage(
                model,
                train_ds,
                val_ds,
                config,
                stage,
                fold_index,
                paths,
                resume_state,
                initial_epoch,
            )

        if model is None:
            model = keras.models.load_model(
                stage_paths(paths, fold_index, "stage2")["latest_model"]
            )

        final_model_path = paths["models"] / f"{fold}_final.keras"
        model.save(final_model_path)

        history_csvs = [stage_paths(paths, fold_index, stage.name)["csv"] for stage in config.stages]
        history_path = paths["history"] / f"{fold}_history.json"
        save_history_from_csv(history_csvs, history_path)
        plot_training_history(history_path, paths["plots"], fold_index)

        metrics = evaluate_classifier(
            model,
            val_ds,
            manifest.labels[val_idx],
            manifest.class_names,
            paths,
            fold_index,
        )
        fold_metrics[fold] = metrics
        resume_state.mark_fold_complete(fold_index)

    if fold_metrics:
        write_json(paths["summary"] / "cross_validation_metrics.json", fold_metrics)
    resume_state.mark_complete()


def _serializable_config(config: Phase1Config) -> dict[str, object]:
    payload = {}
    for key, value in config.__dict__.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
        elif hasattr(value, "__dict__"):
            payload[key] = value.__dict__
        else:
            payload[key] = value
    return payload
