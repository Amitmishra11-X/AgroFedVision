"""Keras callbacks for fault-tolerant training."""

from __future__ import annotations

from pathlib import Path

import keras

from agrofedvision.training.config import Phase1Config, StageConfig
from agrofedvision.training.resume import ResumeState
from agrofedvision.utils.runtime import fold_name


class ResumeStateCallback(keras.callbacks.Callback):
    def __init__(
        self,
        resume_state: ResumeState,
        fold_index: int,
        stage_name: str,
        latest_model_path: Path,
    ) -> None:
        super().__init__()
        self.resume_state = resume_state
        self.fold_index = fold_index
        self.stage_name = stage_name
        self.latest_model_path = latest_model_path

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self.model.save(self.latest_model_path)
        self.resume_state.set_active(
            self.fold_index,
            self.stage_name,
            completed_epochs=epoch + 1,
        )


def stage_paths(paths: dict[str, Path], fold_index: int, stage_name: str) -> dict[str, Path]:
    fold = fold_name(fold_index)
    return {
        "best_model": paths["models"] / f"{fold}_{stage_name}_best.keras",
        "latest_model": paths["models"] / f"{fold}_{stage_name}_latest.keras",
        "checkpoint": paths["checkpoints"] / f"{fold}_{stage_name}_epoch_{{epoch:02d}}.keras",
        "backup": paths["backup"] / fold / stage_name,
        "tensorboard": paths["logs"] / fold / stage_name,
        "csv": paths["history"] / f"{fold}_{stage_name}_training_log.csv",
    }


def build_training_callbacks(
    paths: dict[str, Path],
    config: Phase1Config,
    stage: StageConfig,
    fold_index: int,
    resume_state: ResumeState,
) -> list[keras.callbacks.Callback]:
    spaths = stage_paths(paths, fold_index, stage.name)
    spaths["backup"].mkdir(parents=True, exist_ok=True)
    spaths["tensorboard"].mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=spaths["checkpoint"],
            monitor="val_accuracy",
            mode="max",
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=spaths["best_model"],
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.BackupAndRestore(
            backup_dir=spaths["backup"],
            save_freq="epoch",
            delete_checkpoint=False,
        ),
        keras.callbacks.TensorBoard(
            log_dir=spaths["tensorboard"],
            histogram_freq=0,
            update_freq="epoch",
        ),
        keras.callbacks.CSVLogger(
            filename=spaths["csv"],
            separator=",",
            append=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
        ResumeStateCallback(
            resume_state=resume_state,
            fold_index=fold_index,
            stage_name=stage.name,
            latest_model_path=spaths["latest_model"],
        ),
    ]
