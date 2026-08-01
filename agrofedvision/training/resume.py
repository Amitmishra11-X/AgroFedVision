"""Persistent training resume state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agrofedvision.utils.json_io import read_json, write_json


def initial_resume_state() -> dict[str, Any]:
    return {
        "current_fold": 0,
        "current_stage": "stage1",
        "stage_epoch": 0,
        "completed_folds": [],
        "completed_stages": {},
        "status": "not_started",
    }


class ResumeState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = read_json(path, initial_resume_state())

    def save(self) -> None:
        write_json(self.path, self.state)

    def set_active(self, fold_index: int, stage_name: str, completed_epochs: int) -> None:
        self.state.update(
            {
                "current_fold": fold_index,
                "current_stage": stage_name,
                "stage_epoch": completed_epochs,
                "status": "running",
            }
        )
        self.save()

    def mark_stage_complete(self, fold_index: int, stage_name: str) -> None:
        key = str(fold_index)
        stages = set(self.state.setdefault("completed_stages", {}).get(key, []))
        stages.add(stage_name)
        self.state["completed_stages"][key] = sorted(stages)
        self.state.update(
            {
                "current_fold": fold_index,
                "current_stage": stage_name,
                "stage_epoch": 0,
                "status": "stage_complete",
            }
        )
        self.save()

    def mark_fold_complete(self, fold_index: int) -> None:
        folds = set(self.state.setdefault("completed_folds", []))
        folds.add(fold_index)
        self.state["completed_folds"] = sorted(folds)
        self.state.update(
            {
                "current_fold": fold_index + 1,
                "current_stage": "stage1",
                "stage_epoch": 0,
                "status": "fold_complete",
            }
        )
        self.save()

    def mark_complete(self) -> None:
        self.state.update({"status": "complete", "stage_epoch": 0})
        self.save()

    def is_fold_complete(self, fold_index: int) -> bool:
        return fold_index in set(self.state.get("completed_folds", []))

    def is_stage_complete(self, fold_index: int, stage_name: str) -> bool:
        stages = self.state.get("completed_stages", {}).get(str(fold_index), [])
        return stage_name in set(stages)

    def epoch_for(self, fold_index: int, stage_name: str) -> int:
        if (
            self.state.get("current_fold") == fold_index
            and self.state.get("current_stage") == stage_name
        ):
            return int(self.state.get("stage_epoch", 0))
        return 0
