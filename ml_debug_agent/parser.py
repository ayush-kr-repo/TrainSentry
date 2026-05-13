from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import ExperimentRun

REQUIRED_COLUMNS = {
    "epoch",
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
}


def load_training_log(path: str | Path) -> ExperimentRun:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Training log not found: {source}")

    frame = pd.read_csv(source)
    frame.columns = [column.strip() for column in frame.columns]
    _validate_frame(frame, source)

    frame = frame.sort_values("epoch").reset_index(drop=True)
    return ExperimentRun(name=source.stem, source=source, frame=frame)


def _validate_frame(frame: pd.DataFrame, source: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{source} is missing required columns: {missing_list}")

    if frame.empty:
        raise ValueError(f"{source} does not contain any training rows")

    for column in REQUIRED_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"{source} contains null values in column: {column}")

    if not frame["epoch"].is_monotonic_increasing:
        frame.sort_values("epoch", inplace=True)

    if frame["epoch"].duplicated().any():
        raise ValueError(f"{source} contains duplicate epoch values")

