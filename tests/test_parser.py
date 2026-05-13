from pathlib import Path

import pytest

from ml_debug_agent.parser import load_training_log


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_load_training_log_sorts_epochs_and_returns_run() -> None:
    run = load_training_log(DATA_DIR / "healthy_run.csv")

    assert run.name == "healthy_run"
    assert list(run.frame.columns[:5]) == [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    ]
    assert run.frame["epoch"].is_monotonic_increasing


def test_load_training_log_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_training_log(DATA_DIR / "missing.csv")

