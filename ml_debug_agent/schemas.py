from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Finding:
    title: str
    severity: Severity
    evidence: str
    recommendation: str


@dataclass(frozen=True)
class ExperimentRun:
    name: str
    source: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class RunAnalysis:
    run: ExperimentRun
    findings: list[Finding]
    best_epoch: int
    best_val_loss: float
    best_val_accuracy: float

    @property
    def critical_count(self) -> int:
        return sum(f.severity == Severity.CRITICAL for f in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(f.severity == Severity.WARNING for f in self.findings)

