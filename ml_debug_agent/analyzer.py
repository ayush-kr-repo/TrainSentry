from __future__ import annotations

from statistics import mean

import pandas as pd

from .schemas import ExperimentRun, Finding, RunAnalysis, Severity


def analyze_run(run: ExperimentRun) -> RunAnalysis:
    frame = run.frame
    best_idx = frame["val_loss"].idxmin()
    best_row = frame.loc[best_idx]

    findings: list[Finding] = []
    findings.extend(_detect_overfitting(frame))
    findings.extend(_detect_loss_spikes(frame))
    findings.extend(_detect_validation_stagnation(frame))
    findings.extend(_detect_metric_drift(frame))
    findings.extend(_detect_suspicious_accuracy(frame))

    if not findings:
        findings.append(
            Finding(
                title="No major training instability detected",
                severity=Severity.INFO,
                evidence="Training and validation metrics move consistently across the run.",
                recommendation="Use this run as a baseline and compare future experiments against it.",
            )
        )

    return RunAnalysis(
        run=run,
        findings=findings,
        best_epoch=int(best_row["epoch"]),
        best_val_loss=float(best_row["val_loss"]),
        best_val_accuracy=float(best_row["val_accuracy"]),
    )


def _detect_overfitting(frame: pd.DataFrame) -> list[Finding]:
    if len(frame) < 4:
        return []

    last_window = frame.tail(max(3, len(frame) // 3))
    train_loss_drop = float(last_window["train_loss"].iloc[0] - last_window["train_loss"].iloc[-1])
    val_loss_change = float(last_window["val_loss"].iloc[-1] - last_window["val_loss"].iloc[0])
    accuracy_gap = float(last_window["train_accuracy"].iloc[-1] - last_window["val_accuracy"].iloc[-1])

    if train_loss_drop > 0.03 and val_loss_change > 0.03 and accuracy_gap > 0.08:
        return [
            Finding(
                title="Possible overfitting",
                severity=Severity.CRITICAL,
                evidence=(
                    f"Late train loss improved by {train_loss_drop:.3f}, "
                    f"but validation loss worsened by {val_loss_change:.3f}; "
                    f"final accuracy gap is {accuracy_gap:.1%}."
                ),
                recommendation=(
                    "Try stronger regularization, early stopping, data augmentation, "
                    "or reducing model capacity."
                ),
            )
        ]
    return []


def _detect_loss_spikes(frame: pd.DataFrame) -> list[Finding]:
    if len(frame) < 3:
        return []

    changes = frame["val_loss"].diff().dropna()
    typical_change = mean(abs(value) for value in changes) or 0.001
    spike_rows = frame.loc[frame["val_loss"].diff() > max(0.08, 2.5 * typical_change)]

    if spike_rows.empty:
        return []

    epochs = ", ".join(str(int(epoch)) for epoch in spike_rows["epoch"].tolist())
    return [
        Finding(
            title="Validation loss spike",
            severity=Severity.WARNING,
            evidence=f"Validation loss increased sharply near epoch(s): {epochs}.",
            recommendation=(
                "Inspect learning rate, batch composition, data pipeline randomness, "
                "and checkpoint selection around the spike."
            ),
        )
    ]


def _detect_validation_stagnation(frame: pd.DataFrame) -> list[Finding]:
    if len(frame) < 6:
        return []

    last_window = frame.tail(5)
    improvement = float(last_window["val_loss"].iloc[0] - last_window["val_loss"].min())

    if improvement <= 0.011:
        return [
            Finding(
                title="Validation stagnation",
                severity=Severity.WARNING,
                evidence=f"Validation loss improved by only {improvement:.3f} over the last 5 epochs.",
                recommendation=(
                    "Consider learning-rate scheduling, feature cleanup, more data, "
                    "or stopping the run earlier."
                ),
            )
        ]
    return []


def _detect_metric_drift(frame: pd.DataFrame) -> list[Finding]:
    if len(frame) < 5:
        return []

    first_half = frame.head(len(frame) // 2)
    second_half = frame.tail(len(frame) // 2)
    first_gap = float((first_half["train_accuracy"] - first_half["val_accuracy"]).mean())
    second_gap = float((second_half["train_accuracy"] - second_half["val_accuracy"]).mean())
    drift = second_gap - first_gap

    if drift > 0.06:
        return [
            Finding(
                title="Train-validation metric drift",
                severity=Severity.WARNING,
                evidence=f"Average accuracy gap increased by {drift:.1%} in the second half of training.",
                recommendation=(
                    "Review whether the model is memorizing training examples or whether "
                    "validation distribution differs from training data."
                ),
            )
        ]
    return []


def _detect_suspicious_accuracy(frame: pd.DataFrame) -> list[Finding]:
    max_val_accuracy = float(frame["val_accuracy"].max())
    min_val_loss = float(frame["val_loss"].min())

    if max_val_accuracy >= 0.995 and min_val_loss < 0.03:
        return [
            Finding(
                title="Suspiciously high validation accuracy",
                severity=Severity.CRITICAL,
                evidence=(
                    f"Validation accuracy reached {max_val_accuracy:.2%} "
                    f"with minimum validation loss {min_val_loss:.3f}."
                ),
                recommendation=(
                    "Run leakage checks, verify train/validation split isolation, "
                    "and confirm labels or target-derived features are not included."
                ),
            )
        ]
    return []
