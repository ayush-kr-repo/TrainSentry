from pathlib import Path

from ml_debug_agent.analyzer import analyze_run
from ml_debug_agent.parser import load_training_log
from ml_debug_agent.schemas import Severity


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _titles(path: str) -> list[str]:
    analysis = analyze_run(load_training_log(DATA_DIR / path))
    return [finding.title for finding in analysis.findings]


def test_overfit_run_detects_overfitting() -> None:
    analysis = analyze_run(load_training_log(DATA_DIR / "overfit_run.csv"))

    assert "Possible overfitting" in [finding.title for finding in analysis.findings]
    assert any(finding.severity == Severity.CRITICAL for finding in analysis.findings)


def test_unstable_run_detects_loss_spike_and_stagnation() -> None:
    titles = _titles("unstable_run.csv")

    assert "Validation loss spike" in titles
    assert "Validation stagnation" in titles


def test_healthy_run_has_info_baseline_finding() -> None:
    analysis = analyze_run(load_training_log(DATA_DIR / "healthy_run.csv"))

    assert analysis.critical_count == 0
    assert analysis.warning_count == 0
    assert analysis.findings[0].severity == Severity.INFO

