from pathlib import Path
from ml_debug_agent.analyzer import analyze_run
from ml_debug_agent.parser import load_training_log
from ml_debug_agent.reporter import build_markdown_report

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def test_report_contains_summary_and_findings() -> None:
    analysis = analyze_run(load_training_log(DATA_DIR / "overfit_run.csv"))
    report = build_markdown_report(analysis)
    assert "overfit_run" in report
    assert "## Summary" in report or "**Epochs" in report
    assert "Possible overfitting" in report or "overfit" in report.lower()
    assert "Recommended next action" in report or "recommendation" in report.lower()