from __future__ import annotations

from .schemas import RunAnalysis, Severity

SEVERITY_LABELS = {
    Severity.INFO: "INFO",
    Severity.WARNING: "WARNING",
    Severity.CRITICAL: "CRITICAL",
}


def build_markdown_report(analysis: RunAnalysis) -> str:
    lines = [
        f"# ML Debug Report: {analysis.run.name}",
        "",
        "## Summary",
        "",
        f"- Source: `{analysis.run.source}`",
        f"- Best epoch by validation loss: `{analysis.best_epoch}`",
        f"- Best validation loss: `{analysis.best_val_loss:.4f}`",
        f"- Best validation accuracy: `{analysis.best_val_accuracy:.2%}`",
        f"- Critical findings: `{analysis.critical_count}`",
        f"- Warnings: `{analysis.warning_count}`",
        "",
        "## Findings",
        "",
    ]

    for index, finding in enumerate(analysis.findings, start=1):
        label = SEVERITY_LABELS[finding.severity]
        lines.extend(
            [
                f"### {index}. {finding.title} [{label}]",
                "",
                f"**Evidence:** {finding.evidence}",
                "",
                f"**Recommended next action:** {finding.recommendation}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"

