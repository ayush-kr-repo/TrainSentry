"""
reporter.py — Markdown report generation for TrainSentry.

Two modes:
  - Rules-based (default): fast, offline, no API key needed.
  - AI-assisted (--ai-report): sends structured findings to Claude and returns
    a rich, engineer-readable debugging narrative with root cause analysis
    and concrete next-step recommendations.
"""

from __future__ import annotations

import os
import json
import textwrap
from dataclasses import asdict
from typing import Optional

from ml_debug_agent.schemas import ExperimentSummary, Finding


# ── Severity ordering for sorting findings ────────────────────────────────────
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


# ─────────────────────────────────────────────────────────────────────────────
# Rules-based report (no API key required)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(summary: ExperimentSummary) -> str:
    """Generate a plain Markdown report from structured findings.

    This is the offline, rules-based path. No external API calls.
    Output is deterministic and works without any credentials.
    """
    lines: list[str] = []

    lines.append(f"# TrainSentry Report — `{summary.experiment_name}`\n")
    lines.append(f"**Epochs analysed:** {summary.total_epochs}  ")
    lines.append(f"**Issues detected:** {len(summary.findings)}\n")
    lines.append("---\n")

    if not summary.findings:
        lines.append("✅ **No issues detected.** Training looks healthy.\n")
        return "\n".join(lines)

    # Sort by severity so critical issues surface first
    sorted_findings = sorted(
        summary.findings,
        key=lambda f: _SEVERITY_ORDER.get(f.severity.lower(), 99),
    )

    lines.append("## Findings\n")
    for i, finding in enumerate(sorted_findings, 1):
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
            finding.severity.lower(), "⚪"
        )
        lines.append(f"### {icon} Finding {i}: {finding.issue_type}")
        lines.append(f"**Severity:** `{finding.severity.upper()}`  ")
        lines.append(f"**Epoch(s):** {finding.epoch_range}  ")
        lines.append(f"\n{finding.description}\n")

        if finding.evidence:
            lines.append("**Evidence:**")
            for k, v in finding.evidence.items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")

        if finding.suggested_action:
            lines.append(f"**Suggested action:** {finding.suggested_action}\n")

        lines.append("---\n")

    # Metric summary table
    lines.append("## Metric Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in summary.metric_summary.items():
        lines.append(f"| {k} | {v} |")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# AI-assisted report (requires ANTHROPIC_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML engineer and debugging assistant.
    You receive structured training log analysis output from a tool called TrainSentry.
    Your job is to write a rich, actionable debugging report in Markdown.

    Your report must include:
    1. A brief executive summary (2-3 sentences) — what is happening overall?
    2. For each detected finding: a plain-English explanation of the root cause,
       why it matters, and 2-3 concrete next steps the engineer should try.
    3. A prioritised action plan at the end — the top 3 things to do next,
       ordered by expected impact.
    4. A confidence note: flag if any finding might be a false positive and why.

    Tone: direct, technical, peer-level. No filler. No bullet soup.
    Format: clean Markdown with headers, tables where appropriate.
    Do not repeat the raw numbers verbatim — interpret them.
""").strip()


def generate_ai_report(
    summary: ExperimentSummary,
    api_key: Optional[str] = None,
) -> str:
    """Generate an LLM-powered debugging report using Claude.

    Falls back to the rules-based report if:
    - No API key is provided and ANTHROPIC_API_KEY env var is not set
    - The API call fails for any reason

    Args:
        summary: Structured experiment summary from the analyzer.
        api_key: Anthropic API key. Reads ANTHROPIC_API_KEY env var if None.

    Returns:
        Markdown string — AI narrative if successful, rules-based fallback otherwise.
    """
    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    if not resolved_key:
        print(
            "⚠️  No ANTHROPIC_API_KEY found. Falling back to rules-based report.\n"
            "   Set ANTHROPIC_API_KEY in your environment to enable AI reports."
        )
        return generate_report(summary)

    try:
        import anthropic  # lazy import — only needed for AI path
    except ImportError:
        print(
            "⚠️  `anthropic` package not installed. Run: pip install anthropic\n"
            "   Falling back to rules-based report."
        )
        return generate_report(summary)

    # Build a structured payload for the model — clean JSON, no raw dataframes
    payload = {
        "experiment_name": summary.experiment_name,
        "total_epochs": summary.total_epochs,
        "metric_summary": summary.metric_summary,
        "findings": [
            {
                "issue_type": f.issue_type,
                "severity": f.severity,
                "epoch_range": f.epoch_range,
                "description": f.description,
                "evidence": f.evidence,
                "suggested_action": f.suggested_action,
            }
            for f in summary.findings
        ],
    }

    user_message = (
        f"Analyse the following TrainSentry output and write a debugging report.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )

    try:
        client = anthropic.Anthropic(api_key=resolved_key)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        ai_content = response.content[0].text

        # Prepend a header so the output is self-contained
        header = (
            f"# TrainSentry AI Report — `{summary.experiment_name}`\n\n"
            f"> *Generated by Claude · {summary.total_epochs} epochs analysed · "
            f"{len(summary.findings)} issue(s) detected*\n\n---\n\n"
        )
        return header + ai_content

    except Exception as exc:  # network error, auth error, rate limit, etc.
        print(f"⚠️  AI report failed ({type(exc).__name__}: {exc}). Falling back to rules-based report.")
        return generate_report(summary)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report (multi-run)
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_report(summaries: list[ExperimentSummary]) -> str:
    """Generate a side-by-side comparison report for multiple runs."""
    lines: list[str] = []
    lines.append("# TrainSentry — Multi-Run Comparison\n")
    lines.append(f"**Runs compared:** {len(summaries)}\n")
    lines.append("---\n")

    # Summary table
    lines.append("## Overview\n")
    lines.append("| Run | Epochs | Issues | Severities |")
    lines.append("|-----|--------|--------|------------|")
    for s in summaries:
        severity_counts = {}
        for f in s.findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
        sev_str = " · ".join(f"{v}× {k}" for k, v in severity_counts.items()) or "none"
        lines.append(f"| `{s.experiment_name}` | {s.total_epochs} | {len(s.findings)} | {sev_str} |")

    lines.append("")

    # Per-run breakdown
    for s in summaries:
        lines.append(f"## `{s.experiment_name}`\n")
        if not s.findings:
            lines.append("✅ No issues detected.\n")
        else:
            for f in s.findings:
                icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                    f.severity.lower(), "⚪"
                )
                lines.append(f"- {icon} **{f.issue_type}** (epoch {f.epoch_range}): {f.description}")
            lines.append("")

        lines.append("**Metrics:**")
        for k, v in s.metric_summary.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("\n---\n")

    return "\n".join(lines)