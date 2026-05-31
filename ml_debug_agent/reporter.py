"""
reporter.py — Markdown report generation for TrainSentry.

Two modes:
  - Rules-based (default): fast, offline, no API key needed.
    Function: build_markdown_report(analysis) — original interface preserved.
  - AI-assisted: sends structured findings to Gemini for a rich narrative.
    Function: generate_ai_report(analysis, api_key=None)
"""

from __future__ import annotations

import os
import json
import textwrap
from typing import Optional

from ml_debug_agent.schemas import RunAnalysis, Finding, Severity


# ── Severity helpers ──────────────────────────────────────────────────────────
_SEVERITY_ICON = {
    Severity.CRITICAL: "🔴",
    Severity.WARNING:  "🟡",
    Severity.INFO:     "🔵",
}

_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.WARNING:  1,
    Severity.INFO:     2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Rules-based report — original interface preserved
# ─────────────────────────────────────────────────────────────────────────────

def build_markdown_report(analysis: RunAnalysis) -> str:
    """Generate a plain Markdown report from structured findings.

    Offline, deterministic, no API key needed.
    Original function name preserved for backwards compatibility.
    """
    lines: list[str] = []

    lines.append(f"# TrainSentry Report — `{analysis.run.name}`\n")
    lines.append(f"**Epochs analysed:** {len(analysis.run.frame)}  ")
    lines.append(f"**Issues detected:** {len(analysis.findings)}  ")
    lines.append(f"**Best val loss:** {analysis.best_val_loss:.4f} (epoch {analysis.best_epoch})  ")
    lines.append(f"**Best val accuracy:** {analysis.best_val_accuracy:.4f}\n")
    lines.append("---\n")

    if not analysis.findings:
        lines.append("✅ **No issues detected.** Training looks healthy.\n")
        return "\n".join(lines)

    sorted_findings = sorted(
        analysis.findings,
        key=lambda f: _SEVERITY_ORDER.get(f.severity, 99),
    )

    lines.append("## Findings\n")
    for i, finding in enumerate(sorted_findings, 1):
        icon = _SEVERITY_ICON.get(finding.severity, "⚪")
        lines.append(f"### {icon} Finding {i}: {finding.title}")
        lines.append(f"**Severity:** `{finding.severity.value.upper()}`\n")
        lines.append(f"**Evidence:** {finding.evidence}\n")
        lines.append(f"**Recommendation:** {finding.recommendation}\n")
        lines.append("---\n")

    lines.append("## Summary\n")
    lines.append(f"- 🔴 Critical: {analysis.critical_count}")
    lines.append(f"- 🟡 Warnings: {analysis.warning_count}")
    lines.append(f"- ✅ Best epoch: {analysis.best_epoch}")
    lines.append(f"- 📉 Best val loss: {analysis.best_val_loss:.4f}")
    lines.append(f"- 🎯 Best val accuracy: {analysis.best_val_accuracy:.4f}")

    return "\n".join(lines)


# Alias for new-style callers
generate_report = build_markdown_report


# ─────────────────────────────────────────────────────────────────────────────
# AI-assisted report — powered by Gemini (free tier)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML engineer and debugging assistant.
    You receive structured training log analysis from TrainSentry.
    Write a rich, actionable debugging report in Markdown.

    Your report must include:
    1. A brief executive summary (2-3 sentences) — what is happening overall?
    2. For each finding: plain-English root cause explanation, why it matters,
       and 2-3 concrete next steps the engineer should try.
    3. A prioritised action plan — top 3 things to do next, ordered by impact.
    4. A confidence note: flag any finding that might be a false positive.

    Tone: direct, technical, peer-level. No filler. No bullet soup.
    Format: clean Markdown with headers and tables where appropriate.
    Do not repeat raw numbers verbatim — interpret them.
""").strip()


def generate_ai_report(
    analysis: RunAnalysis,
    api_key: Optional[str] = None,
) -> str:
    """Generate a Gemini-powered debugging report from structured findings.

    Falls back to rules-based report if no API key or if the call fails.
    Get a free key at: https://aistudio.google.com/app/apikey
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")

    if not resolved_key:
        print(
            "⚠️  No GEMINI_API_KEY found. Falling back to rules-based report.\n"
            "   Get a free key at: https://aistudio.google.com/app/apikey\n"
            "   Set GEMINI_API_KEY in your .env file to enable AI reports."
        )
        return build_markdown_report(analysis)

    try:
        import google.generativeai as genai
    except ImportError:
        print(
            "⚠️  `google-generativeai` not installed.\n"
            "   Run: pip install google-generativeai\n"
            "   Falling back to rules-based report."
        )
        return build_markdown_report(analysis)

    payload = {
        "experiment_name": analysis.run.name,
        "total_epochs": len(analysis.run.frame),
        "best_epoch": analysis.best_epoch,
        "best_val_loss": analysis.best_val_loss,
        "best_val_accuracy": analysis.best_val_accuracy,
        "critical_count": analysis.critical_count,
        "warning_count": analysis.warning_count,
        "findings": [
            {
                "title": f.title,
                "severity": f.severity.value,
                "evidence": f.evidence,
                "recommendation": f.recommendation,
            }
            for f in analysis.findings
        ],
    }

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Analyse this TrainSentry output and write a debugging report.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )

    try:
        genai.configure(api_key=resolved_key)
        model = genai.GenerativeModel("gemini-3.5-flash")
        response = model.generate_content(prompt)   
        ai_content = response.text

        header = (
            f"# TrainSentry AI Report — `{analysis.run.name}`\n\n"
            f"> *Generated by Gemini 2.0 Flash · {len(analysis.run.frame)} epochs · "
            f"{len(analysis.findings)} issue(s) detected*\n\n---\n\n"
        )
        return header + ai_content

    except Exception as exc:
        print(f"⚠️  AI report failed ({type(exc).__name__}: {exc}). Using rules-based fallback.")
        return build_markdown_report(analysis)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_report(analyses: list[RunAnalysis]) -> str:
    """Side-by-side comparison of multiple experiment runs."""
    lines: list[str] = []
    lines.append("# TrainSentry — Multi-Run Comparison\n")
    lines.append(f"**Runs compared:** {len(analyses)}\n")
    lines.append("---\n")

    lines.append("## Overview\n")
    lines.append("| Run | Epochs | Critical | Warnings | Best Val Loss | Best Epoch |")
    lines.append("|-----|--------|----------|----------|---------------|------------|")
    for a in analyses:
        lines.append(
            f"| `{a.run.name}` | {len(a.run.frame)} | "
            f"{a.critical_count} | {a.warning_count} | "
            f"{a.best_val_loss:.4f} | {a.best_epoch} |"
        )

    lines.append("")
    for a in analyses:
        lines.append(f"## `{a.run.name}`\n")
        if not a.findings:
            lines.append("✅ No issues detected.\n")
        else:
            for f in a.findings:
                icon = _SEVERITY_ICON.get(f.severity, "⚪")
                lines.append(f"- {icon} **{f.title}**: {f.evidence}")
            lines.append("")
        lines.append("---\n")

    return "\n".join(lines)