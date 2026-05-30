"""
cli.py — Command-line interface for TrainSentry.

Usage examples:
    python -m ml_debug_agent.cli analyze data/overfit_run.csv
    python -m ml_debug_agent.cli analyze data/overfit_run.csv --ai-report
    python -m ml_debug_agent.cli analyze data/overfit_run.csv --ai-report --save
    python -m ml_debug_agent.cli compare data/healthy_run.csv data/overfit_run.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml_debug_agent.parser import load_log
from ml_debug_agent.analyzer import analyze
from ml_debug_agent.reporter import generate_report, generate_ai_report, generate_comparison_report


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze a single training log file."""
    log_path = Path(args.file)

    if not log_path.exists():
        print(f"❌ File not found: {log_path}")
        sys.exit(1)

    print(f"📂 Loading: {log_path.name}")
    df = load_log(log_path)

    print(f"🔍 Analyzing {len(df)} epochs...")
    summary = analyze(df, experiment_name=log_path.stem)

    # Choose report mode
    if args.ai_report:
        print("🤖 Generating AI-assisted report via Claude...\n")
        report = generate_ai_report(summary, api_key=args.api_key)
    else:
        report = generate_report(summary)

    print(report)

    # Optionally save to file
    if args.save:
        suffix = "_ai_report.md" if args.ai_report else "_report.md"
        out_path = log_path.with_suffix("").parent / (log_path.stem + suffix)
        out_path.write_text(report, encoding="utf-8")
        print(f"\n💾 Report saved to: {out_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare multiple training log files side by side."""
    summaries = []

    for file_str in args.files:
        path = Path(file_str)
        if not path.exists():
            print(f"❌ File not found: {path}")
            sys.exit(1)
        df = load_log(path)
        summary = analyze(df, experiment_name=path.stem)
        summaries.append(summary)

    report = generate_comparison_report(summaries)
    print(report)

    if args.save:
        out_path = Path("comparison_report.md")
        out_path.write_text(report, encoding="utf-8")
        print(f"\n💾 Comparison saved to: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainsentry",
        description="TrainSentry — AI-assisted ML training log analyzer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── analyze subcommand ────────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a single training log CSV",
    )
    analyze_parser.add_argument("file", help="Path to training log CSV")
    analyze_parser.add_argument(
        "--ai-report",
        action="store_true",
        help="Generate AI-assisted narrative report via Claude (requires ANTHROPIC_API_KEY)",
    )
    analyze_parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)",
    )
    analyze_parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to a Markdown file alongside the input CSV",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # ── compare subcommand ────────────────────────────────────────────────────
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple training log CSVs side by side",
    )
    compare_parser.add_argument(
        "files",
        nargs="+",
        help="Paths to two or more training log CSVs",
    )
    compare_parser.add_argument(
        "--save",
        action="store_true",
        help="Save comparison report to comparison_report.md",
    )
    compare_parser.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()