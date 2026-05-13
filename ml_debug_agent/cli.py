from __future__ import annotations

import argparse
from pathlib import Path

from .analyzer import analyze_run
from .parser import load_training_log
from .reporter import build_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ML training logs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze one training run.")
    analyze_parser.add_argument("log_path", type=Path)
    analyze_parser.add_argument("--output", type=Path, help="Optional markdown report path.")

    compare_parser = subparsers.add_parser("compare", help="Compare multiple training runs.")
    compare_parser.add_argument("log_paths", type=Path, nargs="+")

    args = parser.parse_args()

    if args.command == "analyze":
        report = _analyze_one(args.log_path)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
            print(f"Wrote report to {args.output}")
        else:
            print(report)
        return

    if args.command == "compare":
        _compare_runs(args.log_paths)


def _analyze_one(log_path: Path) -> str:
    run = load_training_log(log_path)
    analysis = analyze_run(run)
    return build_markdown_report(analysis)


def _compare_runs(log_paths: list[Path]) -> None:
    analyses = [analyze_run(load_training_log(path)) for path in log_paths]
    analyses.sort(key=lambda item: (item.critical_count, item.warning_count, item.best_val_loss))

    print("run,best_epoch,best_val_loss,best_val_accuracy,critical,warnings")
    for analysis in analyses:
        print(
            ",".join(
                [
                    analysis.run.name,
                    str(analysis.best_epoch),
                    f"{analysis.best_val_loss:.4f}",
                    f"{analysis.best_val_accuracy:.4f}",
                    str(analysis.critical_count),
                    str(analysis.warning_count),
                ]
            )
        )


if __name__ == "__main__":
    main()

