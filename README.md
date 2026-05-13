# ML Debug Agent

AI-assisted training log analyzer for machine learning experiments. It parses
PyTorch-style CSV logs, detects common training issues, and generates debugging
reports that help engineers iterate faster on model training loops.

## Why this project exists

Modern ML teams run many experiments, but the first debugging pass is often
manual: compare curves, inspect train/validation gaps, find unstable epochs,
and summarize what to try next. ML Debug Agent automates that first pass with a
rules-first analysis engine and a lightweight dashboard.

## Features

- Parse experiment logs from CSV files.
- Detect overfitting, loss spikes, validation stagnation, metric drift, and
  suspiciously high validation accuracy.
- Generate plain-English debugging reports with severity labels and suggested
  next actions.
- Compare multiple experiment runs from the command line.
- Visualize training and validation curves in a Streamlit dashboard.
- Includes unit tests and GitHub Actions CI.

## Project structure

```text
ml-debug-agent/
  ml_debug_agent/
    analyzer.py      # Training anomaly detection logic
    cli.py           # Command-line interface
    dashboard.py     # Streamlit dashboard
    parser.py        # CSV loading and validation
    reporter.py      # Markdown/plain-text report generation
    schemas.py       # Dataclasses shared across the project
  data/
    healthy_run.csv
    overfit_run.csv
    unstable_run.csv
  tests/
    test_analyzer.py
    test_parser.py
    test_reporter.py
  .github/workflows/ci.yml
  requirements.txt
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Analyze one run:

```bash
python -m ml_debug_agent.cli analyze data/overfit_run.csv
```

Compare several runs:

```bash
python -m ml_debug_agent.cli compare data/healthy_run.csv data/overfit_run.csv data/unstable_run.csv
```

Launch the dashboard:

```bash
streamlit run ml_debug_agent/dashboard.py
```

Run tests:

```bash
pytest -q
```

## Expected log format

The analyzer expects a CSV file with these columns:

```text
epoch,train_loss,val_loss,train_accuracy,val_accuracy
```

Optional columns are allowed and preserved by the parser.

## Resume-ready description

Built an AI-assisted ML debugging tool that parses training logs, detects
overfitting, loss spikes, metric drift, validation stagnation, and suspicious
accuracy patterns, then generates experiment reports and dashboard
visualizations with pytest coverage and GitHub Actions CI.

