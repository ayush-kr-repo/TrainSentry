# TrainSentry

AI-assisted training log analyzer for machine learning experiments. TrainSentry parses PyTorch-style CSV logs, detects common training issues, and generates debugging reports that help engineers iterate faster on model training loops.

## Overview

Modern ML teams run many experiments, but the first debugging pass is often manual: comparing curves, inspecting train/validation gaps, finding unstable epochs, and deciding what to try next.

TrainSentry automates that first pass with a rules-first analysis engine, command-line reports, and a lightweight Streamlit dashboard.

## Features

- Parses experiment logs from CSV files
- Detects overfitting, validation loss spikes, validation stagnation, metric drift, and suspiciously high validation accuracy
- Generates plain-English debugging reports with severity labels, evidence, and suggested next actions
- Compares multiple experiment runs from the command line
- Visualizes training and validation curves in a Streamlit dashboard
- Includes unit tests with `pytest`
- Includes GitHub Actions CI for automated test runs

## Project Structure

```text
TrainSentry/
  ml_debug_agent/
    analyzer.py      # Training anomaly detection logic
    cli.py           # Command-line interface
    dashboard.py     # Streamlit dashboard
    parser.py        # CSV loading and validation
    reporter.py      # Markdown report generation
    schemas.py       # Shared dataclasses
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
  README.md
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Analyze one training run:

```bash
python -m ml_debug_agent.cli analyze data/overfit_run.csv
```

Compare multiple runs:

```bash
python -m ml_debug_agent.cli compare data/healthy_run.csv data/overfit_run.csv data/unstable_run.csv
```

Launch the dashboard:

```bash
streamlit run ml_debug_agent/dashboard.py
```

Then open:

```text
http://localhost:8501
```

## Expected Log Format

TrainSentry expects a CSV file with these required columns:

```text
epoch,train_loss,val_loss,train_accuracy,val_accuracy
```

Example:

```csv
epoch,train_loss,val_loss,train_accuracy,val_accuracy
1,0.910,0.920,0.570,0.550
2,0.760,0.780,0.660,0.640
3,0.620,0.650,0.750,0.710
```

Optional columns are allowed and preserved by the parser.

## Example Findings

TrainSentry can identify issues such as:

- **Possible overfitting:** training loss improves while validation loss worsens
- **Validation loss spike:** validation loss increases sharply around specific epochs
- **Validation stagnation:** validation loss stops improving across recent epochs
- **Metric drift:** train-validation accuracy gap grows during training
- **Suspicious validation accuracy:** unusually high validation accuracy may indicate leakage

## Tech Stack

- Python
- Pandas
- Streamlit
- Plotly
- Pytest
- GitHub Actions
