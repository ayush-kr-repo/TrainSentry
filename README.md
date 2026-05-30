<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=🛡️%20TrainSentry&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=AI-Assisted%20ML%20Training%20Log%20Analyzer&descAlignY=58&descSize=20&descColor=a78bfa&animation=fadeIn" />

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=20&pause=1000&color=A78BFA&center=true&vCenter=true&width=650&lines=Detect+Overfitting+%7C+Spikes+%7C+Stagnation;Claude-Powered+Debugging+Narratives;Rules+Engine+%2B+LLM+Report+Generation;CLI+%7C+Streamlit+Dashboard+%7C+CI+Tested" alt="Typing SVG" />
</a>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Claude](https://img.shields.io/badge/Claude-AI%20Reports-7c3aed?style=for-the-badge&logoColor=white)](https://anthropic.com)
[![Pytest](https://img.shields.io/badge/Pytest-Tested-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![CI](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](/.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-a78bfa?style=for-the-badge)](LICENSE)

> **TrainSentry automates the first debugging pass of ML training runs.**
> It parses PyTorch-style CSV logs, detects training anomalies with a rules engine,
> and optionally generates rich debugging narratives powered by Claude.

</div>

---

## 💡 What Problem Does It Solve?

Modern ML teams run many experiments. The first debugging pass is almost always manual — comparing loss curves, spotting train/val gaps, finding unstable epochs, deciding what to tune next.

TrainSentry automates that pass:

- 🔴 **Overfitting** — training loss improves while validation loss diverges
- 📈 **Validation loss spikes** — sharp increases around specific epochs
- 🟡 **Validation stagnation** — loss plateaus across recent epochs with no improvement
- 📉 **Metric drift** — growing train/val accuracy gap across training
- ⚠️ **Suspicious validation accuracy** — unusually high val accuracy flagging potential leakage

**Without AI report:** fast, offline, deterministic — a structured Markdown report with severity labels, evidence, and suggested next actions.

**With AI report (`--ai-report`):** Claude reads the structured findings and writes a rich debugging narrative — root cause analysis, concrete next steps, a prioritised action plan, and false-positive flags.

---

## 📸 Screenshots

> Run `streamlit run ml_debug_agent/dashboard.py` locally to see the full dashboard.

| View | Description |
|------|-------------|
| 📈 Training Curves | Side-by-side loss + accuracy curves with train/val gap bar chart |
| 🔍 Findings Panel | Severity-sorted issues with evidence, epoch ranges, suggested actions |
| 📄 Rules Report | Offline Markdown report — no API key needed |
| 🤖 AI Report | Claude-generated narrative — root cause analysis + prioritised action plan |

---

## 🚀 Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run tests
```bash
pytest -q
```

### Analyze a single run (rules-based)
```bash
python -m ml_debug_agent.cli analyze data/overfit_run.csv
```

### Analyze with AI-assisted report
```bash
export ANTHROPIC_API_KEY=your_key_here
python -m ml_debug_agent.cli analyze data/overfit_run.csv --ai-report
```

### Compare multiple runs
```bash
python -m ml_debug_agent.cli compare data/healthy_run.csv data/overfit_run.csv data/unstable_run.csv
```

### Save report to file
```bash
python -m ml_debug_agent.cli analyze data/overfit_run.csv --ai-report --save
# → outputs/overfit_run_ai_report.md
```

### Launch dashboard
```bash
streamlit run ml_debug_agent/dashboard.py
# → http://localhost:8501
```

---

## 📋 Expected Log Format

TrainSentry expects a CSV with these required columns:

```text
epoch, train_loss, val_loss, train_accuracy, val_accuracy
```

```csv
epoch,train_loss,val_loss,train_accuracy,val_accuracy
1,0.910,0.920,0.570,0.550
2,0.760,0.780,0.660,0.640
3,0.620,0.650,0.750,0.710
```

Optional columns are allowed and preserved by the parser.

---

## 🤖 AI Report — How It Works

The `--ai-report` flag sends structured findings to Claude and returns an engineer-readable debugging report.

```
Detected Findings (rules engine)
            │
            ▼
    Structured JSON payload
    {issue_type, severity, epoch_range,
     evidence, suggested_action}
            │
            ▼
    Claude (claude-sonnet-4)
            │
            ▼
    AI Debugging Report
    ├── Executive summary
    ├── Per-finding root cause analysis
    ├── Concrete next steps per issue
    ├── Prioritised action plan (top 3)
    └── False-positive confidence notes
```

**No API key?** TrainSentry falls back to the rules-based report automatically — no crash, no silent failure, just a clear warning message.

### Example AI report output

```markdown
## Executive Summary
The overfit_run experiment shows a textbook overfitting pattern beginning
at epoch 8. Training loss continues to decrease while validation loss
diverges sharply — the model is memorising the training set rather than
generalising. The train/val accuracy gap confirms this.

## Finding 1: Overfitting — Root Cause
The validation loss inflection at epoch 8 coincides with the point where
training accuracy exceeds 92%. At this saturation point, further gradient
updates are driven almost entirely by training-set noise...

## Prioritised Action Plan
1. Add dropout (0.3–0.5) after dense layers — highest expected impact
2. Reduce learning rate by 10x from epoch 8 with ReduceLROnPlateau
3. Add L2 regularisation (λ = 1e-4) to all weight matrices
```

---

## 🧠 Tech Stack

| Tech | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Log parsing and feature computation |
| Streamlit | Interactive dashboard |
| Plotly | Training curve visualizations |
| Claude (Anthropic) | AI-assisted debugging narrative generation |
| Pytest | Unit test suite |
| GitHub Actions | CI — runs tests on every push |

---

## 🏗️ Architecture

```
CSV Training Log
      │
      ▼
parser.py          ── Load, validate, and type-check the log file
      │
      ▼
analyzer.py        ── Rules engine: overfitting, spikes,
      │                stagnation, drift, leakage detection
      ▼
schemas.py         ── ExperimentSummary + Finding dataclasses
      │
      ├──── reporter.py (rules) ──► Structured Markdown report
      │
      └──── reporter.py (AI) ────► Claude API
                                        │
                                        ▼
                                   AI Debugging Narrative
                                   (root cause + action plan)
      │
      ├──── cli.py      ──► Terminal output + --save flag
      └──── dashboard.py ──► Streamlit UI (4 tabs)
```

---

## 🔍 Detection Engine

| Issue | Detection Logic |
|-------|----------------|
| **Overfitting** | `train_loss` decreasing while `val_loss` increasing over a sliding window |
| **Val loss spike** | `val_loss` increases sharply (> threshold) within a single epoch |
| **Val stagnation** | `val_loss` delta < ε across the last N epochs |
| **Metric drift** | Train/val accuracy gap grows monotonically across training |
| **Suspicious val accuracy** | `val_accuracy` exceeds configurable upper bound (default: 0.99) |

---

## 📁 Project Structure

```
TrainSentry/
├── ml_debug_agent/
│   ├── analyzer.py      # Training anomaly detection — rules engine
│   ├── cli.py           # CLI: analyze + compare subcommands + --ai-report flag
│   ├── dashboard.py     # Streamlit dashboard — 4 tabs including AI report
│   ├── parser.py        # CSV loading, validation, type checking
│   ├── reporter.py      # Rules report + Claude AI report with graceful fallback
│   └── schemas.py       # ExperimentSummary + Finding dataclasses
├── data/
│   ├── healthy_run.csv
│   ├── overfit_run.csv
│   └── unstable_run.csv
├── tests/
│   ├── test_analyzer.py
│   ├── test_parser.py
│   └── test_reporter.py
├── .github/workflows/
│   └── ci.yml           # Automated pytest on every push
├── requirements.txt
└── README.md
```

---

## 🧪 Testing & CI

```bash
# Full test suite
pytest -q

# With coverage
pytest --cov=ml_debug_agent --cov-report=term-missing
```

GitHub Actions runs the full test suite on every push to `main` and every pull request — zero-config CI.

---

## 🔮 Future Improvements

- W&B and MLflow native integration (pull logs without CSV export)
- Statistical anomaly detection — CUSUM and IQR-based spike detection
- Multi-run AI comparison report (identify which experiment to continue)
- Automatic hyperparameter suggestion based on detected failure modes
- Support for JSON and TensorBoard log formats
- Slack/Discord alert webhook for long-running training jobs

---

## 📄 License

[![License: MIT](https://img.shields.io/badge/License-MIT-a78bfa?style=for-the-badge)](LICENSE)

<div align="center">

*Built by [Ayush Kumar](https://github.com/ayush-kr-repo) · [LinkedIn](https://www.linkedin.com/in/ayush-kumar-74a67730a/)*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" />

</div>