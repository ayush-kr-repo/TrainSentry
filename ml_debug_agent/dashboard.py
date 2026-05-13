from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml_debug_agent.analyzer import analyze_run
from ml_debug_agent.parser import load_training_log
from ml_debug_agent.reporter import build_markdown_report

st.set_page_config(page_title="ML Debug Agent", layout="wide")

st.title("ML Debug Agent")

uploaded_file = st.file_uploader("Upload a training CSV", type=["csv"])
sample_path = PROJECT_DIR / "data" / "overfit_run.csv"

if uploaded_file is not None:
    temp_path = PROJECT_DIR / ".streamlit_uploaded_run.csv"
    temp_path.write_bytes(uploaded_file.getbuffer())
    run = load_training_log(temp_path)
else:
    run = load_training_log(sample_path)

analysis = analyze_run(run)
frame = run.frame

summary_cols = st.columns(4)
summary_cols[0].metric("Best epoch", analysis.best_epoch)
summary_cols[1].metric("Best val loss", f"{analysis.best_val_loss:.4f}")
summary_cols[2].metric("Best val accuracy", f"{analysis.best_val_accuracy:.2%}")
summary_cols[3].metric("Findings", len(analysis.findings))

loss_frame = frame.melt(
    id_vars="epoch",
    value_vars=["train_loss", "val_loss"],
    var_name="metric",
    value_name="value",
)
accuracy_frame = frame.melt(
    id_vars="epoch",
    value_vars=["train_accuracy", "val_accuracy"],
    var_name="metric",
    value_name="value",
)

chart_cols = st.columns(2)
with chart_cols[0]:
    st.plotly_chart(
        px.line(loss_frame, x="epoch", y="value", color="metric", title="Loss curves"),
        use_container_width=True,
    )

with chart_cols[1]:
    st.plotly_chart(
        px.line(accuracy_frame, x="epoch", y="value", color="metric", title="Accuracy curves"),
        use_container_width=True,
    )

st.subheader("Debug report")
st.markdown(build_markdown_report(analysis))

with st.expander("Raw log"):
    st.dataframe(pd.DataFrame(frame), use_container_width=True)

