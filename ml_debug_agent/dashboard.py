"""
dashboard.py — Streamlit dashboard for TrainSentry.

Launch with:
    streamlit run ml_debug_agent/dashboard.py
"""

from __future__ import annotations

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from ml_debug_agent.parser import load_log
from ml_debug_agent.analyzer import analyze
from ml_debug_agent.reporter import generate_report, generate_ai_report


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrainSentry",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrainSentry")
st.caption("AI-assisted ML training log analyzer")

# ── Sidebar — file upload + settings ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded = st.file_uploader(
        "Upload training log CSV",
        type=["csv"],
        help="Required columns: epoch, train_loss, val_loss, train_accuracy, val_accuracy",
    )

    st.divider()

    st.subheader("🤖 AI Report")
    use_ai = st.toggle(
        "Enable AI-assisted report",
        value=False,
        help="Uses Claude to generate a rich debugging narrative. Requires an Anthropic API key.",
    )

    api_key_input = ""
    if use_ai:
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            help="Leave blank to use ANTHROPIC_API_KEY environment variable",
        )

    st.divider()
    st.caption("TrainSentry · Rules engine + Claude")

# ── Main content ──────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload a training log CSV to get started.")

    st.subheader("Expected CSV format")
    st.code(
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy\n"
        "1,0.910,0.920,0.570,0.550\n"
        "2,0.760,0.780,0.660,0.640\n"
        "3,0.620,0.650,0.750,0.710",
        language="csv",
    )

    # Show sample data from repo if available
    sample_paths = list(Path("data").glob("*.csv")) if Path("data").exists() else []
    if sample_paths:
        st.subheader("Or load a sample")
        selected_sample = st.selectbox("Sample file", [p.name for p in sample_paths])
        if st.button("Load sample"):
            df = load_log(Path("data") / selected_sample)
            st.session_state["df"] = df
            st.session_state["name"] = Path(selected_sample).stem
            st.rerun()

    st.stop()

# ── Load and analyze ──────────────────────────────────────────────────────────
df = load_log(uploaded)
experiment_name = Path(uploaded.name).stem
summary = analyze(df, experiment_name=experiment_name)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_curves, tab_findings, tab_report, tab_ai = st.tabs([
    "📈 Training Curves",
    "🔍 Findings",
    "📄 Rules Report",
    "🤖 AI Report",
])

# ── Tab 1: Training Curves ────────────────────────────────────────────────────
with tab_curves:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loss Curves")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=df["epoch"], y=df["train_loss"],
            mode="lines+markers", name="Train Loss",
            line=dict(color="#6366f1", width=2),
        ))
        fig_loss.add_trace(go.Scatter(
            x=df["epoch"], y=df["val_loss"],
            mode="lines+markers", name="Val Loss",
            line=dict(color="#f43f5e", width=2, dash="dash"),
        ))
        fig_loss.update_layout(
            xaxis_title="Epoch", yaxis_title="Loss",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        st.subheader("Accuracy Curves")
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=df["epoch"], y=df["train_accuracy"],
            mode="lines+markers", name="Train Accuracy",
            line=dict(color="#6366f1", width=2),
        ))
        fig_acc.add_trace(go.Scatter(
            x=df["epoch"], y=df["val_accuracy"],
            mode="lines+markers", name="Val Accuracy",
            line=dict(color="#f43f5e", width=2, dash="dash"),
        ))
        fig_acc.update_layout(
            xaxis_title="Epoch", yaxis_title="Accuracy",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    # Train/Val gap chart
    st.subheader("Train–Validation Gap")
    df["loss_gap"] = df["train_loss"] - df["val_loss"]
    df["acc_gap"] = df["train_accuracy"] - df["val_accuracy"]
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Bar(
        x=df["epoch"], y=df["acc_gap"],
        name="Accuracy Gap",
        marker_color=["#f43f5e" if v > 0.1 else "#6366f1" for v in df["acc_gap"]],
    ))
    fig_gap.update_layout(
        xaxis_title="Epoch", yaxis_title="Train − Val Accuracy",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    # Raw data toggle
    with st.expander("View raw data"):
        st.dataframe(df, use_container_width=True)

# ── Tab 2: Findings ───────────────────────────────────────────────────────────
with tab_findings:
    st.subheader(f"Findings — `{experiment_name}`")

    if not summary.findings:
        st.success("✅ No issues detected. Training looks healthy.")
    else:
        severity_color = {"critical": "🔴", "warning": "🟡", "info": "🔵"}

        for finding in sorted(
            summary.findings,
            key=lambda f: {"critical": 0, "warning": 1, "info": 2}.get(f.severity.lower(), 9),
        ):
            icon = severity_color.get(finding.severity.lower(), "⚪")
            with st.expander(f"{icon} {finding.issue_type} — epoch {finding.epoch_range}", expanded=True):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"**Description:** {finding.description}")
                    if finding.suggested_action:
                        st.info(f"💡 **Suggested action:** {finding.suggested_action}")
                with col_b:
                    st.metric("Severity", finding.severity.upper())
                    if finding.evidence:
                        st.markdown("**Evidence:**")
                        for k, v in finding.evidence.items():
                            st.code(f"{k}: {v}")

    # Metric summary
    st.divider()
    st.subheader("Metric Summary")
    metric_cols = st.columns(len(summary.metric_summary))
    for col, (k, v) in zip(metric_cols, summary.metric_summary.items()):
        col.metric(k.replace("_", " ").title(), v)

# ── Tab 3: Rules Report ───────────────────────────────────────────────────────
with tab_report:
    st.subheader("Rules-Based Report")
    report_md = generate_report(summary)
    st.markdown(report_md)

    st.download_button(
        label="⬇️ Download Report",
        data=report_md,
        file_name=f"{experiment_name}_report.md",
        mime="text/markdown",
    )

# ── Tab 4: AI Report ──────────────────────────────────────────────────────────
with tab_ai:
    st.subheader("🤖 AI-Assisted Debugging Report")

    if not use_ai:
        st.info(
            "Enable **AI-assisted report** in the sidebar to generate a "
            "Claude-powered debugging narrative with root cause analysis and "
            "a prioritised action plan."
        )
    else:
        resolved_key = api_key_input or os.getenv("ANTHROPIC_API_KEY", "")

        if not resolved_key:
            st.warning("⚠️ No API key found. Add your Anthropic API key in the sidebar.")
        else:
            cache_key = f"ai_report_{experiment_name}_{len(summary.findings)}"

            if cache_key not in st.session_state:
                with st.spinner("Generating AI report via Claude..."):
                    ai_report = generate_ai_report(summary, api_key=resolved_key)
                    st.session_state[cache_key] = ai_report

            ai_report = st.session_state[cache_key]
            st.markdown(ai_report)

            col_dl, col_regen = st.columns([2, 1])
            with col_dl:
                st.download_button(
                    label="⬇️ Download AI Report",
                    data=ai_report,
                    file_name=f"{experiment_name}_ai_report.md",
                    mime="text/markdown",
                )
            with col_regen:
                if st.button("🔄 Regenerate"):
                    if cache_key in st.session_state:
                        del st.session_state[cache_key]
                    st.rerun()