from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from ml_debug_agent.parser import load_training_log
from ml_debug_agent.analyzer import analyze_run
from ml_debug_agent.reporter import build_markdown_report, generate_ai_report
from ml_debug_agent.schemas import Severity

st.set_page_config(
    page_title="TrainSentry",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",  # hide sidebar completely — controls are inline
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* Hide sidebar toggle completely — we don't use it */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

.stApp { background: #0a0a0f; }

/* ── Top control bar ── */
.ctrl-bar {
    background: #13131f;
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;
}
.ctrl-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #6366f1;
    letter-spacing: 1.5px; text-transform: uppercase;
    white-space: nowrap;
}

/* ── Header ── */
.ts-header {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0f0c29 100%);
    border-bottom: 1px solid rgba(139,92,246,0.3);
    padding: 1.4rem 2rem 1.2rem;
    margin: -1rem -1rem 1.5rem;
    display: flex; align-items: center;
}
.ts-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.ts-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #6366f1;
    letter-spacing: 2px; text-transform: uppercase; margin-top: 2px;
}

/* ── Metric cards ── */
.ts-metric {
    background: linear-gradient(135deg,#13131f,#1a1a2e);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
}
.ts-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem; font-weight: 700; color: #a78bfa; line-height: 1;
}
.ts-metric-label {
    font-size: 0.72rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0.4rem;
}

/* ── Finding cards ── */
.finding-card {
    background: #13131f; border-left: 3px solid;
    border-radius: 0 10px 10px 0; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.finding-card-critical { border-color: #ef4444; }
.finding-card-warning  { border-color: #eab308; }
.finding-card-info     { border-color: #3b82f6; }
.finding-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 600;
    color: #e2e8f0; margin-bottom: 0.6rem;
}
.finding-evidence { font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem; line-height: 1.5; }
.finding-rec {
    font-size: 0.82rem; color: #a78bfa; background: rgba(139,92,246,0.08);
    border-radius: 6px; padding: 0.5rem 0.8rem; line-height: 1.5;
}
.badge-critical {
    display:inline-block; background:rgba(239,68,68,.15); color:#f87171;
    border:1px solid rgba(239,68,68,.4); border-radius:6px; padding:2px 10px;
    font-family:'JetBrains Mono',monospace; font-size:.7rem; font-weight:600;
    letter-spacing:1px; text-transform:uppercase;
}
.badge-warning {
    display:inline-block; background:rgba(234,179,8,.15); color:#fbbf24;
    border:1px solid rgba(234,179,8,.4); border-radius:6px; padding:2px 10px;
    font-family:'JetBrains Mono',monospace; font-size:.7rem; font-weight:600;
    letter-spacing:1px; text-transform:uppercase;
}
.badge-info {
    display:inline-block; background:rgba(59,130,246,.15); color:#60a5fa;
    border:1px solid rgba(59,130,246,.4); border-radius:6px; padding:2px 10px;
    font-family:'JetBrains Mono',monospace; font-size:.7rem; font-weight:600;
    letter-spacing:1px; text-transform:uppercase;
}
.summary-bar {
    background:#13131f; border:1px solid rgba(99,102,241,.15);
    border-radius:10px; padding:1rem 1.5rem;
    display:flex; gap:2rem; margin-top:1.5rem; flex-wrap:wrap;
}
.summary-item {
    display:flex; align-items:center; gap:.5rem;
    font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#94a3b8;
}
.summary-val { font-weight:700; color:#e2e8f0; }
[data-testid="stTabs"] button {
    font-family:'JetBrains Mono',monospace; font-size:.8rem; letter-spacing:.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ts-header">
  <div>
    <div class="ts-logo">🛡️ TrainSentry</div>
    <div class="ts-tagline">AI-Assisted ML Training Log Analyzer</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Inline control bar: upload + AI toggle + API key ─────────────────────────
ctl1, ctl2, ctl3 = st.columns([2.5, 1, 2])

with ctl1:
    uploaded = st.file_uploader(
        "📂 Upload training log CSV", type=["csv"],
        help="Required: epoch, train_loss, val_loss, train_accuracy, val_accuracy",
    )

with ctl2:
    st.markdown("<div style='padding-top:1.8rem'></div>", unsafe_allow_html=True)
    use_ai = st.toggle("🤖 AI Report", value=False,
                       help="Generate a Gemini-powered debugging narrative")

with ctl3:
    api_key_input = ""
    if use_ai:
        api_key_input = st.text_input(
            "Gemini API Key", type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            placeholder="AIza...",
            help="Free key at aistudio.google.com/app/apikey",
        )
    else:
        st.markdown("<div style='padding-top:1.8rem;font-size:.8rem;color:#475569'>"
                    "Enable AI Report to enter your Gemini API key.</div>",
                    unsafe_allow_html=True)

st.divider()

# ── Data loading ──────────────────────────────────────────────────────────────
run = analysis = df = None

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)
    _run = load_training_log(tmp_path)
    os.unlink(tmp_path)
    run = _run.__class__(
        name=Path(uploaded.name).stem,
        source=_run.source,
        frame=_run.frame,
    )
    analysis = analyze_run(run)
    df = run.frame
    st.session_state.pop("run", None)
    st.session_state.pop("analysis", None)

elif "run" in st.session_state:
    run      = st.session_state["run"]
    analysis = st.session_state["analysis"]
    df       = run.frame

else:
    # Landing page
    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in [
        (col1, "5",   "Detectable Issues"),
        (col2, "4",   "Dashboard Views"),
        (col3, "CSV", "Input Format"),
        (col4, "AI",  "Report Modes"),
    ]:
        col.markdown(f"""
        <div class="ts-metric">
            <div class="ts-metric-value">{val}</div>
            <div class="ts-metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Expected CSV format")
    st.code(
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy\n"
        "1,0.910,0.920,0.570,0.550\n"
        "2,0.760,0.780,0.660,0.640",
        language="csv",
    )

    sample_dir = Path("data")
    if sample_dir.exists():
        samples = list(sample_dir.glob("*.csv"))
        if samples:
            st.markdown("#### Or try a sample run")
            s_col, b_col = st.columns([3, 1])
            with s_col:
                choice = st.selectbox("Sample file", [p.name for p in samples],
                                      label_visibility="collapsed")
            with b_col:
                if st.button("▶ Load", type="primary", use_container_width=True):
                    _run = load_training_log(sample_dir / choice)
                    st.session_state["run"]      = _run
                    st.session_state["analysis"] = analyze_run(_run)
                    st.rerun()
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, len(df),                         "Epochs"),
    (c2, len(analysis.findings),          "Issues Found"),
    (c3, f"{analysis.best_val_loss:.4f}", "Best Val Loss"),
    (c4, analysis.best_epoch,             "Best Epoch"),
]:
    col.markdown(f"""
    <div class="ts-metric">
        <div class="ts-metric-value">{val}</div>
        <div class="ts-metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_curves, tab_findings, tab_report, tab_ai = st.tabs([
    "📈  Training Curves", "🔍  Findings", "📄  Rules Report", "🤖  AI Report",
])

CHART_COLORS = {
    "train": "#6366f1", "val": "#f43f5e",
    "grid": "rgba(99,102,241,0.08)", "bg": "rgba(0,0,0,0)",
}

def styled_fig():
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor=CHART_COLORS["bg"], paper_bgcolor=CHART_COLORS["bg"],
        font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=CHART_COLORS["grid"], linecolor="rgba(99,102,241,0.2)"),
        yaxis=dict(gridcolor=CHART_COLORS["grid"], linecolor="rgba(99,102,241,0.2)"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

# Tab 1
with tab_curves:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Loss Curves")
        fig = styled_fig()
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"],
            mode="lines+markers", name="Train Loss",
            line=dict(color=CHART_COLORS["train"], width=2.5), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"],
            mode="lines+markers", name="Val Loss",
            line=dict(color=CHART_COLORS["val"], width=2.5, dash="dash"), marker=dict(size=5)))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Accuracy Curves")
        fig2 = styled_fig()
        fig2.add_trace(go.Scatter(x=df["epoch"], y=df["train_accuracy"],
            mode="lines+markers", name="Train Acc",
            line=dict(color=CHART_COLORS["train"], width=2.5), marker=dict(size=5)))
        fig2.add_trace(go.Scatter(x=df["epoch"], y=df["val_accuracy"],
            mode="lines+markers", name="Val Acc",
            line=dict(color=CHART_COLORS["val"], width=2.5, dash="dash"), marker=dict(size=5)))
        fig2.update_layout(xaxis_title="Epoch", yaxis_title="Accuracy")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Train–Val Accuracy Gap")
    df2 = df.copy()
    df2["gap"] = df2["train_accuracy"] - df2["val_accuracy"]
    fig3 = styled_fig()
    fig3.add_trace(go.Bar(
        x=df2["epoch"], y=df2["gap"],
        marker_color=["#ef4444" if v > 0.1 else "#6366f1" for v in df2["gap"]],
        marker_line_width=0, name="Accuracy Gap",
    ))
    fig3.update_layout(xaxis_title="Epoch", yaxis_title="Train − Val Accuracy", bargap=0.3)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📊 Raw data"):
        st.dataframe(df.style.background_gradient(cmap="RdPu", axis=0),
                     use_container_width=True)

# Tab 2
with tab_findings:
    st.markdown(f"##### Findings — `{run.name}`")
    _icons = {Severity.CRITICAL: "🔴", Severity.WARNING: "🟡", Severity.INFO: "🔵"}
    _order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    _badge = {Severity.CRITICAL: "critical", Severity.WARNING: "warning", Severity.INFO: "info"}
    _card  = {Severity.CRITICAL: "finding-card-critical",
              Severity.WARNING:  "finding-card-warning",
              Severity.INFO:     "finding-card-info"}

    for f in sorted(analysis.findings, key=lambda x: _order.get(x.severity, 9)):
        icon  = _icons.get(f.severity, "⚪")
        badge = _badge.get(f.severity, "info")
        card  = _card.get(f.severity, "finding-card-info")
        st.markdown(f"""
        <div class="finding-card {card}">
            <div class="finding-title">{icon} {f.title}
                &nbsp;<span class="badge-{badge}">{f.severity.value}</span>
            </div>
            <div class="finding-evidence">📊 {f.evidence}</div>
            <div class="finding-rec">💡 {f.recommendation}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="summary-bar">
        <div class="summary-item">🔴 Critical <span class="summary-val">&nbsp;{analysis.critical_count}</span></div>
        <div class="summary-item">🟡 Warnings <span class="summary-val">&nbsp;{analysis.warning_count}</span></div>
        <div class="summary-item">✅ Best Epoch <span class="summary-val">&nbsp;{analysis.best_epoch}</span></div>
        <div class="summary-item">📉 Best Val Loss <span class="summary-val">&nbsp;{analysis.best_val_loss:.4f}</span></div>
        <div class="summary-item">🎯 Best Val Acc <span class="summary-val">&nbsp;{analysis.best_val_accuracy:.4f}</span></div>
    </div>""", unsafe_allow_html=True)

# Tab 3
with tab_report:
    report_md = build_markdown_report(analysis)
    st.markdown(report_md)
    st.download_button("⬇️ Download Report", data=report_md,
        file_name=f"{run.name}_report.md", mime="text/markdown")

# Tab 4
with tab_ai:
    st.markdown("##### 🤖 AI-Assisted Debugging Report")
    if not use_ai:
        st.info("Enable the **🤖 AI Report** toggle at the top of the page to generate a Gemini-powered debugging narrative.")
    else:
        resolved_key = api_key_input or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            st.warning("⚠️ Enter your Gemini API key in the field at the top of the page.")
        else:
            cache_key = f"ai_{run.name}_{len(analysis.findings)}"
            if cache_key not in st.session_state:
                with st.spinner("Generating AI report via Gemini..."):
                    st.session_state[cache_key] = generate_ai_report(
                        analysis, api_key=resolved_key)
            ai_report = st.session_state[cache_key]
            st.markdown(ai_report)
            dl_col, regen_col = st.columns([2, 1])
            with dl_col:
                st.download_button("⬇️ Download AI Report", data=ai_report,
                    file_name=f"{run.name}_ai_report.md", mime="text/markdown")
            with regen_col:
                if st.button("🔄 Regenerate"):
                    del st.session_state[cache_key]
                    st.rerun()