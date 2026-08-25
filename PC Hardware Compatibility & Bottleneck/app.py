"""
===============================================================
AI-Powered PC Hardware Compatibility & Bottleneck Analyzer
===============================================================
app.py - Streamlit Dashboard Application

Features:
  ✓ Sidebar dropdowns (auto-populated from training encoders)
  ✓ Multi-task prediction (Compatibility, Bottleneck %, Component)
  ✓ Metric cards with confidence scores
  ✓ Plotly gauge chart for bottleneck severity
  ✓ Pie chart for compatibility distribution
  ✓ Prediction history table
  ✓ GROQ-powered LLM recommendation engine
  ✓ Dark-themed professional UI

Run: streamlit run app.py
===============================================================
"""

import os
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import (
    load_artifacts,
    predict_hardware,
    get_llm_recommendation,
    bottleneck_color,
    bottleneck_label,
)

# ---------------------------------------------------------------
# PAGE CONFIG & THEME
# ---------------------------------------------------------------
st.set_page_config(
    page_title="PC Hardware AI Analyzer",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for dark theme and card styling
st.markdown("""
<style>
/* ---- Global dark background ---- */
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ---- Metric cards ---- */
.metric-card {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-card h4 {
    color: #9ca3af;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 0.4rem 0;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 0.3rem;
}

/* ---- Status badges ---- */
.badge-compatible   { color: #10b981; }
.badge-incompatible { color: #ef4444; }
.badge-cpu          { color: #f59e0b; }
.badge-gpu          { color: #6366f1; }
.badge-balanced     { color: #10b981; }

/* ---- Section headers ---- */
.section-header {
    border-left: 4px solid #6366f1;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* ---- Prediction history ---- */
.stDataFrame { font-size: 0.82rem; }

/* ---- Buttons ---- */
.stButton button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.2s;
}
.stButton button:hover {
    background: linear-gradient(135deg, #4f46e5, #4338ca);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99,102,241,0.4);
}

/* ---- Progress bars ---- */
.stProgress > div > div > div > div {
    background-color: #6366f1;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# LOAD ARTIFACTS (cached so they load once per session)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model and encoders…")
def load_all():
    return load_artifacts()

try:
    model, input_encoders, target_encoders, meta = load_all()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    load_error = str(e)

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
col_title, col_logo = st.columns([5, 1])
with col_title:
    st.markdown("""
    <h1 style='color:#6366f1; margin-bottom:0;'>🖥️ AI-Powered PC Hardware</h1>
    <h2 style='color:#e0e0e0; margin-top:0;'>Compatibility & Bottleneck Analyzer</h2>
    <p style='color:#6b7280;'>Multi-Task Deep Learning </p>
    """, unsafe_allow_html=True)
with col_logo:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🤖")

st.divider()

# ---------------------------------------------------------------
# ERROR STATE (model not trained yet)
# ---------------------------------------------------------------
if not model_loaded:
    st.error(f"⚠️ {load_error}")
    st.info("👉 Run `python train_model.py` to train the model, then restart this app.")
    st.stop()

# ---------------------------------------------------------------
# SIDEBAR – INPUT CONTROLS
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Hardware Configuration")
    st.markdown("*Select your PC components:*")
    st.markdown("---")

    # Auto-populate dropdown options from saved encoders
    cpu_options  = list(input_encoders["CPU_Model"].classes_)
    gpu_options  = list(input_encoders["GPU_Model"].classes_)
    mb_options   = list(input_encoders["Motherboard_Model"].classes_)
    ram_options  = list(input_encoders["RAM_Capacity"].classes_)
    res_options  = list(input_encoders["Target_Resolution"].classes_)

    sel_cpu = st.selectbox("🔧 CPU Model",         cpu_options, index=0, help="Select your processor")
    sel_gpu = st.selectbox("🎮 GPU Model",         gpu_options, index=0, help="Select your graphics card")
    sel_mb  = st.selectbox("🔌 Motherboard Model", mb_options,  index=0, help="Select your motherboard")
    sel_ram = st.selectbox("💾 RAM Capacity",       ram_options, index=0, help="Select RAM size & type")
    sel_res = st.selectbox("🖥️ Target Resolution", res_options, index=0, help="Target gaming/display resolution")

    st.markdown("---")

    predict_btn = st.button("🚀 Analyze Build", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"- **{len(cpu_options)}** CPU models")
    st.markdown(f"- **{len(gpu_options)}** GPU models")
    st.markdown(f"- **{len(mb_options)}** Motherboards")
    st.markdown("- Multi-Task ANN (3 outputs)")
    st.markdown("- Powered by TF/Keras")

# ---------------------------------------------------------------
# SESSION STATE – Prediction History
# ---------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------------
if predict_btn:
    with st.spinner("🤖 Running AI inference…"):
        result = predict_hardware(
            cpu_model=sel_cpu,
            gpu_model=sel_gpu,
            motherboard_model=sel_mb,
            ram_capacity=sel_ram,
            target_resolution=sel_res,
        )

    # Store in session history
    st.session_state.history.append({
        "Time":          datetime.datetime.now().strftime("%H:%M:%S"),
        "CPU":           sel_cpu,
        "GPU":           sel_gpu,
        "Motherboard":   sel_mb,
        "RAM":           sel_ram,
        "Resolution":    sel_res,
        "Compatible":    result["is_compatible"],
        "Bottleneck %":  f"{result['bottleneck_pct']:.1f}%",
        "Component":     result["bottleneck_component"],
    })

    # Store latest result for display below
    st.session_state.latest = result
    st.session_state.latest_build = {
        "cpu": sel_cpu, "gpu": sel_gpu, "mb": sel_mb,
        "ram": sel_ram, "res": sel_res,
    }

# ---------------------------------------------------------------
# RESULTS DISPLAY
# ---------------------------------------------------------------
if "latest" in st.session_state:
    r   = st.session_state.latest
    bld = st.session_state.latest_build

    # ---- Row 1: Four KPI cards ----
    st.markdown('<div class="section-header"><h3>📋 Prediction Results</h3></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    # Card 1: Compatibility
    compat_color = "#10b981" if r["is_compatible"] == "Compatible" else "#ef4444"
    compat_icon  = "✅" if r["is_compatible"] == "Compatible" else "❌"
    c1.markdown(f"""
    <div class="metric-card">
        <h4>Compatibility</h4>
        <p class="value" style="color:{compat_color};">{compat_icon} {r['is_compatible']}</p>
        <p class="sub">Confidence: {r['compatibility_conf']*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Card 2: Bottleneck %
    b_color = bottleneck_color(r["bottleneck_pct"])
    c2.markdown(f"""
    <div class="metric-card">
        <h4>Bottleneck %</h4>
        <p class="value" style="color:{b_color};">{r['bottleneck_pct']:.1f}%</p>
        <p class="sub">{bottleneck_label(r['bottleneck_pct'])}</p>
    </div>
    """, unsafe_allow_html=True)

    # Card 3: Primary Bottleneck Component
    comp_icon = {"CPU Bottleneck": "🔧", "GPU Bottleneck": "🎮", "General / Low": "✨"}.get(
        r["bottleneck_component"], "⚙️")
    c3.markdown(f"""
    <div class="metric-card">
        <h4>Bottleneck Component</h4>
        <p class="value" style="font-size:1.3rem;">{comp_icon} {r['bottleneck_component']}</p>
        <p class="sub">Confidence: {r['component_confidence']*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Card 4: Overall Build Score (derived metric)
    score = max(0, 100 - r["bottleneck_pct"]) * (1 if r["is_compatible"] == "Compatible" else 0.1)
    score_color = "#10b981" if score >= 70 else ("#f59e0b" if score >= 40 else "#ef4444")
    c4.markdown(f"""
    <div class="metric-card">
        <h4>Build Score</h4>
        <p class="value" style="color:{score_color};">{score:.0f}/100</p>
        <p class="sub">Performance index</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Row 2: Gauge + Pie + Progress Bars ----
    col_gauge, col_pie, col_bars = st.columns([2, 2, 2])

    # Gauge Chart
    with col_gauge:
        st.markdown("#### 🎯 Bottleneck Gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=r["bottleneck_pct"],
            title={"text": "Bottleneck %", "font": {"size": 14, "color": "#e0e0e0"}},
            delta={"reference": 15, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                "bar":  {"color": bottleneck_color(r["bottleneck_pct"])},
                "bgcolor": "#1f2937",
                "bordercolor": "#374151",
                "steps": [
                    {"range": [0,  15], "color": "#064e3b"},
                    {"range": [15, 30], "color": "#78350f"},
                    {"range": [30, 100], "color": "#7f1d1d"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": r["bottleneck_pct"],
                },
            },
            number={"suffix": "%", "font": {"color": "#e0e0e0", "size": 28}},
        ))
        gauge.update_layout(
            paper_bgcolor="#0e1117",
            font={"color": "#e0e0e0"},
            height=280,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(gauge, use_container_width=True)

    # Pie Chart – Component distribution from history
    with col_pie:
        st.markdown("#### 🥧 Bottleneck Component")
        if len(st.session_state.history) > 0:
            hist_df = pd.DataFrame(st.session_state.history)
            comp_counts = hist_df["Component"].value_counts()
            pie = go.Figure(go.Pie(
                labels=comp_counts.index,
                values=comp_counts.values,
                hole=0.45,
                marker=dict(colors=["#6366f1", "#f59e0b", "#10b981", "#ef4444"]),
                textfont=dict(color="white"),
            ))
            pie.update_layout(
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font={"color": "#e0e0e0"},
                legend=dict(font=dict(color="#e0e0e0")),
                height=280,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("Run predictions to see component distribution.")

    # Progress Bars – Confidence scores
    with col_bars:
        st.markdown("#### 📊 Confidence Scores")
        st.markdown(f"**Compatibility Confidence**")
        st.progress(r["compatibility_conf"])
        st.caption(f"{r['compatibility_conf']*100:.1f}% — {r['is_compatible']}")

        st.markdown(f"**Component Confidence**")
        st.progress(r["component_confidence"])
        st.caption(f"{r['component_confidence']*100:.1f}% — {r['bottleneck_component']}")

        st.markdown(f"**Build Health Score**")
        st.progress(min(score / 100, 1.0))
        st.caption(f"{score:.0f}/100 — Overall performance index")

    # ---- Row 3: Selected Build Summary ----
    st.markdown("---")
    st.markdown("#### 🖥️ Selected Build Summary")
    build_cols = st.columns(5)
    build_items = [
        ("CPU", bld["cpu"], "🔧"),
        ("GPU", bld["gpu"], "🎮"),
        ("Motherboard", bld["mb"], "🔌"),
        ("RAM", bld["ram"], "💾"),
        ("Resolution", bld["res"], "🖥️"),
    ]
    for col, (label, value, icon) in zip(build_cols, build_items):
        col.markdown(f"""
        <div class="metric-card">
            <h4>{icon} {label}</h4>
            <p style="font-size:0.85rem; margin:0; color:#c9d1d9;">{value}</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- Row 4: LLM Smart Recommendation ----
    st.markdown("---")
    st.markdown('<div class="section-header"><h3>🤖 AI Smart Recommendation </h3></div>',
                unsafe_allow_html=True)

    if st.button("💡 Generate AI Recommendation", use_container_width=False):
        with st.spinner("Consulting Your Request..."):
            recommendation = get_llm_recommendation(
                cpu=bld["cpu"], gpu=bld["gpu"],
                motherboard=bld["mb"], ram=bld["ram"],
                resolution=bld["res"], prediction=r,
            )
        st.markdown(recommendation)

else:
    # Landing state – no prediction yet
    st.info("👈 Select your hardware components in the sidebar, then click **Analyze Build**.")

    # Show example capabilities
    st.markdown("---")
    st.markdown("### 🚀 What This Tool Does")
    cap1, cap2, cap3 = st.columns(3)
    with cap1:
        st.markdown("""
        <div class="metric-card">
            <h4>✅ Compatibility Check</h4>
            <p style="font-size:0.9rem;">Checks if your CPU, GPU, and Motherboard combination is physically and electrically compatible.</p>
        </div>
        """, unsafe_allow_html=True)
    with cap2:
        st.markdown("""
        <div class="metric-card">
            <h4>⚡ Bottleneck Analysis</h4>
            <p style="font-size:0.9rem;">Predicts the bottleneck percentage — how much one component is holding back the others.</p>
        </div>
        """, unsafe_allow_html=True)
    with cap3:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 AI Recommendations</h4>
            <p style="font-size:0.9rem;">GROQ-powered LLaMA-3.3-70B provides tailored upgrade advice and optimization tips.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# PREDICTION HISTORY TABLE
# ---------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown('<div class="section-header"><h3>🕐 Prediction History</h3></div>', unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.history)

    # Color-code Compatible column
    def color_compat(val):
        color = "#10b981" if val == "Compatible" else "#ef4444"
        return f"color: {color}; font-weight: bold"

    styled = history_df.style.applymap(color_compat, subset=["Compatible"])
    st.dataframe(styled, use_container_width=True, height=min(300, 55 + len(history_df) * 35))

    col_dl, _ = st.columns([1, 4])
    with col_dl:
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export History CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4b5563; font-size:0.8rem; padding:1rem 0;">
    AI-Powered PC Hardware Compatibility & Bottleneck Analyzer &nbsp;|&nbsp;
    ANN & Deep Learning Final Project &nbsp;|&nbsp;
    Multi-Task TensorFlow/Keras + GROQ LLaMA-3.3-70B
</div>
""", unsafe_allow_html=True)