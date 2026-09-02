import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="IntellexPredict",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "started" not in st.session_state:
    st.session_state.started = False

# -------------------------------------------------
# PREMIUM GLOBAL CSS
# -------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at 20% 30%, #0b0f1a, #06090f 60%);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

/* Navbar */
.navbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding: 10px 0px;
}

.logo {
    font-size:22px;
    font-weight:700;
}

.logo span {
    color:#8b5cf6;
}

.nav-btn {
    background:#1f2937;
    padding:8px 18px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.1);
}

/* Hero */
.hero {
    text-align:center;
    margin-top:100px;
    margin-bottom:120px;
}

.badge {
    display:inline-block;
    padding:8px 18px;
    border-radius:20px;
    background:rgba(139,92,246,0.15);
    border:1px solid rgba(139,92,246,0.4);
    font-size:14px;
    margin-bottom:30px;
}

.gradient-text {
    font-size:68px;
    font-weight:800;
    background: linear-gradient(90deg,#ffffff,#60a5fa,#a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height:1.1;
}

.subtext {
    color:#9ca3af;
    font-size:18px;
    margin-top:25px;
    max-width:750px;
    margin-left:auto;
    margin-right:auto;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#6366f1,#a855f7);
    color:white;
    padding:14px 32px;
    font-size:18px;
    border-radius:40px;
    border:none;
    transition:0.3s;
}

.stButton>button:hover {
    transform:scale(1.05);
    box-shadow:0 0 25px #8b5cf6;
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.08);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}

hr {
    border: 1px solid rgba(255,255,255,0.05);
}

/* ================= INPUT VISIBILITY FIX ================= */

/* All labels (fix invisible G1, G2 etc.) */
label, .stSlider label, .stNumberInput label, .stSelectSlider label {
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}

/* Slider min/max numbers */
.stSlider span {
    color: #cbd5e1 !important;
}

/* Number input styling */
.stNumberInput input {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* Remove ugly grey slider block */
.stSlider > div > div {
    background: transparent !important;
}

/* ================= EXTREME POLISH ================= */

/* Smooth fade animation */
.hero, .card {
    animation: fadeUp 0.8s ease forwards;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Premium slider track glow */
.stSlider div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg,#6366f1,#a855f7) !important;
}

/* Slider handle glow */
.stSlider div[data-baseweb="slider"] span {
    box-shadow: 0 0 15px #8b5cf6 !important;
}

/* Card hover glow upgrade */
.card:hover {
    box-shadow: 0 15px 50px rgba(139,92,246,0.3);
}

/* Navbar glow on hover */
.nav-btn:hover {
    box-shadow: 0 0 20px rgba(139,92,246,0.5);
    cursor: pointer;
}

/* Subtle page fade-in */
.stApp {
    animation: pageFade 0.6s ease-in;
}

@keyframes pageFade {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* ================= FIX PRIOR SUBJECT FAILURES INPUT ================= */

div[data-testid="stNumberInput"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background-color: rgba(255,255,255,0.08) !important;
    font-weight: 600 !important;
}

div[data-testid="stNumberInput"] {
    background-color: rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.pkl"

    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)

    return None

model = load_model()

if model is None:
    st.error("Model file not found.")
    st.stop()

# -------------------------------------------------
# NAVBAR
# -------------------------------------------------
st.markdown("""
<div class="navbar">
    <div class="logo">Intellex<span>Predict</span></div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
if not st.session_state.started:

    st.markdown("""
    <div class="hero">
        <div class="badge">✦ Advanced Neural Prediction Engine v2.0</div>
        <div class="gradient-text">
            Predict Academic <br> Outcomes with AI.
        </div>
        <div class="subtext">
            made by Tarundeep Singh | Powered by Lasso Regression | Ridge Regression | Premium AI Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Initialize Prediction →"):
        st.session_state.started = True
        st.rerun()

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
else:

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Academic Background")
        G1 = st.slider("First Term Grade (G1)", 0, 20, 10)
        G2 = st.slider("Second Term Grade (G2)", 0, 20, 10)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Behavioral Metrics")
        studytime = st.select_slider(
            "Daily Study Commitment",
            options=[1,2,3,4],
            value=2,
            format_func=lambda x: ["<2 hrs","2-5 hrs","5-10 hrs",">10 hrs"][x-1]
        )
        failures = st.number_input("Prior Subject Failures", 0, 5, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Analyze Student Performance"):

        X = np.array([[studytime, failures, G1, G2]])
        prediction = model.predict(X)[0]
        final_score = max(0, min(20, float(prediction)))

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📈 Predictive Analysis Result")

        if final_score >= 15:
            status = "🌟 Distinction Expected"
            color = "#22c55e"
            st.balloons()
        elif final_score >= 10:
            status = "✅ Pass Expected"
            color = "#3b82f6"
        else:
            status = "⚠️ At Risk"
            color = "#ef4444"

        m1, m2 = st.columns(2)

        with m1:
            st.metric("Final Grade (G3)", f"{final_score:.1f} / 20")

        with m2:
            st.markdown(f"<h3 style='color:{color};'>{status}</h3>", unsafe_allow_html=True)

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["G1","G2","Predicted G3"],
            y=[G1, G2, final_score],
            text=[G1, G2, round(final_score,1)],
            textposition="outside",
        ))
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        st.subheader("🔍 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": ["Study Time","Failures","G1","G2"],
            "Importance": model.coef_
        }).sort_values("Importance")

        fig2 = go.Figure(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation='h'
        ))

        fig2.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig2, use_container_width=True)

        st.success("Prediction analysis complete.")

    
    if st.button("⬅ Back to Home"):
        st.session_state.started = False
        st.rerun()

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Powered by Lasso Regression | Premium AI Dashboard")
