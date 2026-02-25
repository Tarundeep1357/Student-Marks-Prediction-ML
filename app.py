import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ MODERN CSS ------------------
st.markdown("""
<style>

/* ----------- 3D Animated Gradient Background ----------- */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1e3c72);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
    overflow-x: hidden;
}

/* Gradient Animation */
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ----------- Glass 3D Cards ----------- */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.37),
        inset 0 0 0 1px rgba(255,255,255,0.05);
    transition: all 0.4s ease;
}

.card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 
        0 20px 40px rgba(0,0,0,0.6),
        inset 0 0 0 1px rgba(255,255,255,0.1);
}

/* ----------- Floating 3D Glow Orbs ----------- */
.glow {
    position: fixed;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    filter: blur(120px);
    opacity: 0.6;
    z-index: -1;
}

.glow1 {
    background: #3b82f6;
    top: -100px;
    left: -100px;
    animation: float1 10s infinite alternate ease-in-out;
}

.glow2 {
    background: #8b5cf6;
    bottom: -100px;
    right: -100px;
    animation: float2 12s infinite alternate ease-in-out;
}

@keyframes float1 {
    0% { transform: translate(0px, 0px); }
    100% { transform: translate(80px, 100px); }
}

@keyframes float2 {
    0% { transform: translate(0px, 0px); }
    100% { transform: translate(-100px, -60px); }
}

/* ----------- Buttons ----------- */
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #3b82f6);
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
    color: white;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #3b82f6;
}

/* ----------- Text Styling ----------- */
h1, h2, h3 {
    color: #f1f5f9;
}

.stMarkdown, .stText {
    color: #e2e8f0;
}

</style>

<!-- Floating Orbs -->
<div class="glow glow1"></div>
<div class="glow glow2"></div>

""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return pickle.load(open("model.pkl", "rb"))
    return None

model = load_model()

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("Navigation")
    st.info("Lasso Regression model predicting final grades.")
    st.divider()
    st.write("1. Enter student metrics.")
    st.write("2. Click Analyze.")
    st.write("3. View insights.")

# ------------------ HEADER ------------------
st.markdown("<h1 style='text-align:center;'>🎓 AI Student Success Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;'>Interactive ML Dashboard with Explainable AI</p>", unsafe_allow_html=True)

if model is None:
    st.error("Model file not found.")
    st.stop()

# ------------------ INPUT SECTION ------------------
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

# ------------------ PREDICTION ------------------
if st.button("🚀 Analyze Student Performance"):

    X = np.array([[studytime, failures, G1, G2]])
    prediction = model.predict(X)[0]
    final_score = max(0, min(20, float(prediction)))

    st.divider()
    st.subheader("📈 Predictive Analysis Result")

    # -------- STATUS LOGIC --------
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

    # -------- METRICS --------
    m1, m2 = st.columns(2)

    with m1:
        st.metric("Final Grade (G3)", f"{final_score:.1f} / 20")

    with m2:
        st.markdown(f"<h3 style='color:{color};'>{status}</h3>", unsafe_allow_html=True)

    # -------- CIRCULAR PROGRESS --------
    def circular_progress(value):
        html_code = f"""
        <div style="display:flex;justify-content:center;margin-top:20px;">
            <div style="
                width:200px;height:200px;
                border-radius:50%;
                background: conic-gradient(#3b82f6 {value*5}%, #334155 0%);
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:32px;
                font-weight:bold;
                color:white;">
                {value:.1f}
            </div>
        </div>
        """
        components.html(html_code, height=250)

    circular_progress(final_score)

    # ------------------ STRATEGIC RECOMMENDATION ------------------
    st.subheader("💡 Strategic Recommendations")

    if final_score >= 15:
        st.success("Maintain current performance. Consider advanced concept mastery.")
    elif final_score >= 10:
        st.info("Increase study time slightly and review weak modules.")
    else:
        st.error("Immediate academic intervention required.")

    # ------------------ INTERACTIVE CHART ------------------
    st.divider()
    st.subheader("📊 Grade Progression Path")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["G1","G2","Predicted G3"],
        y=[G1, G2, final_score],
        text=[G1, G2, round(final_score,1)],
        textposition="outside",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------ FEATURE IMPORTANCE ------------------
    st.divider()
    st.subheader("🔍 Feature Importance")

    feature_names = ["Study Time","Failures","G1","G2"]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.coef_
    })

    importance_df = importance_df.sort_values(by="Importance", key=abs)

    fig2 = go.Figure(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation='h'
    ))

    fig2.update_layout(template="plotly_dark", height=400)

    st.plotly_chart(fig2, use_container_width=True)

    # ------------------ WHY THIS PREDICTION ------------------
    st.divider()
    st.subheader("🧠 Why This Prediction?")

    if G2 > 15:
        st.success("Strong second term performance boosted prediction.")
    if studytime >= 3:
        st.info("High study commitment positively impacted score.")
    if failures == 0:
        st.success("No past failures significantly improved final outcome.")
    if failures >= 2:
        st.warning("Past failures negatively influenced prediction.")

st.divider()
st.caption("Powered by Lasso Regression | Interactive Explainable AI Dashboard")