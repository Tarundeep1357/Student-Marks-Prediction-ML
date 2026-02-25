import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration for a professional look
st.set_page_config(
    page_title="AI Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "MAX Level" UI/UX
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    .prediction-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #1e293b;
        text-align: center;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return pickle.load(open("model.pkl", "rb"))
    return None

model = load_model()

# Sidebar content
with st.sidebar:
    # st.image("generated-icon.png", width=100)
    st.title("Navigation")
    st.info("This AI model uses Lasso Regression to predict final grades based on early academic performance.")
    st.divider()
    st.subheader("How to use")
    st.write("1. Enter student metrics in the main panel.")
    st.write("2. Click 'Analyze & Predict'.")
    st.write("3. View detailed insights and recommendations.")

# Header Section
st.markdown("<h1 class='prediction-header'>🎓 AI Student Success Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Leveraging Machine Learning to optimize educational outcomes</p>", unsafe_allow_html=True)

if model is None:
    st.error("🚨 **System Offline**: Model file not found. Please run the training script (`analysis.py`) to initialize the AI.")
    st.stop()

# Layout: 2 Columns for inputs
col_a, col_b = st.columns([1, 1], gap="large")

with col_a:
    st.subheader("📊 Academic Background")
    with st.container():
        G1 = st.slider("First Term Grade (G1)", 0, 20, 10, help="Initial internal assessment marks (0-20 scale)")
        G2 = st.slider("Second Term Grade (G2)", 0, 20, 10, help="Mid-term assessment marks (0-20 scale)")

with col_b:
    st.subheader("🧠 Behavioral Metrics")
    with st.container():
        studytime = st.select_slider(
            "Daily Study Commitment",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: ["< 2 hrs", "2-5 hrs", "5-10 hrs", "> 10 hrs"][x-1]
        )
        failures = st.number_input("Prior Subject Failures", min_value=0, max_value=5, value=0, help="Number of times the student has failed subjects in the past")

st.divider()

# Prediction Logic
if st.button("🚀 Analyze Student Performance"):
    # Prepare input
    X = np.array([[studytime, failures, G1, G2]])
    prediction = model.predict(X)[0]
    
    # Bound the result realistically
    final_score = max(0, min(20, float(prediction)))
    
    # Results Presentation
    st.markdown("<h2 style='text-align: center;'>Predictive Analysis Result</h2>", unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    
    # Visual Logic
    if final_score >= 15:
        status = "Distinction Expected"
        color = "green"
        emoji = "🌟"
        st.balloons()
    elif final_score >= 10:
        status = "Pass Expected"
        color = "blue"
        emoji = "✅"
    else:
        status = "At Risk"
        color = "red"
        emoji = "⚠️"

    with m1:
        st.metric("Final Grade (G3)", f"{final_score:.1f} / 20", delta=f"{final_score - ((G1+G2)/2):.1f} vs Avg", delta_color="normal")
    with m2:
        st.write(f"<p style='color:{color}; font-size: 24px; font-weight: bold; padding-top: 15px;'>{emoji} {status}</p>", unsafe_allow_html=True)
    with m3:
        st.progress(final_score / 20)

    # Detailed Feedback
    st.subheader("💡 Strategic Recommendations")
    if final_score >= 15:
        st.success("Targeting Top Tier: Maintain current study habits. Consider peer-tutoring to reinforce concepts.")
    elif final_score >= 10:
        st.info("Steady Progress: Focus on identifying weak areas in G1/G2 modules. Increasing study time by 1 hour could boost the final score by ~10%.")
    else:
        st.error("Critical Intervention Required: Schedule 1-on-1 sessions. Focus heavily on fundamental concepts from G1 and G2.")

    # Visualization of Input impact (Simple mock-up logic for UX)
    st.divider()
    st.subheader("📈 Performance Context")
    
    # Creating a small dataframe for a quick plot
    comparison_data = pd.DataFrame({
        "Stage": ["G1", "G2", "Predicted G3"],
        "Score": [G1, G2, final_score]
    })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Stage", y="Score", data=comparison_data, palette="viridis", ax=ax)
    ax.set_ylim(0, 20)
    ax.set_ylabel("Marks")
    ax.set_title("Grade Progression Path")
    st.pyplot(fig)

st.divider()
st.caption("Powered by Advanced Regression Analysis | Data Source: Student Performance Dataset")
