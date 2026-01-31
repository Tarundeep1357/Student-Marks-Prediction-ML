'''
Students Marks Prediction- Streamlit Application

This file:
1. Loads the trained  ML model
2. Creates a web interface for user input
3. Predicts final marks.
4. Displays result in real-time.
'''

import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Student Marks Predictor")

studytime = st.number_input("Study time (hours per day)", min_value=1, max_value=24)
failures = st.number_input("Number of past failures", min_value=0, max_value=5)
G1 = st.number_input("1st interal Marks", min_value=0, max_value=20)
G2 = st.number_input("2nd interal Marks", min_value=0, max_value=20)

if st.button("Predict Marks"):
    X = np.array([[studytime, failures, G1, G2]])
    prediction = model.predict(X)
    st.success(f"Predicted Final Marks (G3): {prediction[0]:.2f}")

st.caption("Note: Previous Internal marks (G1 AND G2) have the highest impact on final marks (G3) prediction.")





