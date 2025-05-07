import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('mental_health_model.pkl')

st.title("Mental Health Prediction")

# Collect user input
age = st.slider("Age", 10, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
work_interfere = st.selectbox("Work Interference", ["Never", "Rarely", "Sometimes", "Often"])

# Map inputs to numeric (your model expects numbers)
gender_map = {"Male": 0, "Female": 1, "Other": 2}
family_history_map = {"Yes": 1, "No": 0}
work_interfere_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}

# Convert inputs to model format
input_data = np.array([
    age,
    gender_map[gender],
    family_history_map[family_history],
    work_interfere_map[work_interfere]
]).reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {'May have mental health condition' if prediction[0]==1 else 'Unlikely to have mental health condition'}")
