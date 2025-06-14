import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("AI-Powered Diabetes Risk Predictor")

# Sidebar inputs
st.sidebar.header("User Input Features")

def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0.0, 846.0, 79.0)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 20.0)
    diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 21, 100, 33)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Load dataset from local file
df = pd.read_csv("diabetes.csv")

# Prepare features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display output
st.subheader("Prediction Result")
st.write("🔴 At Risk of Diabetes" if prediction[0] == 1 else "🟢 Not at Risk of Diabetes")

st.subheader("Prediction Probability")
st.write(f"Confidence: {np.max(prediction_proba) * 100:.2f}%")
