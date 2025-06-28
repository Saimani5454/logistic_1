import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
with open("logistic_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.caption("Enter passenger information to predict survival odds.")

with st.form("predict_form"):
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 29)
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children aboard", 0, 10, 0)
    fare = st.slider("Fare", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Manual one-hot encoding
    sex_male = 1 if sex == "male" else 0
    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0

    features = pd.DataFrame([[
        pclass, age, sibsp, parch, fare,
        sex_male, embarked_Q, embarked_S
    ]], columns=[
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
        'Sex_male', 'Embarked_Q', 'Embarked_S'
    ])

    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]
    prediction = model.predict(features_scaled)[0]

    st.subheader("🧠 Prediction Result")
    st.write(f"**Survival Probability:** {prob:.2%}")
    if prediction == 1:
        st.success("🎉 Passenger likely **survived**.")
    else:
        st.error("💀 Passenger likely **did not survive**.")
