import streamlit as st
import numpy as np
import pandas as pd
import pickle

#Load saved model and scaler
with open('NB_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("🏦 Bank Churn Prediction App")

# User Inputs
credit_score = st.slider("Credit Score", 300, 900, 650)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years with bank)", 0, 10, 3)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
products_number = st.selectbox("Number of Products", [1, 2, 3, 4])
credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)
country = st.selectbox("Country", ["France", "Germany", "Spain"])

# Convert categorical values
gender = 1 if gender == "Male" else 0
credit_card = 1 if credit_card == "Yes" else 0
active_member = 1 if active_member == "Yes" else 0
country_Germany = 1 if country == "Germany" else 0
country_Spain = 1 if country == "Spain" else 0

# Construct input array
input_data = np.array([[credit_score, gender, age, tenure, balance,
                        products_number, credit_card, active_member,
                        estimated_salary, country_Germany, country_Spain]])

# Scale the input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("❌ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")

        



