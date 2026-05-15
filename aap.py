import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load Model
model = load_model("credit_card_fraud_ann_model.h5")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection App")

st.write("Enter transaction details below to check whether transaction is Fraud or Genuine.")

# Input Fields
time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

v_features = []

for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0)
    v_features.append(value)

# Prediction Button
if st.button("Predict"):

    # Prepare Input Data
    input_data = np.array([[time] + v_features + [amount]])

    # Prediction
    prediction = model.predict(input_data)

    result = prediction[0][0]

    # Output
    if result > 0.5:
        st.error("⚠️ Fraudulent Transaction Detected")
        st.write(f"Prediction Score: {result:.4f}")
    else:
        st.success("✅ Genuine Transaction")
        st.write(f"Prediction Score: {result:.4f}")
