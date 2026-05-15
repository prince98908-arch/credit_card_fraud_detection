import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("credit_card_fraud_ann_model.h5")

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")

st.write("Fill transaction details below.")

# Inputs
time = st.number_input("Transaction Time", value=0.0)

v_values = []

feature_labels = [
    "Transaction Feature 1",
    "Transaction Feature 2",
    "Transaction Feature 3",
    "Transaction Feature 4",
    "Transaction Feature 5",
    "Transaction Feature 6",
    "Transaction Feature 7",
    "Transaction Feature 8",
    "Transaction Feature 9",
    "Transaction Feature 10",
    "Transaction Feature 11",
    "Transaction Feature 12",
    "Transaction Feature 13",
    "Transaction Feature 14",
    "Transaction Feature 15",
    "Transaction Feature 16",
    "Transaction Feature 17",
    "Transaction Feature 18",
    "Transaction Feature 19",
    "Transaction Feature 20",
    "Transaction Feature 21",
    "Transaction Feature 22",
    "Transaction Feature 23",
    "Transaction Feature 24",
    "Transaction Feature 25",
    "Transaction Feature 26",
    "Transaction Feature 27",
    "Transaction Feature 28",
]

for label in feature_labels:
    val = st.number_input(label, value=0.0)
    v_values.append(val)

amount = st.number_input("Transaction Amount", value=0.0)

if st.button("Predict Fraud"):

    input_data = np.array([[time] + v_values + [amount]])

    prediction = model.predict(input_data)

    score = prediction[0][0]

    if score > 0.5:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")

    st.write("Prediction Score:", round(float(score), 4))
