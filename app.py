import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🍷 Wine Quality Prediction App")
st.write("Predict whether a wine is **Good** or **Bad**")

# Input fields
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.7)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.0)
chlorides = st.slider("Chlorides", 0.01, 0.3, 0.08)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 75.0, 15.0)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 300.0, 46.0)
density = st.slider("Density", 0.990, 1.005, 0.997)
pH = st.slider("pH", 2.5, 4.0, 3.3)
sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6)
alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)

# Prediction
if st.button("Predict Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                             residual_sugar, chlorides, free_sulfur_dioxide,
                             total_sulfur_dioxide, density, pH,
                             sulphates, alcohol]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.success("🍷 This is a GOOD quality wine")
    else:
        st.error("🍷 This is a BAD quality wine")
