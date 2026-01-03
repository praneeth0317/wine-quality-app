import streamlit as st
import numpy as np
import pickle
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="centered"
)

# -----------------------------
# Load Model & Scaler Safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# -----------------------------
# Title & Description
# -----------------------------
st.title("🍷 Wine Quality Prediction App")

st.markdown("""
This application predicts whether a wine is **GOOD** or **AVERAGE**
based on its **chemical properties**.

📌 **Definition used in this app:**  
- Quality **≥ 7 → GOOD wine**  
- Quality **< 7 → AVERAGE wine**
""")

st.divider()

# -----------------------------
# Input Sliders
# -----------------------------
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
alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.0)

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Quality"):
    # Prepare input
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH,
        sulphates, alcohol
    ]])

    scaled_input = scaler.transform(input_data)

    # Model prediction
    prediction = model.predict(scaled_input)[0]
    confidence = model.predict_proba(scaled_input)[0][1]

    st.subheader(" Prediction Result")

    if prediction == 1:
        st.success("🍷 GOOD quality wine")
    else:
        st.warning("🍷 AVERAGE quality wine")

    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    st.caption(
        "Confidence shows how strongly the model believes this wine "
        "belongs to the predicted category."
    )

    st.divider()

    # -----------------------------
    # Explanation Section
    # -----------------------------
    st.subheader(" What Makes a Good Wine?")

    st.markdown("""
    Based on historical wine data, **GOOD quality wines** typically have:

    - 🍺 **Higher Alcohol** (above **11.5%**)
    - 🧪 **Higher Sulphates** (around **0.7 – 1.0**)
    - ⚗️ **Lower Volatile Acidity**
    - ⚖️ **Balanced pH** (≈ **3.2 – 3.4**)
    - 📉 **Slightly Lower Density**
    """)

    st.divider()

    # -----------------------------
    # Personalized Suggestions
    # -----------------------------
    st.subheader("🔎 How to Improve This Wine")

    suggestions = False

    if alcohol < 11.5:
        st.info("🔹 Increasing **alcohol content** may improve quality.")
        suggestions = True

    if sulphates < 0.7:
        st.info("🔹 Higher **sulphates** are commonly seen in good wines.")
        suggestions = True

    if volatile_acidity > 0.6:
        st.info("🔹 Lower **volatile acidity** often improves taste.")
        suggestions = True

    if not suggestions:
        st.success("✅ Your wine properties are already close to good-quality ranges!")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Built using Machine Learning & Streamlit 🍷")
