import streamlit as st
import numpy as np
import pickle
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
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
# App Title & Standards
# -----------------------------
st.title("🍷 Wine Quality Prediction Application")

st.markdown("""
### 📌 Wine Quality Standards Used
- ✅ **Good Wine** → Quality **≥ 7**
- ⚠️ **Average Wine** → Quality **= 6**
- ❌ **Bad Wine** → Quality **≤ 5**

Values close to the standard are treated as **Average**.
""")

st.divider()

# -----------------------------
# User Inputs
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
# Prediction Block (SAFE)
# -----------------------------
if st.button("🔍 Predict Wine Quality"):

    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH,
        sulphates, alcohol
    ]])

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    # -----------------------------
    # Prediction Result
    # -----------------------------
    st.subheader("📊 Prediction Result")

    if prediction == 2:
        st.success("🍷 GOOD quality wine")
    elif prediction == 1:
        st.warning("🍷 AVERAGE quality wine")
    else:
        st.error("🍷 BAD quality wine")

    st.markdown("### 📈 Confidence Levels")
    st.write(f"❌ Bad Wine: **{probabilities[0]*100:.2f}%**")
    st.write(f"⚠️ Average Wine: **{probabilities[1]*100:.2f}%**")
    st.write(f"✅ Good Wine: **{probabilities[2]*100:.2f}%**")

    st.divider()

    # -----------------------------
    # WHAT MAKES A GOOD WINE (YOUR REQUIRED TEXT)
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
    # Suggestions Section
    # -----------------------------
    st.subheader("🔎 How to Improve This Wine")

    improvement = False

    if alcohol < 11.5:
        st.info("🔹 Increase **alcohol** content to improve quality.")
        improvement = True

    if sulphates < 0.7:
        st.info("🔹 Higher **sulphates** are common in good wines.")
        improvement = True

    if volatile_acidity > 0.6:
        st.info("🔹 Lower **volatile acidity** improves wine quality.")
        improvement = True

    if not improvement:
        st.success("✅ This wine already meets good-quality standards!")

st.divider()
st.caption("Machine Learning based Wine Quality Prediction 🍷")
