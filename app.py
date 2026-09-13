import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model, scaler, and columns
# -----------------------------
model = joblib.load("knn_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# If columns.pkl contains a DataFrame, convert it to a list of column names
if isinstance(expected_columns, pd.DataFrame):
    expected_columns = expected_columns.columns.tolist()
elif isinstance(expected_columns, pd.Index):
    expected_columns = expected_columns.tolist()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.markdown("Provide the following details to predict the risk of heart disease.")

# User Inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0,
    6.0,
    1.0,
    0.1
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    # Create raw input
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns exactly as during training
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # Optional probability (only if supported)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_input)[0][1] * 100
        st.write(f"**Estimated Risk Score:** {probability:.1f}%")