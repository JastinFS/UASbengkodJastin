import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_model.pkl")

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction")
st.write("Please fill in the following customer information:")

# === Customer Demographics ===
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (Months)", min_value=0, step=1)

# === Services ===
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)
OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)
DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)
TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)
StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)
StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

# === Contract & Billing ===
Contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0
)

# === Input DataFrame ===
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# === Prediction ===
if st.button("🔍 Predict Churn"):
    proba = model.predict_proba(input_data)
    churn_prob = proba[0][1] * 100  # Churn probability (%)

    if churn_prob >= 50:
        prediction = "CHURN"
        risk_level = "HIGH RISK"
    else:
        prediction = "NO CHURN"
        risk_level = "LOW RISK"

    st.markdown("---")
    st.subheader("📊 Prediction Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="📈 Churn Probability",
        value=f"{churn_prob:.2f}%"
    )

    col2.metric(
        label="⚠️ Risk Level",
        value=risk_level
    )

    col3.metric(
        label="🔮 Prediction",
        value=prediction
    )

    st.progress(churn_prob / 100)

    if prediction == "CHURN":
        st.error("❌ The customer is predicted to **CHURN**.")
    else:
        st.success("✅ The customer is predicted to **NOT CHURN**.")

    st.caption(
        "ℹ️ Note: Customers are classified as churn if the churn probability is ≥ 50%."
    )
