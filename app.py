import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Prediksi Churn Telco", layout="centered")
st.title("📊 Prediksi Churn Pelanggan Telco")
st.write("Silakan isi seluruh data pelanggan berikut:")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (bulan)", min_value=0, step=1)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

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

if st.button("🔍 Prediksi Churn"):
    proba = model.predict_proba(input_data)
    churn_prob = proba[0][1] * 100  # Probabilitas churn (%)

    # Threshold klasifikasi
    if churn_prob >= 50:
        prediction = "CHURN"
        risiko = "RISIKO TINGGI"
    else:
        prediction = "TIDAK CHURN"
        risiko = "RISIKO RENDAH"

    st.markdown("---")
    st.subheader("📊 Hasil & Ringkasan")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="📈 Probabilitas Churn",
        value=f"{churn_prob:.2f}%"
    )

    col2.metric(
        label="⚠️ Tingkat Risiko",
        value=risiko
    )

    col3.metric(
        label="🔮 Prediksi",
        value=prediction
    )

    st.progress(churn_prob / 100)

    if prediction == "CHURN":
        st.error("❌ Pelanggan diprediksi **CHURN**")
    else:
        st.success("✅ Pelanggan diprediksi **TIDAK CHURN**")

    st.caption("📌 Catatan: Pelanggan dikategorikan churn jika probabilitas ≥ 50%.")

