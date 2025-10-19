import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan tools
rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoders.pkl")

# Load dataset untuk referensi fitur
df = pd.read_csv("me_cfs_vs_depression_dataset (1).csv")

# Pastikan target dihapus
if 'diagnosis' in df.columns:
    df = df.drop(columns=['diagnosis'])

st.title("🧠 ME/CFS vs Depression Classifier")
st.write("Masukkan data pasien di bawah ini untuk memprediksi kemungkinan ME/CFS atau Depression.")

# Input user
user_input = {}
for col in df.columns:
    if df[col].dtype == 'object':
        options = list(df[col].unique())
        user_input[col] = st.selectbox(f"{col}", options)
    else:
        min_val = 0.0
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        if "age" in col.lower():
            user_input[col] = st.number_input(f"{col}", min_value=0, max_value=int(max_val), step=1)
        else:
            user_input[col] = st.slider(f"{col}", float(min_val), float(max_val), float(mean_val), step=0.1)

# Buat DataFrame dari input
input_df = pd.DataFrame([user_input])

# Encode kolom kategorikal
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

# Scale data
input_scaled = scaler.transform(input_df)

# Prediksi
if st.button("🔍 Prediksi"):
    prediction = rf.predict(input_scaled)[0]
    prediction_proba = rf.predict_proba(input_scaled)[0]
    decoded_pred = label_encoders['diagnosis'].inverse_transform([prediction])[0]

    st.subheader("🩺 Hasil Prediksi")
    st.write(f"**Prediksi:** {decoded_pred}")

    # Tampilkan probabilitas semua kelas
    st.write("**Probabilitas:**")
    for cls, prob in zip(rf.classes_, prediction_proba):
        st.write(f"- {cls}: {prob:.2f}")
