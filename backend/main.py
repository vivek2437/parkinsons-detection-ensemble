import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
from utils.feature_extraction import extract_features

st.title("🎤 Parkinson's Voice Detection")

uploaded_file = st.file_uploader("Upload a voice sample (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    # Load and extract features
    audio_data, sr = librosa.load(audio_path, sr=None)
    features = extract_features(audio_data, sr).reshape(1, -1)

    # Load model
    model = joblib.load("models/xgb_model.pkl")

    # Predict
    proba = model.predict_proba(features)[0][1]
    pred = "🟢 Normal" if proba < 0.5 else "🔴 Parkinson’s Detected"

    # Show output
    st.audio(uploaded_file)
    st.subheader(f"Prediction: {pred}")
    st.write(f"Probability: **{proba:.4f}**")

else:
    st.info("Please upload a voice file to begin.")
