import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
from utils.feature_extraction import extract_features

st.title("🎤 Parkinson's Voice Detection")

uploaded_file = st.file_uploader("Upload a voice sample (.wav format)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    # Load audio
    audio_data, sr = librosa.load(audio_path, sr=None)
    features = extract_features(audio_data, sr).reshape(1, -1)

    model = joblib.load("models/xgb_model.pkl")
    proba = model.predict_proba(features)[0][1]
    pred = "🟢 Normal" if proba < 0.5 else "🔴 Parkinson’s Detected"

    st.audio(uploaded_file)
    st.write(f"**Prediction:** {pred}")
    st.write(f"**Probability:** {proba:.4f}")

