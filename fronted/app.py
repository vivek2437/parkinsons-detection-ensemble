import streamlit as st
# import sounddevice as sd
import wavio
import requests
import io
import numpy as np

st.title("🎤 Parkinson's Voice Detection")

duration = st.slider("🎙 Recording duration (seconds)", 3, 10, 6)
sample_rate = 16000

if st.button("Start Recording"):
    st.info("Recording... Speak now 🎙")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    # Save the recording locally
    wavio.write("recorded_voice.wav", audio, sample_rate, sampwidth=2)
    st.success("✅ Recording complete! File saved as recorded_voice.wav")

    # Send to backend
    try:
        with open("recorded_voice.wav", "rb") as f:
            files = {"file": f}
            response = requests.post("http://127.0.0.1:8000/predict_audio", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                st.error(f"Error: {data['error']}")
            else:
                score = data["risk_score"]
                label = data["label"]

                st.subheader("🩺 Prediction Result")
                st.write(f"**Label:** {label}")
                st.write(f"**Risk Score:** {score:.2f}")

                # Confidence Meter
                confidence = round(score * 100, 1)
                if confidence < 40:
                    st.success(f"✅ Low Risk ({confidence}% confidence)")
                elif confidence < 70:
                    st.warning(f"⚠️ Moderate Risk ({confidence}% confidence)")
                else:
                    st.error(f"🚨 High Risk Detected ({confidence}% confidence)")

        else:
            st.error(f"Server Error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Request failed: {e}")
