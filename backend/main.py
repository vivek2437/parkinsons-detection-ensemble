from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils.feature_extraction import extract_features                       

import librosa
import numpy as np
import joblib
import io
import traceback
from pydub import AudioSegment
from pydub.utils import which

app = FastAPI()

# ---------------------------------------------------------------
# 🔧 Enable CORS for frontend
# ---------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for development)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# 🔧 Set FFmpeg / FFprobe paths explicitly
# ---------------------------------------------------------------
AudioSegment.converter = r"C:\Users\ASUS\Downloads\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\ASUS\Downloads\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin\ffprobe.exe"

# ---------------------------------------------------------------
# 🔁 Load models at startup
# ---------------------------------------------------------------
print("🔁 Loading models...")
xgb_model = joblib.load("models/xgb_model.pkl")
lgb_model = joblib.load("models/lgb_model.pkl")
cat_model = joblib.load("models/cat_model.pkl")
meta_model = joblib.load("models/meta_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("✅ All models loaded successfully!")

# ---------------------------------------------------------------
# 🧪 Endpoint for audio prediction
# ---------------------------------------------------------------
@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # 1️⃣ Read uploaded audio
        contents = await file.read()

        # 2️⃣ Convert any audio format to standard WAV (mono, 16kHz)
        audio = AudioSegment.from_file(io.BytesIO(contents), format="wav")
        audio = audio.set_channels(1).set_frame_rate(16000)

        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # 3️⃣ Load audio with librosa
        y, sr = librosa.load(wav_io, sr=16000)

        # 4️⃣ Extract features
        features = extract_features(y, sr).reshape(1, -1)

        # 5️⃣ Scale features
        X_scaled = scaler.transform(features)

        # 6️⃣ Model predictions
        xgb_pred = xgb_model.predict_proba(X_scaled)[:, 1]
        lgb_pred = lgb_model.predict_proba(X_scaled)[:, 1]
        cat_pred = cat_model.predict_proba(X_scaled)[:, 1]

        # 7️⃣ Stacking for ensemble
        stack_input = np.column_stack((xgb_pred, lgb_pred, cat_pred))
        final_pred = meta_model.predict_proba(stack_input)[:, 1][0]

        # 8️⃣ Return result
        result = {
            "risk_score": float(final_pred),
            "label": "Parkinson’s Positive" if final_pred > 0.5 else "Healthy"
        }
        return result

    except Exception as e:
        print("❌ Error in /predict_audio:", str(e))
        traceback.print_exc()
        return {"error": str(e)}

# ---------------------------------------------------------------
# Optional: root endpoint for quick check
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Parkinson's voice prediction API is running."}
