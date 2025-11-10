import numpy as np
import librosa

def extract_features(audio_data, sr):
    
    # Pitch
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    # RMS
    rms = librosa.feature.rms(y=audio_data)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio_data)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)   # 13 features
    mfcc_stds = np.std(mfcc, axis=1)     # 13 features

    # Combine features: pitch, rms, zcr, first 13 MFCC means, first 3 MFCC stds
    features = np.hstack([
        pitch_mean, pitch_std,
        rms_mean, rms_std,
        zcr_mean, zcr_std,
        mfcc_means,           # 13
        mfcc_stds[:3]         # 3 → total 22
    ])

    return features.astype(np.float32)
