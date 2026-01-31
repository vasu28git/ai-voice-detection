import base64
import io
import os
import tempfile
import numpy as np
import librosa
from config import SAMPLE_RATE, SKIP_SAMPLES, WINDOW_SAMPLES, N_MFCC


def decode_base64_audio(audio_base64: str) -> bytes:
    try:
        audio_bytes = base64.b64decode(audio_base64)
        return audio_bytes
    except Exception as e:
        raise ValueError(f"Invalid Base64 encoding: {str(e)}")


def load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        finally:
            os.unlink(tmp_path)
        
        if len(y) > SKIP_SAMPLES:
            y = y[SKIP_SAMPLES:]
        if len(y) > WINDOW_SAMPLES:
            y = y[:WINDOW_SAMPLES]
        else:
            y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        
        return y
    except Exception as e:
        raise ValueError(f"Failed to load audio: {str(e)}")


def extract_features(audio_bytes: bytes) -> np.ndarray:
    y = load_audio_from_bytes(audio_bytes)
    
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)  
    mfcc_std = np.std(mfcc, axis=1)    
    
    rms = np.mean(librosa.feature.rms(y=y)[0])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)[0])

    features = np.hstack([
        mfcc_mean,
        mfcc_std,
        rms,
        zcr,
        centroid
    ])
    
    return features
