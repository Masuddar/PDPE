import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Disable DL for performance - TensorFlow is too slow to load
# Set this to True if you want to enable deep learning (requires waiting for TF to load)
ENABLE_DL = False

@st.cache_resource
def _load_dl_models():
    """Load and cache DL models (disabled by default for performance)"""
    if not ENABLE_DL:
        return None, None
        
    try:
        from tensorflow.keras.models import load_model
        scaler = joblib.load("models/scaler.pkl")
        # Try .keras format first, then .h5
        try:
            autoencoder = load_model("models/autoencoder.keras")
        except:
            autoencoder = load_model("models/autoencoder.h5", compile=False)
        return autoencoder, scaler
    except Exception as e:
        print(f"Warning: Could not load DL models: {e}")
        return None, None

def dl_anomaly_score(row):
    """Calculate DL anomaly score (using mock for performance)"""
    if not ENABLE_DL:
        # Fast mock implementation - returns low anomaly score
        # In production, this would use the actual autoencoder
        return 0.0
    
    autoencoder, scaler = _load_dl_models()
    
    # Fallback if models couldn't load
    if autoencoder is None or scaler is None:
        return 0.0
        
    x = scaler.transform(pd.DataFrame([row]))
    recon = autoencoder.predict(x, verbose=0)
    return float(((x - recon) ** 2).mean())
