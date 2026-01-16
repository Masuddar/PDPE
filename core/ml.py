import pandas as pd
import joblib
import streamlit as st

@st.cache_resource
def load_ml_model():
    """Load and cache the Isolation Forest model"""
    try:
        return joblib.load("models/isolation_forest.pkl")
    except:
        return None

def ml_anomaly_check(row):
    iso_model = load_ml_model()
    if iso_model is None:
        return "NORMAL"  # Fallback
    
    df = pd.DataFrame([row])
    return "ANOMALOUS" if iso_model.predict(df)[0] == -1 else "NORMAL"
