import pandas as pd
import numpy as np

def load_data(file):
    """Load CSV data."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        return f"Error loading file: {e}"

def validate_columns(df):
    """Ensure required columns exist."""
    required = ['patient_id', 'age', 'blood_type', 'diagnosis', 'last_visit_date']
    missing = [col for col in required if col not in df.columns]
    return missing

def preprocess_data(df):
    """Simple preprocessing for ML models (e.g., categorical encoding)."""
    df_encoded = df.copy()
    # Dummy encoding for diagnosis or blood type if needed
    # (In a real app, this would use the saved scaler/encoder)
    return df_encoded
