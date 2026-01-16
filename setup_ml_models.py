import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# Create mock data for training models
np.random.seed(42)
n_samples = 1000
data = {
    "age": np.random.randint(20, 80, n_samples),
    "resting_bp": np.random.randint(90, 180, n_samples),
    "cholesterol": np.random.randint(150, 350, n_samples),
    "max_heart_rate": np.random.randint(80, 200, n_samples),
    "oldpeak": np.round(np.random.uniform(0, 4, n_samples), 1)
}
df = pd.DataFrame(data)

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

print("Training Isolation Forest...")
iso_model = IsolationForest(contamination=0.1, random_state=42)
iso_model.fit(df)
joblib.dump(iso_model, 'models/isolation_forest.pkl')

print("Training Scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
joblib.dump(scaler, 'models/scaler.pkl')

print("ML Models created successfully in models/")
