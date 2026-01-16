import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import os

# Create mock data for training
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

print("Training Scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
joblib.dump(scaler, 'models/scaler.pkl')

print("Training Autoencoder with current Keras...")
input_dim = df.shape[1]

# Build autoencoder using functional API (more compatible)
encoder_input = keras.Input(shape=(input_dim,))
encoded = layers.Dense(4, activation='relu')(encoder_input)
encoded = layers.Dense(2, activation='relu')(encoded)
decoded = layers.Dense(4, activation='relu')(encoded)
decoder_output = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = keras.Model(encoder_input, decoder_output)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=32, verbose=1)

# Save model in new format
autoencoder.save('models/autoencoder.keras')  # Use .keras format instead of .h5
print("Autoencoder saved to models/autoencoder.keras")

print("Models created successfully!")
