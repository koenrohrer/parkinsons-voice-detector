import numpy as np
import pandas as pd
import joblib

# Load the preprocessing objects
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Define the feature names (in the exact order used during training)
FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP: PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR',
    'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

# Prompt User for Values
print("Enter the following measurements:")
values = []
for feat in FEATURES:
    while True:
        try:
            v = float(input(f" {feat}: "))
            values.append(v)
            break
        except ValueError:
            print("Please enter a valid number!")
        
# Scale and Predict
x = np.array([values])
x_scaled = scaler.transform(x)
pred = model.predict(x_scaled)[0]

print("\n Prediction:",
      "⚠️Parkinson's detected" if pred == 1 else "✅No Parkinson's detected")