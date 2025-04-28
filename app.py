# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize the app
app = FastAPI(title="Health State Prediction API", version="3.2")

# Load Random Forest model
with open("RandomForest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load scaler if available (optional)
try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    scaler = None

# Request Body
class SensorData(BaseModel):
    heart_rate: float
    acc_variance: float

# Home Route
@app.get("/")
def read_root():
    return {"message": "🚀 Welcome to the Health State Prediction API v3.2"}

# Prediction Endpoint
@app.post("/predict")
def predict_health_state(data: SensorData):
    input_features = np.array([[data.heart_rate, data.acc_variance]])
    
    # Apply scaling if scaler exists
    if scaler:
        input_scaled = scaler.transform(input_features)
    else:
        input_scaled = input_features

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Mapping prediction to labels
    label_mapping = {
        0: "Healthy",
        1: "Warning",
        2: "Danger"
    }
    label = label_mapping.get(prediction, "Unknown")

    # Generate English heart condition message
    hr = data.heart_rate

    if hr < 55:
        heart_condition = "❗ Heart rate is lower than normal. Please check your health condition."
    elif 55 <= hr <= 92:
        heart_condition = "✅ Heart rate is within the normal range."
    elif 92 < hr <= 110:
        heart_condition = "⚠️ Slightly elevated heart rate. Monitoring is recommended."
    elif 110 < hr <= 135:
        heart_condition = "⚠️ Noticeably high heart rate. Consider resting or consulting a doctor."
    else:  # hr > 135
        heart_condition = "🚨 Critically high heart rate detected. Immediate medical attention is advised."

    return {
        "prediction": int(prediction),
        "state": label,
        "message": heart_condition
    }
