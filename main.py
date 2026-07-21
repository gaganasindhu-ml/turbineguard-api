from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('turbineguard_model.pkl')
scaler = joblib.load('turbineguard_scaler.pkl')
kmeans = joblib.load('turbineguard_kmeans.pkl')
feature_columns = joblib.load('feature_columns.pkl')

rollavg_slope_cols = [col for col in feature_columns if 'rollavg' in col or 'slope' in col]

@app.get("/")
def health_check():
    return {"status": "TurbineGuard API is running"}

class EngineReading(BaseModel):
    features: Dict[str, float]

@app.post("/predict")
def predict_rul(reading: EngineReading):
    input_df = pd.DataFrame([reading.features])
    input_df = input_df[feature_columns[:-1]]

    cluster_input = scaler.transform(input_df[rollavg_slope_cols])
    input_df['health_cluster'] = kmeans.predict(cluster_input)[0]
    input_df = input_df[feature_columns]

    predicted_rul = float(model.predict(input_df)[0])

    if predicted_rul < 20:
        risk = "Critical"
    elif predicted_rul < 50:
        risk = "Warning"
    elif predicted_rul < 90:
        risk = "Moderate"
    else:
        risk = "Healthy"

    return {"predicted_RUL": round(predicted_rul, 1), "risk_tier": risk}
