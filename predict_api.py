import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

try:
    model = joblib.load("artifacts/best_model_pipeline.pkl")
except:
    model = None

class PatientData(BaseModel):
    age: float; sex: float; cp: float; trestbps: float; chol: float
    fbs: float; restecg: float; thalach: float; exang: float
    oldpeak: float; slope: float; ca: float; thal: float

@app.post("/predict")
def predict(data: PatientData):
    if not model: raise HTTPException(status_code=500, detail="Model uninitialized")
    input_data = pd.DataFrame([data.model_dump()])
    prediction = int(model.predict(input_data)[0])
    confidence = float(model.predict_proba(input_data)[0][prediction])
    return {"heart_disease_risk_present": prediction, "confidence_score": confidence}
