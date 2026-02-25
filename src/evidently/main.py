from fastapi import FastAPI
from pydantic import BaseModel
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset
import pandas as pd
import os

app = FastAPI()

REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "/app/data/train.csv")
reference_data = pd.read_csv(REFERENCE_DATA_PATH)[["review"]]
current_data = []

class PredictionLog(BaseModel):
    review: str

@app.post("/log")
def log_prediction(log: PredictionLog):
    current_data.append({"review": log.review})
    return {"status": "logged"}

@app.get("/drift")
def get_drift():
    if len(current_data) < 10:
        return {"status": "not enough data", "minimum": 10, "current": len(current_data)}
    
    current_df = pd.DataFrame(current_data)
    
    reference = Dataset.from_pandas(reference_data)
    current = Dataset.from_pandas(current_df)
    
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference, current)
    
    return result.dict()

@app.get("/healthz")
def health():
    return {"status": "ok"}