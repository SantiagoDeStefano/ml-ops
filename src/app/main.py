from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification
import torch
import os

MODEL_DIR = os.getenv("MODEL_DIR", "/mnt/models/model")

ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    ml["model"].eval()
    yield
    ml.clear()

app = FastAPI(lifespan=lifespan)

class Request(BaseModel):
    input_ids: list

@app.post("/predict")
def predict(req: Request):
    model = ml["model"]
    input_ids = torch.tensor(req.input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    return {"logits": outputs.logits.tolist()}

@app.get("/healthz")
def health():
    return {"status": "ok"}