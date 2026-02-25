from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import httpx
import torch
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource


KSERVE_URL = os.getenv("KSERVE_URL", "http://review-sentiment-predictor-00001-private.default.svc.cluster.local/predict")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/tokenizer")
JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "jaeger.monitoring.svc.cluster.local:4317")

provider = TracerProvider(
    resource=Resource.create({"service.name": "fastapi-gateway"})
)
exporter = OTLPSpanExporter(endpoint=JAEGER_ENDPOINT, insecure=True)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    ml["config"] = AutoConfig.from_pretrained(MODEL_DIR)
    yield
    ml.clear()

app = FastAPI(lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)

class Request(BaseModel):
    text: str

@app.post("/predict")
async def predict(req: Request):
    tokenizer = ml["tokenizer"]
    config = ml["config"]

    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)
    payload = {"input_ids": inputs["input_ids"].tolist()}

    async with httpx.AsyncClient() as client:
        response = await client.post(KSERVE_URL, json=payload, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    logits = torch.tensor(response.json()["logits"])
    probs = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    label = config.id2label[predicted_class]

    return {
        "label": label,
        "confidence": probs[0][predicted_class].item()
    }

@app.get("/healthz")
def health():
    return {"status": "ok"}