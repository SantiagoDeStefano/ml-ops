from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

KSERVE_URL = os.getenv("KSERVE_URL")
if not KSERVE_URL:
    raise RuntimeError("KSERVE_URL is not set. Put it in .env")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: TextRequest):
    payload = {"instances": [[5.1, 3.5, 1.4, 0.2]]}
    print("PREDICT HIT")
    try:
        response = requests.post(KSERVE_URL, json=payload)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    except Exception as e:
        print("ERROR TYPE:", type(e))
        print("ERROR DETAIL:", repr(e))
        raise HTTPException(status_code=500, detail=repr(e))

@app.get("/health")
def health():
    print("KSERVE_URL =", KSERVE_URL)
    return {"ok": True}
