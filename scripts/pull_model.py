import mlflow
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "review-sentiment-transformer")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
print(f"Pulling model from {model_uri}")

mlflow.artifacts.download_artifacts(model_uri, dst_path="models/")
print("Model pulled successfully.")